# v14: causal, near-real-time procedure step recognition on IndustReal

v14 recognises assembly/maintenance steps in IndustReal egocentric videos as they happen, using only past frames, and
raises mistake alerts (wrong order, skipped step, incorrect install). It is v13 plus the official IndustReal
assembly state detector (ASD) as extra evidence in the state filter.

```
frame (10 fps) ─ even frames ─ DINOv2-small localizer ─ 336-px crop ─ 2 × DINOv2-base part verifiers ─┐
               ─ odd frames  ─ official IndustReal ASD (YOLOv8m, whole-assembly state) ───────────────┤
live action recognition (per-frame top-1 action) ─────────────────────────────────────────────────────┤
                                                      causal state filter (NB + LR step model)
                                                      ├─ completed steps, reported as soon as confirmed
                                                      └─ mistake alerts (task-graph rules + end-of-recording check)
```

## Results (IndustReal test split, official evaluator, single run of the frozen system)

POS / F1 / average delay. **Live AR** = causal action-recognition predictions (deployable).
**GT AR** = ground-truth action labels revealed causally; an oracle, for diagnosis only, never deployable.

| System | All recordings | Recordings with errors | Without errors |
|---|---|---|---|
| B3, IndustReal paper (Table 4) | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s | – |
| v13, live AR | 0.802 / 0.899 / 7.5 s | 0.747 / 0.839 / 9.0 s | 0.851 / 0.953 / 6.1 s |
| **v14, live AR** | **0.812 / 0.904 / 6.3 s** | **0.747 / 0.839 / 7.3 s** | **0.870 / 0.960 / 5.4 s** |
| v14, GT AR (oracle, diagnostic) | 0.803 / 0.913 / 5.0 s | 0.728 / 0.858 / 5.0 s | 0.869 / 0.962 / 5.0 s |

95 % bootstrap intervals over recordings (v14, live AR): all-recordings POS 0.73–0.89, error-recordings POS 0.63–0.85.

**Mistake detection** (test, 32 annotated mistakes, ±30 s matching):

| System | Found | Precision | False alarms / h |
|---|---|---|---|
| Baseline: official IndustReal PSR on ASD + same rules | 4 / 32 | 21 % | 6.0 |
| v13, live AR | 7 / 32 | 50 % | **2.8** |
| **v14, live AR** | **9 / 32** | **53 %** | 3.2 |
| v14, GT AR (oracle) | 7 / 32 | 50 % | 2.8 |

v14 by type (live AR): wrong order 4/4, skipped step 2/10, incorrect install 3/18.

**Latency** (RTX 4060 Laptop, 3,600 val frames, decode + ASD on odd frames + localizer and both verifiers on even
frames + filter): median 35.5 ms, p95 78.6 ms, mean 50.4 ms, peak GPU 602 MB. The action-recognition model's own
compute is **not** included (its outputs are read from files).

**Causality**: cutting val recordings at 96 points changes no output before the cut (0 mismatches).

Full outputs: [`results/`](results/) ([`results/README.md`](results/README.md)).

## How v14 was chosen

v14 adds one thing to v13 (`psr/v14.py`): where the ASD's most confident box (confidence ≥ 0.3) names an assembly
state, belief states with a different installed/absent pattern are down-weighted (weight 1, ε 0.02). The ASD runs on
odd frames. Everything else is v13 (`psr/v11.py`, unchanged).

1. **Dev data: val only.** The ASD was trained on train, so train is in-sample for it.
2. **Pre-registered grid** (`selection/v14/asd_grid.py`): weight {0.3, 1, 3} × confidence {0.3, 0.5, 0.7} × ε
   {0.02, 0.05, 0.1} × rate {every frame, odd frames}. Criterion: v13's criterion on val with live AR + with true AR.
   A config was eligible only if live-AR val mistakes were not worse than v13's (`selection/v14/asd_select.py`).
3. **Latency rule.** The every-frame argmax had p95 103.6 ms (> 100 ms), so the best odd-frame config was used, as
   pre-registered.
4. **Freeze, then one test run.** Nothing changed afterwards. Log: [`docs/EXPERIMENT_LOG_v11_v14.md`](docs/EXPERIMENT_LOG_v11_v14.md).

v13's own settings (`tau` 0.05, `theta` 0.6, `d0` 12) were chosen earlier by a pre-registered train+val selection with
simulated live AR (`selection/v13/`).

## Repository layout

| Path | What it is |
|---|---|
| `psr/` | The system: state filter (`filter_tracker.py`), v13 additions (`v11.py`), ASD evidence (`v14.py`), step models, localizer/verifier code, official-metric wrapper |
| `model/` | Frozen v14 package: `config.json` (incl. `asd`), step-model tables and LR, `SHA256SUMS` (incl. ASD weights), `FROZEN_AT`. `mistake_detector.pkl` is v9's frame-level detector, unused |
| `eval_v14.py` | Runs v14 causally on a split and scores it with the official evaluator. `eval_v13.py` = v13 |
| `cache/` | Verifier outputs and ASD outputs for val and test, localizer prototypes, exemplar bank. Lets the metrics be recomputed on a CPU |
| `selection/` | Development harness (out-of-fold verifiers, leave-one-subject-out step models); `v13/` simulator, grid and selection; `v14/` ASD grid, saved grid outputs and selection |
| `tools/` | `run_asd.py`, `causality_check.py`, `latency_bench.py`, `bootstrap_ci.py`, `verify_frozen.py`, step-model builders |
| `run_pipeline.py`, `extract_crops*.py`, `predict_verifier.py`, `train_verifier.py` | Pipeline from video frames (GPU + verifier weights) |
| `third_party/` | Official IndustReal PSR metric code and procedure definition, ASR error labels (unmodified, Apache 2.0) |
| `results/` | Outputs of the frozen v14 run: test (live and GT AR), val, latency, logs |
| `docs/` | Experiment log excerpt (v11–v14); v9 design notes and figures |

## What is not in this repository

| Item | Needed for | Where to get it |
|---|---|---|
| IndustReal dataset (frames + annotations) | everything (labels for scoring, frame counts) | the IndustReal authors, under the dataset's terms. Set `INDUSTREAL_DATA` |
| Live AR predictions, `<video>/ar.jsonl` (val + test) | live-AR results, selection, causality, latency | a causal verb × noun TCN action-recognition model (teammate's work, not published here). `--ar-dir` / `LIVE_AR_DIR` |
| Teammate's `assist` package (task graph + mistake rules) | mistake alerts and scoring | not published here. Set `TEAMMATE_AR` to its repository root |
| ASD weights `asd_best_IndustRealandSynthetic.pt` (311 MB) | recomputing `cache/asd_*`, latency benchmark | `assembly_state_detection_model_weights.zip`, IndustReal dataset page (data.4tu.nl, Apache-2.0). Place in `weights/` |
| Verifier weights `verif_base_r336.pt`, `verif_base_r336_e6u6.pt` (348 MB each) | latency benchmark, pipeline from frames | too large for git; attach to a GitHub Release (or Git LFS) and place in `weights/` |

## Reproduce

Tested with Python 3.10 on Windows 11 (CUDA 12.1, RTX 4060 Laptop). Install `requirements.txt`, `pyyaml`, and for
the ASD `pip install ultralytics==8.3.40 --no-deps`.

```bash
export INDUSTREAL_DATA=/path/to/IndustReal       # contains train/, validation/, test/
export TEAMMATE_AR=/path/to/teammate_ar          # repository with assist/
export LIVE_AR_DIR=/path/to/live_ar              # <video>/ar.jsonl for val + test
```

**0. Check the frozen package:**

```bash
python tools/verify_frozen.py
```

**1. Test and val metrics** (CPU, about a minute each; cached verifier and ASD outputs). Writes `results/reproduced/`
for comparison with `results/v14_*.json`.

```bash
python eval_v14.py --split test --ar predicted --ar-dir $LIVE_AR_DIR --out results/reproduced
python eval_v14.py --split test --ar gt --out results/reproduced
python eval_v14.py --split val --ar predicted --ar-dir $LIVE_AR_DIR --out results/reproduced
python tools/bootstrap_ci.py results/reproduced/v14_predictedAR_test.json
```

**2. Selection** (from the saved grid outputs, a few minutes on CPU):

```bash
cd selection
python v14/asd_select.py         # SELECTED {"W": 1.0, "CMIN": 0.3, "EPS": 0.05, "RATE": 1}; best odd-frame: EPS 0.02, RATE 2
python v13/selection.py          # v13's filter settings
```

Rebuild the v14 grid (9 shards, about 15 minutes in parallel): `for i in 0 1 2 3 4 5 6 7 8; do python v14/asd_grid.py $i 9; done`.

**3. Causality check** (val, 96 cut points; expects 0 mismatches):

```bash
python tools/causality_check.py $LIVE_AR_DIR
```

**4. Latency** (GPU; verifier and ASD weights in `weights/`):

```bash
python tools/latency_bench.py --ar-dir $LIVE_AR_DIR
```

**5. Recompute the ASD outputs** (GPU, about 1.5 h for val + test):

```bash
python tools/run_asd.py val
python tools/run_asd.py test
```

## Fairness and caveats

- **No test data** in training, tuning or selection. Ground-truth PSR labels are read only after a recording has been
  processed, for scoring. GT AR numbers are an oracle diagnostic, never deployable performance.
- **Small margins.** v14's gains over v13 and B3 are inside the bootstrap intervals; on error recordings POS and F1
  equal v13's.
- **One metric worse than v13:** false alarms 3.2 vs 2.8 per hour (8 vs 7 false alerts over 2.5 h).
- **Dev evidence is thin.** Val has 16 recordings. In nested leave-one-subject-out CV the all-recordings POS/F1 gain was
  flat and one fold added false alarms. With the final verifiers on val (in-sample for them) ASD made results worse.
- **ASD.** Its authors picked its checkpoint on val mAP (val mildly optimistic). The original project rule kept ASD out
  of the inference path; it was allowed as an extra input by the project owner for v14. Seven exploratory configs were
  seen before the pre-registration was written.
- **Test-set history.** The test split was evaluated 13 times over the project; v14 once.
- **Latency** excludes the action-recognition model (68.6 ms/frame reported by its author on an RTX 2080 Ti).

## Credits and licences

See [`NOTICE.md`](NOTICE.md). The IndustReal metric code, procedure definition, ASR error labels and ASD model are
Apache 2.0 (Schoonbeek et al.). DINOv2 backbones are downloaded from Hugging Face at run time.
