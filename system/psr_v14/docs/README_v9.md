# v9: causal, near-real-time procedure step recognition on IndustReal

Procedure step recognition (PSR) on the [IndustReal](https://github.com/TimSchoonbeek/IndustReal) dataset
(Schoonbeek et al., WACV 2024): from egocentric RGB video of a toy-car assembly, report each completed assembly step
(e.g. "Install front chassis") as soon as it happens, using only past and current frames.

v9 combines **action recognition (AR)** with **per-part visual verifiers** in a **causal probabilistic state filter**.
It is evaluated with the unmodified official IndustReal evaluator and compared with the paper's best baseline, B3.

## Results (test set, official evaluator)

POS = procedure order similarity, F1 = step F1, delay = average detection delay τ (lower is better).

| System | All recordings POS / F1 / delay | Recordings with errors POS / F1 / delay |
|---|---|---|
| **v9 + predicted AR** (deployable) | **0.804 / 0.893 / 7.6 s** | **0.700 / 0.817 / 9.0 s** |
| v9 + GT AR (oracle, diagnostic only) | 0.833 / 0.918 / 5.7 s | 0.718 / 0.851 / 5.6 s |
| B3, IndustReal paper (Table 4) | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s |

- **v9 + predicted AR beats B3 on all recordings** (POS +0.007, F1 +0.010) and is about 3x faster (7.6 s vs 22.4 s).
- **On recordings with errors** it is much faster and ties on F1 (+0.001), but **trails on POS (−0.031)**.
- **GT AR** uses the dataset's ground-truth action labels. It shows the ceiling with perfect action recognition and is
  **not deployable performance**.
- Recordings without errors: 0.895 / 0.960 / 6.4 s (predicted AR), 0.935 / 0.978 / 5.8 s (GT AR).

**Reproducibility check**: running `run_pipeline.py` from this folder from scratch (new localizer crops, both
verifiers, evaluation) reproduced both test rows above exactly, both the summary and the per-video POS / F1.

### What "recordings with errors" means

The official protocol scores only correct steps against `PSR_labels.csv`. "Recordings with errors" is the same metric
on the 15 test videos where the worker made a mistake. **Neither B3 nor v9's PSR output reports mistake steps**
("Incorrectly installed X"); the separate frame-level mistake detector is described in the next section;
the IndustReal paper defines no mistake-detection metric for PSR, and its evaluation code never reads
`PSR_labels_with_errors.csv`. As an extra (non-official) number, `eval_psr.py` also scores against that file
(`with_errors_labels_all`): 0.763 / 0.867 with predicted AR, 0.792 / 0.893 with GT AR. Each missed mistake counts as a
missed step there.

| Mistake detector (frame level, vs `error_labels.json`, test) | AUC | AP (chance) |
|---|---|---|
| All error frames (1,087) | 0.941 | 0.599 (0.041) |

## Mistake detection (added module)

v9 itself reports only correct steps (as B3). A separate, causal **mistake detector** flags frames where the assembly
shows an execution error. It is evaluated against the frame-level **error-state labels** published with
Schoonbeek et al., *Supervised Representation Learning Towards Generalizable Assembly State Recognition* (RA-L 2024;
`third_party/asr_error_labels/error_labels.json`, Apache-2.0): 3,347 labelled error frames from the IndustReal
recordings in four categories (missing component, incorrect placement, incorrect orientation, part-level error).

**Detector**: the [CLS, mean-patch] features that v9's verifier `verif_base_r336` already computes for the localizer
crop → MLP (256 hidden units) → causal mean of the last 5 scores. No extra backbone pass; ≈ 1 ms on the CPU.
Trained on the train + val recordings only.

**Test set** (IndustReal test recordings, 32 videos; frames per crop, every 2nd frame; frozen before the test run,
which was run once):

| Error category | Error frames | AUC | AP (chance) | Balanced AP* |
|---|---|---|---|---|
| All | 1,087 | **0.941** | **0.599** (0.041) | – |
| Missing component | 732 | 0.942 | 0.549 (0.028) | 0.945 |
| Incorrect placement | 24 | 0.784 | 0.002 (0.001) | 0.650 |
| Incorrect orientation | 169 | 0.982 | 0.688 (0.007) | 0.985 |
| Part-level error | 162 | 0.918 | 0.042 (0.006) | 0.867 |

\*Balanced AP: each error frame against one random normal frame (chance 0.5, mean of 20 draws), the scale of the ASR
paper's Fig. 8. **Not the same protocol**: the paper compares each image with a synthetic image of the annotated
*intended* state (ground truth); our detector does not know the intended state. For orientation the paper's best
model reaches ≈ 95 (ResNet) / ≈ 93 (ViT-S), for part-level ≈ 73 / ≈ 69, for placement ≈ 71 / ≈ 74, for missing ≈ 100
(read from their Fig. 8).

Labelling for training and scoring (`psr/mistakes.py`): a crop within ±1 frame of a labelled error frame is an error;
a crop more than 150 frames from every labelled error frame and from every frame that `PSR_labels_raw.csv` marks as
incorrectly installed is normal; everything else is ignored. The file's own `split` field differs from IndustReal's
PSR split and is ignored; splits always follow IndustReal's folders. Because the frames near an error are ignored,
AUC and AP above are optimistic: an alarm shortly before or after an error is not counted against the detector.

**As a live alarm** (test, threshold 0.42 fixed on train + val; computed from `results/mistake_scores_test.npz`):
- Frame level (scored frames only): precision 0.48, recall 0.68.
- Error episodes (labelled error frames ≤ 150 frames apart merged into one episode; caught if the alarm fires within
  ±30 frames of it): **91 of 104 caught**. Missing component 81 / 87, incorrect orientation 6 / 7, part-level 4 / 9,
  incorrect placement 0 / 1.
- False alarms: 59 of the 221 alarm segments are more than 150 frames from any labelled error frame and from any
  frame `PSR_labels_raw.csv` marks as incorrectly installed. That is about one every 2.5 minutes of video
  (90,054 test frames at 10 fps).

**What this is and is not**
- An **additional, frame-level** metric. It is not part of the official IndustReal PSR protocol, and B3 reports no
  number on it.
- It says *that* the assembly currently looks wrong (and is strongest for orientation and missing-component errors);
  it does not output the PSR step "Incorrectly installed X". Turning these alarms into PSR mistake events was tried and
  did not work: on train + val (oracle AR, nested 4-fold) it caught 2 of the 19 "Incorrectly installed" events with
  8 false alarms (see `EXPERIMENT_LOG` notes in
  the development repository), so the official PSR outputs above are unchanged.
- Rare categories have very low AP at their real frequency (placement: 24 test frames), so expect false alarms.
- Development: variants (logistic regression / MLP, with or without v9's tracked state as input, smoothing 1/5/9)
  were compared on train + val with 4 subject folds by the mean per-category AP; the MLP with smoothing 5 was frozen.
  `results/mistakes_trainval_oof.json` holds those out-of-fold numbers.

```bash
python extract_mistake_features.py test      # features of the verifier backbone (needs cache/crops_hr/test)
python eval_mistakes.py --split test         # -> results/mistakes_test.json, results/mistake_scores_test.npz
python tools/train_mistake_detector.py oof   # train+val cross-validation (needs features for train and val)
python tools/train_mistake_detector.py final # retrain model/mistake_detector.pkl
```

## How it works

```
frame t ──► 1. localizer (DINOv2-small patch matching) finds the assembly, crops it        (every 2nd frame)
        ──► 2. two DINOv2-base verifiers: P(part k attached) for the 10 parts
action t ─► 3. AR top-1 action (predicted or GT); fastening actions are "triggers"
        ──► 4. step model: which step did this trigger most likely complete?   (naive Bayes + logistic regression)
        ──► 5. causal state filter over 11-part assembly states:
                 triggers move belief along likely steps; each verifier reading reweights states,
                 weighted by view quality; only legal procedure steps are allowed
        ──► 6. report a step when the new state is very probable, stable for 3 readings,
               and the changed parts are visually confirmed over the last 6 readings
```

Compared with B3, which detects one of 22 whole-assembly states with a detector and accumulates confidence, v9:
- **checks each part separately**, so it handles part combinations not seen in training;
- **fuses action recognition and vision** in one filter: actions say *when* a step probably happened, vision confirms *what*;
- **keeps several weighted state hypotheses** instead of one guess;
- **weights evidence by view quality** (sharpness, hand occlusion).

Every decision at frame t uses only frames, AR outputs and verifier readings up to t (no future frames, no smoothing
over the future). Ground-truth PSR labels are read only after a video is fully processed, for scoring.

**Latency**: PSR side end-to-end ≈ 41 ms per frame on an RTX 4060 Laptop GPU (mean 40.8, median 30.8, p95 76.6 ms over
3,600 frames; `results/latency_v9.json`), including frame decoding, localizer, both verifiers and the filter.
**Not included**: the AR model's own compute time (predicted AR is read from precomputed causal predictions).

## Repository layout

```
eval_psr.py            evaluation: causal tracker + official metrics, --ar predicted | gt
run_pipeline.py        full inference on a split: crops -> verifiers -> eval (PSR + mistakes)
extract_mistake_features.py   verifier-backbone features for the mistake detector -> cache/errfeat_v9_<split>
eval_mistakes.py       mistake detector vs error_labels.json (frame level)
extract_crops.py       step 1: localizer crops (every 2nd frame)            -> cache/crops/<split>
extract_crops_hr.py    step 1b: the same boxes at 336 px                    -> cache/crops_hr/<split>
predict_verifier.py    step 2: verifier probabilities                      -> cache/<tag>_<split>
train_verifier.py      verifier model definition and training script
psr/                   tracker, state filter, step models, AR providers, localizer, official-eval wrapper
model/                 frozen configuration, step models (config.json, tables.json, lr.pkl), mistake_detector.pkl, SHA256SUMS
weights/               verifier weights (not in git, see below)
cache/                 localizer prototypes (shipped); generated crops and predictions (ignored)
third_party/industreal_psr/   official IndustReal psr_utils.py + procedure_info.json (unmodified, Apache-2.0)
third_party/asr_error_labels/ error_labels.json of the ASR paper (unmodified, Apache-2.0)
tools/train_mistake_detector.py  trains model/mistake_detector.pkl on train + val
tools/build_step_models.py    rebuilds model/tables.json and model/lr.pkl from train + val
demo/                  interactive web demo (demo_v9.py + ui/v9.html)
results/               evaluation outputs (JSON)
```

## Setup

1. **Python environment** (tested: Python 3.10, PyTorch 2.5.1 + CUDA 12.1, Windows 11):
   ```bash
   pip install -r requirements.txt
   ```
   The DINOv2 backbones are downloaded from Hugging Face on first use.

2. **IndustReal dataset**: download it from the authors and point `INDUSTREAL_DATA` to its root, which must contain
   `train/train_annotations/`, `validation/val_annotations/` and `test/test_annotations/` (one folder per recording
   with `rgb/`, `AR_labels.csv`, `PSR_labels.csv`, ...). Default: `./data`.
   ```bash
   export INDUSTREAL_DATA=/path/to/IndustReal
   ```

3. **Verifier weights** (2 × 348 MB, too large for a normal git push): place them in `weights/`:
   - `weights/verif_base_r336.pt` (SHA256 `fe509b183537684a4be54d962ff5a2c9df406aceb4173ce1d03c9b6c8c76709b`)
   - `weights/verif_base_r336_e6u6.pt` (SHA256 `381c6a3a164e5db3e33f0a0f6f26c984271a529c4fa77c910fee6b2969d03658`)

   Download them from this repository's GitHub Release, or retrain them (see below).

4. **Predicted AR** (for the deployable setting): one folder per recording with an `ar.jsonl` holding one JSON row per
   frame, e.g. `{"f": 0, "top": [["background", 0.25], ["take_objects", 0.05], ...]}`, computed causally (frame t only
   from frames ≤ t). The predictions used for the results above come from a separate causal verb × noun TCN + VideoMAE
   AR model (not part of this repository).

## Run the test evaluation

```bash
python run_pipeline.py --split test --ar-dir /path/to/ar_predictions
```

This extracts crops, runs both verifiers and evaluates with predicted AR and with GT AR (oracle). Outputs:
`results/v9_predictedAR_test.json` and `results/v9_gtAR_test.json` (summary + per-video results and predicted steps).

To re-run only the evaluation once the verifier outputs exist in `cache/`:
```bash
python eval_psr.py --split test --ar predicted --ar-dir /path/to/ar_predictions
python eval_psr.py --split test --ar gt
```

## Demo

```bash
python demo/demo_v9.py --preload        # needs PSR_AR_DIR=/path/to/ar_predictions and the val + test results
```
Open http://127.0.0.1:8766. It plays a recording and shows, frame by frame: predicted-AR top-5 vs AR ground truth,
the assembly state from v9 + predicted AR, v9 + GT AR and the PSR ground truth, a timeline, step-by-step delays
(paired with the official evaluator's rules) and per-frame latency. Ground-truth mistakes are marked for reference
(display only). A "Mistake detector" panel shows the causal P(error) per frame, an alarm when it exceeds the
threshold (env `PSR_MISTAKE_THR`, default 0.42 = best frame F1 on train + val out-of-fold scores), and two timeline lanes: labelled error frames from `error_labels.json`
(display only) and the detector's alarms. Verifiers can run live on the GPU or from `cache/` (the mistake detector
needs `cache/errfeat_v9_<split>` in cached mode).

## Training (how the frozen model was built)

All training used the **train + val** recordings only; the test set was never used for training, tuning or selection.

- **Verifiers**: DINOv2-base, 336 px crops, per-part binary heads, labels from `PSR_labels.csv`.
  ```bash
  python extract_crops.py train,val && python extract_crops_hr.py train,val 336
  VBACK=base VRES=336 python train_verifier.py final                          # -> verif_base_r336
  VBACK=base VRES=336 VEPOCHS=6 VUNFREEZE=6 python train_verifier.py final    # -> verif_base_r336_e6u6
  ```
  (writes `cache/<tag>_final.pt`; not re-run for this release.)
- **Step models**: `python tools/build_step_models.py out/` rebuilds `tables.json` and `lr.pkl`; verified to be
  byte-identical to `model/`.
- **Filter settings** (`model/config.json`): chosen by nested 4-fold cross-validation over train + val subjects.

## Fairness and limitations

- Causal inference, unmodified official metrics, no test data in training or selection.
- **PSR-level mistake detection is not solved**: v9 reports 0 of the 19 "Incorrectly installed" test events (same as
  B3). The added frame-level mistake detector (above) flags visible errors well but is not converted into these events.
  Incorrectly installed wheels look like correct ones from this camera, and incorrect brackets/screws look absent;
  train + val contain only 19 such events.
- The test set was scored for several system versions during development; v9 was the best of three versions that
  were each tested once, so its test score is slightly optimistic.
- Another paper (STORM-PSR) quotes B3 as F1 0.891 / τ 21.0 s instead of the WACV paper's 0.883 / 22.4 s; against that
  figure v9's F1 lead is 0.002.
- Delays do not include the AR model's compute time.
- `psr/core.py` and `psr/vision.py` differ from the frozen copies listed in `model/SHA256SUMS` only in how data,
  official-code and cache paths are configured (environment variables instead of fixed local paths).

## Pushing to GitHub

The verifier weights are ignored by git (`.gitignore`). Either attach them to a GitHub Release, or track them with
Git LFS (each file is 348 MB, above GitHub's 100 MB limit for normal files):
```bash
git lfs install
git lfs track "weights/*.pt"      # then remove the weights/*.pt line from .gitignore
```

## Exported per-frame predictions

`results/psr_predictions/` holds v9's live predictions (predicted AR, deployable setting) for all 48 val + test
videos, in the same layout as the AR export: one `<video>/psr.jsonl` per video, one row per frame
(`f`, `t`, reported `state` as 11 bits in COMPONENTS order, `new` steps reported at that frame, `new_with_errors`,
filter `top`-3 states with posterior, view quality `q`), plus `predictions.json` with each video's reported steps.
Regenerate with:

    python tools/export_predictions.py --ar-dir <dir with <video>/ar.jsonl>

The step lists are identical to `results/v9_predictedAR_{val,test}.json`. Val was used for tuning, so only the
test numbers are fair.
