# End-to-end step recognition (paper Tables 4, 6, 7)

v14 is the state tracker of STARE (`system/psr_v14`); v13 is v14 without the state detector.

Rules fixed before any measurement: [`PREREGISTRATION.md`](PREREGISTRATION.md). Each AR label reaches v14 only when
the live system would have it (measured compute included), v14 is re-tuned on validation for each input with the
pre-registered grid, and test is scored once per input with the official evaluator. No future frames.

## Results (32 test recordings, official evaluator, one test score each)

| System | All: POS / F1 / delay | With errors: POS / F1 / delay | Mistakes found | False alarms / h |
|---|---|---|---|---|
| **v14 + TCN, end to end** (labels 0.8 s late) | **0.819 / 0.907 / 6.5 s** | **0.747 / 0.839 / 7.5 s** | 9 / 32 | 3.6 |
| v14 + ensemble, end to end (labels 2.5 s late, finals 1.95 s after an action's end) | 0.764 / 0.879 / 7.2 s | 0.661 / 0.797 / 8.0 s | 9 / 32 | 5.2 |
| B3, IndustReal paper | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s | – | – |
| *Ablations ([`PREREGISTRATION_ablations.md`](PREREGISTRATION_ablations.md)), same protocol:* | | | | |
| v13 + TCN = v14 without the state detector, end to end (labels 0.8 s late) | 0.809 / 0.903 / 7.7 s | 0.747 / 0.839 / 9.4 s | 7 / 32 | 3.2 |
| v14 without action recognition (vision only, drift prior 1e-5 chosen on val; `PREREGISTRATION_noar.md`) | 0.772 / 0.893 / 7.9 s | 0.689 / 0.815 / 7.6 s | 8 / 32 | 3.2 |
| v14 + annotated actions (oracle AR, diagnostic only; labels 0.1 s late = v14 compute) | 0.814 / 0.915 / 5.0 s | 0.728 / 0.858 / 5.0 s | 7 / 32 | 2.8 |

Both re-tunes chose the frozen v14 setting (ASD weight 1, confidence 0.3, eps 0.02, odd frames):
[`results/retune_tcn.json`](results/retune_tcn.json), [`results/retune_ens.json`](results/retune_ens.json).

Without the state detector (v13), the tracker does the same per-frame work as the earlier v9 (localizer + verifiers on
even frames, filter every frame, no ASD), so its delay uses that measured p95 of 68.5 ms
([`results/latency_v9.json`](results/latency_v9.json), `latency_v9.py`, which runs the v14 code with the settings in
`metrics/psr/model_v9_settings`; the network disk was slow during that run, but any p95 between 52 and 152 ms gives the
same 8-frame shift). A v9 + TCN row (0.835 / 0.911 / 8.0 s) was also scored; it is not in the paper.
The oracle re-tune chose ASD weight 3, confidence 0.5, eps 0.02 ([`results/retune_oracle.json`](results/retune_oracle.json));
with zero shift the same code reproduces the unshifted oracle run (`test_once.py check-oracle`).

## Measured compute (RTX 2080 Ti, test recordings 03_assy_0_1, 10_main_0_1, 17_main_0_1; [`results/latency.json`](results/latency.json))

| Part | Time |
|---|---|
| AR path (DINOv2 x 4, VideoMAE x 3, TCN), chunks of 4 frames | 87.1 ms / frame (mean, first recording incl. warm-up) |
| v14 (localizer + verifiers on even frames, ASD on odd, filter) | 31.8 ms / frame mean, 67.8 ms p95 |
| Ensemble pass (3 V-JEPA 2 + VideoMAE, 3 samplings) | 1234 ms per action |
| AR label delay L = 0.3 s chunk wait + 4 x 87.1 ms + 67.8 ms | 0.716 s -> TCN shift 8 frames |
| Ensemble shift = 0.5 s confirmation + L + pass | 2.45 s -> 25 frames |

**Real time:** AR path + v14 on one RTX 2080 Ti = 119 ms / frame > 100 ms (10 fps): **not real time on one GPU**.
Each part alone is under 100 ms (AR path on one GPU, v14 + ensemble on the other); that two-GPU setup was not
measured running together.

## Notes
- Validation inputs are out-of-fold TCN labels (the final TCN trained on validation), as in the original v14 selection.
- The ensemble's test score with the old 1.7 s stream (0.815 / 0.898 / 6.9 s) was seen before this comparison; its
  settings were chosen on validation, so its validation labels are somewhat optimistic.
- The re-tune reruns only v14's own grid; v11-v13 settings are unchanged for both inputs.
- Latency comes from measured component times, not a frame-by-frame live run; each feature model decodes the video
  itself, so the AR time is conservative.

## Reproduce (from STARE)
GPU, once (their outputs are in `stare_outputs.tar`):
```bash
bash metrics/e2e/run_gpu.sh                                          # results/latency.json + work/val_ensemble
CUDA_VISIBLE_DEVICES=0 python metrics/e2e/latency_v9.py             # results/latency_v9.json (delay without the ASD)
```
CPU, every table row (this is what `metrics/reproduce_all.py` runs, into `metrics/reproduced/e2e`):
```bash
python metrics/e2e/streams.py tcn ens v13                            # delayed label streams, val + test
for n in tcn ens oracle; do                                          # re-tune v14 on val (grid split over 7 processes)
  for i in 0 1 2 3 4 5 6; do python metrics/e2e/retune.py grid $n $i 7 & done; wait
  python metrics/e2e/retune.py select $n
done
python metrics/e2e/test_once.py check-oracle                         # shift 0 must give the unshifted oracle run
python metrics/e2e/test_once.py tcn ens oracle v13                   # one test score each -> results/
for i in 0 1 2 3 4 5; do python metrics/e2e/noar.py grid $i 6 & done; wait   # w/o action recognition
python metrics/e2e/noar.py select && python metrics/e2e/noar.py test
```
`test_once.py` refuses to score an input twice.
