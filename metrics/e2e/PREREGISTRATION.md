# End-to-end delay, fair comparison: v14 + TCN vs v14 + ensemble (fixed before any measurement)

Written 2026-11-10, before the latency measurement, the validation runs and the test scoring below.

## Question
PSR on the 32 test recordings when every label reaches v14 only at the time the live system would really have it
(compute included), for two action-recognition inputs, each with v14 re-tuned on validation for that input:
1. **v14 + TCN**: the causal verb x noun TCN (DINOv2 + VideoMAE features).
2. **v14 + ensemble**: final labels of the action-end ensemble (3 V-JEPA 2 + VideoMAE + TCN), TCN labels until they exist.

## Latency (measured once, on the server's RTX 2080 Ti, `latency.py`)
- Live processing in chunks of 4 frames (0.4 s, the VideoMAE clip stride). Every component of the AR path runs with
  batch 4 (one VideoMAE clip per chunk), from the video file; mean compute per frame `c_ar` = total time / frames.
- v14 per-frame compute: p95 of its own per-frame timing (localizer + verifiers on even frames, ASD on odd, filter).
- Ensemble pass `P`: the teammate's benchmark (`run_action_end.benchmark`, every member, every sampling, one action).
- Recordings timed: test recordings 1, 12 and 23 of the official test list (chosen before measuring).
- **AR-label delay** `L = 0.3 s (wait for the chunk to fill) + 4 * c_ar + v14 p95`.
  TCN stream shift `D_tcn = ceil(10 * L)` frames.
- Ensemble: a final label exists at `ended_at + L + P`; stream shift `D_ens = ceil(10 * (0.5 + L + P))` frames
  (the rule of make_final_label_stream.py, 0.5 s confirmation + pass, now plus L). Before its final label exists, a
  frame gets the TCN label. Hardware: AR path + v14 on GPU 0, ensemble on GPU 1.
- Real-time check: `c_ar + v14 mean <= 100 ms` (10 fps). If it fails, the shift is reported but the system is
  stated as not real time on this GPU.

## Validation inputs (no test data, no future frames)
- TCN on validation = the out-of-fold labels v14 was originally selected with (`output/assist/<val>/ar.jsonl`;
  each validation video labelled by TCN folds that did not train on it). The final TCN saw validation in training,
  so its validation labels are not used.
- Ensemble on validation: the released V-JEPA 2 runs + VideoMAE (trained on the 36 train videos, settings chosen on
  validation) on the spans detected from the same out-of-fold TCN labels; its TCN term uses those labels (top 5 of
  each frame kept, the remaining probability spread evenly; spans are exact because they use the top 1).

## Re-tuning (validation only, identical for both inputs)
- The pre-registered v14 grid (`selection/v14/asd_grid.py`): ASD weight {0.3, 1, 3} x confidence {0.3, 0.5, 0.7}
  x eps {0.02, 0.05, 0.1}, plus the v13 control. Odd-frame ASD only (rate 2): the original latency rule removed
  every-frame ASD (p95 103.6 ms > 100 ms) and that is kept.
- Criterion and eligibility exactly as `selection/v14/asd_select.py` (live-AR val + oracle val; eligible only if
  live-AR val mistakes are not worse than the v13 control). The oracle half does not depend on the AR input, so the
  original oracle runs are reused; only the live half is rerun with the shifted stream.
- The best eligible config is used if its criterion beats the v13 control; otherwise the v13 control (ASD off).
- Everything below v14 (v11-v13 settings) is unchanged for both inputs.

## Test
- Each chosen setting is scored once on test with the official evaluator (`eval_v14.py`), on its shifted stream.
- Reported: POS / F1 / delay (all, with errors), mistakes, false alarms / h, the shifts used, and the measured times.

## Known limits (stated in the report)
- The ensemble's test score with the old v14 setting has been seen before this comparison.
- The ensemble's settings were chosen on validation, so its validation labels are somewhat optimistic.
- Latency is computed from measured component times, not a frame-by-frame live run.
