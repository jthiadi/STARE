# v14: v13 + official IndustReal ASD as extra state evidence (single test run, 13th test use)

v14 is v13 (same verifiers, step models, filter settings, mistake rules) plus the official IndustReal assembly state
detector (YOLOv8, `asd_best_IndustRealandSynthetic.pt`) as extra whole-state evidence in the state filter
(`psr/v14.py`). ASD runs on odd frames, alternating with the verifiers. Settings in `../model/config.json` (`asd`:
w 1, cmin 0.3, eps 0.02, rate 2).

## How it was selected
- Pre-registered before the grid (EXPERIMENT_LOG "v14 PRE-REGISTRATION"): grid w × cmin × eps × rate (54 configs);
  criterion = v13 criterion on val live AR + val oracle AR; eligible only if live-AR val mistakes are not worse than v13;
  latency rule p95 ≤ 100 ms. **Val only**: the ASD was trained on train, so train is in-sample for it.
- Disclosures: 7 exploratory configs were seen before the pre-registration; the ASD authors picked its checkpoint on
  val mAP (val mildly optimistic); the every-frame argmax failed the latency rule (p95 103.6 ms), so the best
  odd-frame config was used, as pre-registered.
- Dev (val, out-of-fold verifiers): live 0.877/0.891/6.9 s, err 0.753/0.775 (v13 0.861/0.883/7.7 s, 0.733/0.764).
  Nested leave-one-subject-out: live 0.861/0.882/7.0 s, err 0.753/0.775, live mistakes 2/13 at 3.8 FA/h
  (v13 2/13, 0 FA/h). Sanity check with the final verifiers on val (in-sample for them): v14 was worse than v13
  (0.921 vs 0.938 POS, mistakes 4/13 vs 7/13).

## Checks
- With ASD weight 0, the code reproduces v13's steps and alerts exactly (val).
- Causality: 96 cut points on val, 0 mismatches.
- Latency (RTX 4060 Laptop, 3,600 val frames; decode + ASD on odd frames + localizer + 2 verifiers on even frames +
  filter; AR model compute NOT included; other apps used ~30 % GPU): median 35.5 ms, p95 78.6 ms, mean 50.4 ms,
  peak GPU 602 MB (`latency_v14.json`). v13: median 27.8, p95 83.0, mean 39.7 ms.

## TEST (single run, official evaluator), POS / F1 / delay

| | All | Error recordings | Without errors | With-errors labels | Mistakes found / precision / FA per h |
|---|---|---|---|---|---|
| **v14, live AR (deployable)** | **0.812 / 0.904 / 6.32 s** | 0.747 / 0.839 / 7.33 s | 0.870 / 0.960 / 5.42 s | 0.769 / 0.878 | **9/32, 53 %, 3.2** |
| v13, live AR | 0.802 / 0.899 / 7.46 s | 0.747 / 0.839 / 9.04 s | 0.851 / 0.953 / 6.06 s | 0.759 / 0.873 | 7/32, 50 %, 2.8 |
| v14, GT AR (oracle, diagnostic) | 0.803 / 0.913 / 5.00 s | 0.728 / 0.858 / 5.02 s | 0.869 / 0.962 / 4.98 s | 0.761 / 0.887 | 7/32, 50 %, 2.8 |
| B3 (paper) | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s | | | ASD baseline 4/32, 21 %, 6.0 |

Mistakes by type (live AR): skipped 2/10, order 4/4, incorrect install 3/18 (v13: 2/10, 4/4, 1/18); 17 alerts,
8 false (v13: 14 alerts, 7 false). Bootstrap 95 % CI (live): all POS [0.73, 0.89], error-recordings POS [0.63, 0.85].
Per recording, POS changed on 4 of 32 (2 better, 2 worse).

**Versus v13:** better or equal on every step metric (error-recordings POS and F1 equal, all others better), 2 more
mistakes found at higher precision, but **one more false alarm** (2.8 → 3.2 per hour), and +7.7 ms median latency.
All gains are small compared with the bootstrap intervals.
