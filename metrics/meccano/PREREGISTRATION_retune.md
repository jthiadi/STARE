# STARE-lite on MECCANO: validation re-tune of the reporting settings (fixed before running)

Written 2026-11-11, after the first test score of STARE-lite with IndustReal's frozen settings
(0.521 / 0.705 / 211.0 s) and before any re-tuned configuration was run on validation or test.

**This is the second look at the MECCANO test set.** Both results are reported: the frozen-settings transfer and this
validation re-tune.

## What is re-tuned (validation only, 2 videos)
Only the step tracker's reporting settings; every model (action TCN, verifier, step model, triggers) is unchanged.

| Setting | Grid | IndustReal value |
|---|---|---|
| confirmation threshold `conf_thr` (mean log-odds of the changed parts) | 1, 2, 3, 5 | 5.0 |
| confirmation readings `conf_n` | 3, 6 | 6 |
| report posterior `theta` | 0.4, 0.6 | 0.6 |
| emission temperature `tau` | 0.05, 0.1 | 0.05 |

32 configurations. Everything else as in the frozen run.

### Amendment (before the grid finished; only validation was used)
After 3 of the 32 configurations (validation delays 225-235 s even at `conf_thr` 1), a validation-only look at the
step timings showed the cause is not the confirmation gate: the belief moves only on action triggers, and when the
weaker MECCANO action model misses one, the reported state lags until a later trigger (delays grow along a video,
then several steps are reported together). STARE's filter has two existing options for this, switched off in the
frozen IndustReal settings: a per-frame drift prior `eps` (a step can happen without a trigger, so visual evidence
alone moves the belief) and visual recovery `rec_n` (N consecutive contradicting readings correct the reported part).
Final grid (36 configurations, replaces the one above):

| Setting | Grid | IndustReal value |
|---|---|---|
| `conf_thr` | 1, 3, 5 | 5.0 |
| drift prior `eps` | 0, 1e-4, 1e-3 | 0 |
| visual recovery `rec_n` (with `rec_thr` 3.0, expected steps only) | 0, 12 | 0 |
| `tau` | 0.05, 0.1 | 0.05 |

`conf_n` 6 and `theta` 0.6 stay at IndustReal's values.

## Criterion (IndustReal's selection criterion, without the with-errors subset MECCANO's evaluator does not split)
`POS + F1 - 0.02 * delay_s` on validation (STORM-PSR's evaluator, per-video means, as reported). Ties: the
configuration closest to IndustReal's values (fewest changed settings), then the higher `conf_thr`.

## Test
The chosen configuration is scored once on the 7 test videos. Reported next to the frozen-settings result and the
published rows, with this note.
