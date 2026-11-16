# Ablation: STARE without action recognition (vision only) — fixed before any run below

Written 2026-11-13, before the validation grid and the single test score.

## Question
How much does action recognition add? STARE with **no action input**: no triggers, no step model. Steps can only
come from the visual evidence (part verifiers on even frames, state detector on odd frames, both unchanged).

## Design
- Without triggers the frozen tracker never leaves its start state (its drift prior is 0), so the variant uses the
  tracker's existing drift prior `eps` (used on MECCANO): every frame, a small probability `eps` moves to the states
  one legal step group away; the visual evidence then decides. Everything else is v14's frozen setting (ASD weight 1,
  confidence 0.3, eps_ASD 0.02, odd frames; tau 0.05, theta 0.6, 3 stable readings, confirmation 6 / 5.0).
- `eps` is selected on the **16 validation recordings only**, with the selection harness of v14
  (`system/psr_v14/selection`: out-of-fold verifiers, so no validation recording is read by a verifier trained on it).
  Grid: eps in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. Criterion: v14's PSR criterion
  (POS + F1, all and with-error recordings, averaged, minus 0.02 x delay in s). Highest wins.
- Test: scored **once** with the official evaluator (`eval_v14.py`) with an all-background action stream, as the
  other end-to-end rows. There are no action labels to delay; visual readings are used at their own frame, as in
  every end-to-end row (v14's own compute, 67.8 ms p95, is under one frame).
- Reported next to the end-to-end rows: POS / F1 / delay (all, with errors), mistakes / 32, false alarms / h.
  Mistakes from actions (early warnings) do not exist without actions; the step-based rules run unchanged.

## Reproduce
`python metrics/e2e/noar.py grid <shard> <n>`, `select`, `test` (writes `results/retune_noar.json`, `results/test_noar/`,
adds `noar` to `results/e2e_summary.json`; refuses a second test score).
