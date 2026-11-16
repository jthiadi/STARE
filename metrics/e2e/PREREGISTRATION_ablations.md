# End-to-end ablations: v14 + oracle AR, v9 + TCN (fixed before any measurement or score below)

Written 2026-11-11, before the v9 latency measurement, the oracle re-tune and either test score. Same rules as
[`PREREGISTRATION.md`](PREREGISTRATION.md) (Table 5) unless stated here, so both rows compare directly with
**v14 + TCN, end to end: 0.819 / 0.907 / 6.5 s**.

## 1. v14 + oracle AR, end to end (diagnostic only, not deployable)
- Input: the annotated actions (`AR_labels.csv`), revealed causally exactly as `OracleARProvider` (segments with
  start <= f <= end), but frame t sees the annotation of frame t - D (nothing before D).
- Delay: perfect AR has no model, so no chunk wait and no AR compute. Only v14's own compute remains:
  `L_oracle = v14 p95` (67.8 ms, `results/latency.json`, already measured), `D_oracle = ceil(10 * L_oracle)` frames.
- Re-tune: the identical pre-registered v14 grid and rule as Table 5 (`retune.py`), with this shifted oracle stream
  as the "live" half on the 16 validation recordings; the oracle half is the original one, as for TCN.
- Implementation check before the test score: with D = 0 the shifted provider must reproduce Table 3's oracle row
  (0.803 / 0.913 / 5.0 s) exactly. That row is already reported, so this is no new look at test.
- Test scored once with the official evaluator. Reported as an oracle diagnostic only.

## 2. v9 + TCN, end to end (ablation: v14 -> v9, same AR)
- v9 = the v14 code with v9's frozen settings (`metrics/psr/model_v9_settings`: no ASD, tau 0.06, theta 0.8, d0 0),
  which gives exactly the frozen v9's steps (checked in `reproduce_all.py`) plus the same mistake rules as v14.
- Delay: `L_v9 = 0.3 s + 4 * c_ar + v9 p95`, with `c_ar` = 87.08 ms / frame (the same TCN path, already measured)
  and the v9 p95 measured now (`latency_v9.py`): the v14 live step with the ASD branch removed (localizer + both
  verifiers on even frames, tracker every frame), same GPU (RTX 2080 Ti, GPU 0), same 3 test recordings.
  `D_v9 = ceil(10 * L_v9)` frames on the same TCN labels.
- No re-tune: v9 has no ASD, so v14's ASD grid does not apply, and v9's own 192-config selection is not rerun.
  v9 keeps its frozen settings. (For TCN, v14's re-tune also chose v14's frozen setting, so both rows use their
  frozen settings.)
- Test scored once with the official evaluator.

## Reported
POS / F1 / delay (all, with errors), mistakes found / 32, false alarms / h, the shift used and the measured times,
next to the Table 5 rows. `test_once.py` refuses to score any input twice.

## 3. Addition (2026-11-13, before its test score): v14 without the state detector (= v13) + TCN, end to end
- Settings: `metrics/psr/model_v13_settings` (v14's frozen config with ASD weight 0; reproduces the frozen v13's
  recorded test result in video time). No re-tune: without the ASD, v14's grid does not apply (as for v9).
- Delay: the same per-frame work as v9 (localizer + verifiers on even frames, filter every frame, no ASD), so the
  measured v9 p95 (68.5 ms, `results/latency_v9.json`) is used: `D = D_v9 = 8` frames on the same TCN labels.
- Test scored once with the official evaluator (`test_once.py v13`), reported next to the other end-to-end rows.
