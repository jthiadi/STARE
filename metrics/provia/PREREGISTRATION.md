# STARE-TCN mistake alerts on PROVIA's IndustReal protocol (fixed before scoring)

Written 2026-11-10, before our alerts were scored with PROVIA's protocol.

## Benchmark (PROVIA's, unchanged)
- PROVIA (Wen et al., arXiv 2609.20638), code `third_party/PROVIA` (commit 178e78f, unmodified).
- Steps and mistakes: PROVIA's own IndustReal conversion (`make_manifest.sh`: `provia.data.cli convert` +
  `project_single_actor --sample-rate-hz 8 --action-marks`). Test: 194 step completions, 17 mistakes (incorrect
  completions, "Incorrectly ..." steps of `PSR_labels_with_errors.csv`), as in their paper.
- Matching and metrics: PROVIA's functions, imported unchanged: Voronoi-cell assignment of decisions to annotated
  completions (`completion_metrics._cell_assignment`), alarm scoring (`alarm_protocol.evaluate`, grace 1), step-level
  pooling (max score of the decisions assigned to a step, 0 if none) with AP / AUROC (scikit-learn).

## Our system (frozen, nothing tuned for this)
- STARE-TCN (v14 + TCN), its frozen test run (`system/psr_v14/results/v14_predictedAR_test.json`, the Table 3 row).
- Decisions = the times at which it reports completed steps (steps reported at the same time = one decision), on
  PROVIA's 8 Hz grid. Two score-0 anchor decisions at the recording's start and end, so the time span is the whole
  recording (PROVIA's false-alarm rate is per minute of the decision span).
- Each alert is placed on the latest step decision at or before its time (the first step decision if none; dropped,
  and counted, if the recording has no step decision).
- Score: 1 on a decision carrying an alert, else 0. No threshold is chosen: one operating point.
  - **Primary**: install-type alerts only (incorrect_install, possible_incorrect_install, wrong_part), the mistake
    type PROVIA's IndustReal labels contain.
  - **Secondary**: every alert type (also wrong_order and skipped_step).

## Report
Recall of all mistakes, false alarms per minute of correct operation, median delay, step-level AP / AUROC, on test;
next to PROVIA's published IndustReal numbers (Tables II and IV of their paper). Validation is scored the same way
for information only. PROVIA's numbers are as published, not re-run.
