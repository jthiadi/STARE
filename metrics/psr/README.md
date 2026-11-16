# Additional PSR checks (IndustReal test set, compute excluded; not in the paper)

The paper reports step recognition end to end (`metrics/e2e`, Tables 4, 6, 7). This folder keeps the earlier checks in
the published protocol, where delay is counted in video time and compute is excluded (as B3 and STORM-PSR report), and
the settings files `model_v9_settings/` and `model_v13_settings/` that the end-to-end ablations use.

POS / F1 / average delay, scored with the official IndustReal PSR evaluator (`system/psr_v14/third_party/industreal_psr/psr_utils.py`,
unmodified) or the teammate's port of it (`system/assistant/assist/evaluation/psr_eval.py`, checked equal to the official code; it
reproduces v14's frozen numbers exactly). 32 test recordings; "with errors" = the paper's group (15 recordings). Mistakes: the
teammate's rules and ±30 s matching (32 annotated mistakes, 2.5 h). Ground truth is read only after a recording is processed.

## v14 by AR input, and v9 vs v14 (test)

Mistakes: found out of 32, false alarms per hour. These rows are not checked by `metrics/reproduce_all.py`. The v14 rows
are recomputed with the commands in the last column. The frozen v9 package is no longer in this repository, so its
rows are kept as recorded; v9's settings run through the v14 code (`model_v9_settings`) give the same steps.

| Row | All: POS / F1 / delay | With errors: POS / F1 / delay | Mistakes | Command (in `reproduce_all.py`) |
|---|---|---|---|---|
| B3 (IndustReal paper) | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s | – | published |
| **v14 + TCN** (selected input) | **0.812 / 0.904 / 6.3 s** | **0.747 / 0.839 / 7.3 s** | 9, 3.2 | `psr_v14/eval_v14.py --split test --ar predicted --ar-dir system/assistant/output/assist` |
| v14 + ensemble (V-JEPA 2 ensemble + VideoMAE + TCN) | 0.815 / 0.898 / 6.9 s | 0.754 / 0.825 / 7.9 s | 9, 4.0 | same, `--ar-dir` = stream from `make_final_label_stream.py` |
| v14 + annotated actions (oracle AR, diagnostic) | 0.803 / 0.913 / 5.0 s | 0.728 / 0.858 / 5.0 s | 7, 2.8 | `eval_v14.py --split test --ar gt` |
| v9 + TCN | 0.804 / 0.893 / 7.6 s | 0.700 / 0.817 / 9.0 s | 7, 2.4 | frozen `psr_v9/eval_psr.py --split test --ar predicted`; mistakes from `eval_v14.py --model model_v9_settings` |
| v9 + ensemble | 0.814 / 0.893 / 8.0 s | 0.743 / 0.820 / 9.4 s | 7, 4.4 | `eval_v14.py --model model_v9_settings`, ensemble stream |
| v9 + annotated actions (oracle AR, diagnostic) | 0.833 / 0.918 / 5.7 s | 0.718 / 0.851 / 5.6 s | – | frozen `psr_v9/eval_psr.py --split test --ar gt` |

- The ensemble stream is causal: a final label exists only once its action's end is confirmed (0.5 s) plus the ensemble pass
  (1.21 s, measured), so v14 gets the stream 1.7 s late, with the TCN label wherever no final label exists yet
  (`make_final_label_stream.py`). This 1.7 s is inside the delays above.
- `model_v9_settings/` = v9's settings (tau 0.06, theta 0.8, no d0, no ASD) in v14's format; v9 and v14 share the same step
  models and verifiers. Run through the v14 code it gives exactly the frozen v9's steps (checked when v9 was archived), so
  its mistake counts are v9's.
- v9 and v14 were each selected on train / val with the TCN input and frozen before test. The ensemble and oracle rows were
  scored on test afterwards as an ablation and chose nothing.
- Saved copies: `results/v9/`, `results/v14_frozen_rerun/`.
- Not in the tables: v14 fed by V-JEPA 2 alone scored 0.844 POS, but that label stream was built after looking at test
  results; it is a test-informed diagnostic (`results/diagnostic_labels/`), not a result.

## Other systems and runs (test)


| System (test) | All: POS / F1 / delay | With errors: POS / F1 / delay | Mistakes found / precision / false alarms per h |
|---|---|---|---|
| B3, IndustReal paper (WACV'24, Table 4) | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s | – |
| STORM-PSR (CVIU'25), as published | 0.812 / 0.901 / 15.5 s | – | – |
| Official IndustReal PSR on the released ASD (rerun) | 0.815 / 0.878 / 16.9 s | 0.731 / 0.807 / 14.8 s | 4/32, 21 %, 6.0 (same rules) |
| Teammate engine, TCN actions + released ASD | 0.795 / 0.892 / 15.9 s | 0.704 / 0.804 / 13.6 s | 8/32, 35 %, 6.0 |
| **v14 (frozen; TCN live actions; verifier / ASD outputs from the dataset's JPEG frames)** | **0.812 / 0.904 / 6.3 s** | **0.747 / 0.839 / 7.3 s** | **9/32, 53 %, 3.2** |
| v14, same, rerun on the lab server (identical) | 0.812 / 0.904 / 6.3 s | 0.747 / 0.839 / 7.3 s | 9/32, 53 %, 3.2 |
| Full system: v14 run live on the video, fed by TCN + V-JEPA 2 actions | 0.788 / 0.893 / 6.9 s | 0.708 / 0.829 / 8.0 s | 8/32, 42 %, 4.4 |
| Full system: teammate engine with the full system's v14 steps as state | 0.809 / 0.895 / 7.5 s | 0.734 / 0.833 / 9.1 s | 8/32, 20 %, 13.2 |
| Engine oracle (annotated actions + states), upper bound | 0.954 / 0.964 / 0.7 s | 0.902 / 0.923 / 0.7 s | 32/32, 94 %, 0.8 |

Diagnostic (which change costs what; frozen v14 with its cached JPEG-frame outputs, only the actions swapped):

| v14 input actions (test) | All: POS / F1 / delay | With errors: POS / F1 | Mistakes |
|---|---|---|---|
| TCN, recomputed from the video by our run of their pipeline | 0.812 / 0.904 / 6.3 s (= frozen exactly) | 0.747 / 0.839 | 9/32, 53 %, 3.2 |
| TCN + V-JEPA 2 (their newest live label) | 0.805 / 0.900 / 6.8 s | 0.716 / 0.828 | 9/32, 50 %, 3.6 |

v14 was selected with the TCN actions; the V-JEPA 2-adjusted actions were never validated for it. Running live on the
compressed video frames (not the dataset's JPEGs) costs about 0.017 POS more.

95 % bootstrap intervals over recordings (v14 frozen): all POS 0.73-0.89, with-errors POS 0.63-0.85: every POS difference in the
table above is inside them. Test-set use: v14 was selected on train + val and scored on test once (13th test use in the project);
the full-system rows and the diagnostic were scored after that, on the same test set, and changed nothing.

## Recreate

From `system/assistant` on the GPU machine, with the data and weights in place (see the main README):

```bash
python -m assist.evaluation.psr_eval --runs gt_gt pred_asd pred_asdv14       # baseline + engine rows -> output/assist/eval/
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_asd pred_asdv14
python -m scripts.mistakes.compare_v14 --actions pred                        # v14 inside the teammate's scorers
python -m live.run_video --test 0/2                                         # full system, all models (one per GPU; 1/2 on the other)
python ../../metrics/psr/score_full_system.py                               # -> results/full_system_test.json
```
Frozen v14 itself (cached outputs; frames from video if the dataset has no JPEGs: `system/server/run_frozen.py`):
```bash
python ../server/run_frozen.py eval_v14.py --split test --ar predicted --ar-dir output/assist --out <dir>
python ../psr_v14/tools/bootstrap_ci.py <dir>/v14_predictedAR_test.json
```

## Files

`results/`: `full_system_test.json`, `v14_frozen_rerun/`, `v14_inside_teammate_scorer.json`, `psr_test_release.*` and
`eval_test.*` (teammate scorers; the `pred_v14` PSR row there is a lossy conversion for the dashboard, use
`v14_inside_teammate_scorer.json` instead), `diagnostic_labels/`. v14's own frozen results: `system/psr_v14/results/`.
