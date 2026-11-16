# Action recognition metrics (paper Tables 1-3)

Every number in Tables 1-3 is recomputed and checked by `python metrics/reproduce_all.py --data <IndustReal root>` (from
the STARE folder, CPU). Each script below writes `results/<name>.json` and runs from `system/assistant`.

## Table 1. Action recognition on IndustReal (3,678 test actions, 74 actions, top-1)

*Official*: each annotated action cut at its true start / end. *Live*: the actions are the ones the causal TCN detects
itself (the live system's `build_events`); each is labelled with frames up to its confirmed end, and a true action counts
as right if its most-overlapping detection has its label (none = wrong; the teammate's `best_overlap_accuracy`).

| Model | Official | Live | Script -> result key |
|---|---|---|---|
| SlowFast (IndustReal paper) | 60.39 | – | published |
| MViTv2 (IndustReal paper, RGB) | 65.25 | – | published |
| Fine-tuned V-JEPA 2 ViT-L, hand crop, single run (`vjft_hand_k12`; val 68.26) | 70.20 | – | `score_vjepa_paper_protocol.py` (prints one line per saved run) |
| Causal TCN | 55.74 | 53.72 | `score_live_models_paper_protocol.py` -> "TCN alone"; `score_live_condition.py` -> "TCN" |
| Final label (3 V-JEPA 2 + VideoMAE + TCN + order prior; val 73.81) | 75.07 | 55.52 | `score_final_labels_paper_protocol.py`; `score_live_condition.py` -> "Final labels ..." |

How: the TCN's official score is the mean of its causal per-frame probabilities over the segment. The final label is the
teammate's `action_end_benchmark.py` unchanged (its guard kept: validation must give the locked 73.81 or test is not
scored); the live system calls the same function (`assist/extract/action_end.py`) on the detected actions. Inputs are the
model outputs in `system/assistant/output/` (made by `system/assistant/live` on the GPU, every model run on the video).

## Table 2. What each part of the final label adds (test, diagnostic)

The deployed recipe rebuilt one part at a time; nothing here was chosen (all settings were fixed on validation). Only test
is shown: the deployed TCN has seen the validation videos (see "Note on val" below), so validation rows that use it would
be inflated. The script refuses to write unless the full recipe gives val 73.81 / test 75.07.
`score_final_label_breakdown.py` -> `final_label_breakdown.json`; test: `system/assistant/tests/test_ar_final_label_breakdown.py`.

| Step | IndustReal | MECCANO (`output/meccano_action_end/ablation.json`, not in the paper) |
|---|---|---|
| VideoMAE alone | 62.18 | 51.82 |
| V-JEPA 2, single run (area / hand / hand seed 1) | 69.55 / 70.20 / 70.12 | 57.88 / 60.86 / 61.60 |
| Mean of 3 V-JEPA 2 runs | 72.76 | 62.98 |
| + VideoMAE (mean of 4) | 73.25 | 62.98 |
| + TCN (w 0.95) | 73.74 | 62.98 |
| + order prior (= final label) | **75.07** | **62.45** |
| TCN alone | 55.74 | 32.66 |

## Table 3. Action recognition on MECCANO (2,823 test actions, 61 actions, top-1)

| Model | Top-1 | Source |
|---|---|---|
| SlowFast (MECCANO paper, CVIU 2023) | 45.16 | published |
| Swin3D-B (MECCANO challenge winner) | 50.80 (53.03 trained on train + val) | published |
| Final label, retrained on MECCANO (val 66.12) | 62.45 | `score_meccano_final_labels.py` -> `meccano_final_labels.json` |

The final-label method with the IndustReal settings unchanged; the models are retrained on the 11 MECCANO training videos
and the one choice (which of two VideoMAE runs) was made on validation (`output/meccano_action_end/action_end_val.json`).
How the models were trained: `system/assistant/meccano_action_end/README.md`. Test:
`system/assistant/tests/test_meccano_final_labels_score.py`.

## More numbers (not in the result tables)

### The official protocol: other models

Each of the 3,678 annotated test segments is cut at its true start and end and classified among the 74 actions (background
excluded), as in the IndustReal paper (Table 3: SlowFast 60.4 %, MViTv2 65.3 %, RGB).

| Model | Val | Test | How it was scored here |
|---|---|---|---|
| IndustReal paper: SlowFast / MViTv2 (RGB) | | 60.4 / 65.3 | published |
| **Final labels in the system** (action-end recipe: 3 fine-tuned V-JEPA 2 + fine-tuned VideoMAE, weight 0.95 vs the TCN, order prior; locked on val) | **73.81** | **75.07** | recomputed: the teammate's benchmark, unchanged; its guard (val must give the locked 73.81) passed |
| V-JEPA 2 fine-tuned, hand crop 12 layers (one run) | 68.26 | 70.20 | saved per-segment outputs, rescored |
| V-JEPA 2 fine-tuned, best single run (hand, context 0.25, 15 epochs) | 69.45 | 72.54 | saved outputs, rescored |
| V-JEPA 2 retrained on action beginnings (the one in the live label) | 62.45 | 64.55 | saved outputs, rescored |
| V-JEPA 2 frozen + attentive probe (3 seeds) | 54.77 | 56.63 | saved outputs, rescored |
| Causal TCN alone, from its saved top-5 labels (Table 1 uses the full probabilities: 55.74) | 49.17 | 55.66 | live labels (top-5 kept) |

All 14 saved V-JEPA 2 runs give exactly the val / test numbers recorded when they were trained
(`results/vjepa_runs_paper_protocol.txt`). The final-label recipe was chosen on val by the teammate and locked before test
was scored; the test set had been used before for related models (their README: 77.16 / 77.6 / 78.1 for offline ensembles),
so 75.07 is not a first look at test, but nothing in it was chosen on test.

### The live labels (what the assistant shows while an action runs), test

Computed from the video by our run of the teammate's pipeline (`system/assistant/live`, steps 1-3):

| Label | Frames right | Right at each action's end | Paper protocol, from the saved top-5 labels | Teammate's README |
|---|---|---|---|---|
| Causal TCN (recomputed) | 48.37 % | 39.78 % | 55.66 % | 48.4 % / 39.8 % (identical to their saved labels) |
| TCN + action-beginning V-JEPA 2, w 0.1 (live label) | 48.76 % | 43.18 % | 55.46 % | 48.8 % / 43.2 % |
| Final labels on the actions the system detects (not true boundaries) | 55.52 % of true actions labelled right | | | 55.46 % |

The recomputed TCN agrees with the teammate's saved labels on 99.92 % of frames (checked on 13_main_2_1) and gives the same
scores on all 32 test recordings; the V-JEPA 2 window scores and the action-end scores are identical to their saved ones (max
difference 0.0, checked on 13_main_2_1).

Note on val: the recomputed TCN gets 70.0 % of val frames right against 44.6 % for the teammate's saved val labels, which were
out-of-fold; the deployed TCN ensemble has therefore seen the val videos. This only affects the val guard of the final-label
benchmark (which still gives exactly the locked 73.81); every test number above is unaffected.

## Recreate

From `system/assistant` on the GPU machine:
```bash
python ../../metrics/ar/score_vjepa_paper_protocol.py                       # saved V-JEPA 2 runs (CPU)
python -m live.run_video --test 0/2                                         # live labels + final labels for test (and 1/2)
python ../../metrics/ar/score_live_labels.py test ar_live.jsonl ar_vjepa_prefix.jsonl      # -> results/live_labels_test.json
for r in <the 16 val recordings>; do python -m live.step_ar $r; done        # TCN on val (for the benchmark's val guard)
python ../../metrics/ar/score_final_labels_paper_protocol.py                # -> results/final_labels_paper_protocol.json
```
