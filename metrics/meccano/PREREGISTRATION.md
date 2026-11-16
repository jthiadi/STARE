# STARE-lite on MECCANO: procedure step recognition (fixed before running)

Written 2026-11-11, before any MECCANO model was trained or scored. Goal: test whether STARE's design transfers to a
second assembly dataset. This is a **reduced** version of the IndustReal system, built within about 12 hours; every
difference is listed below and must be stated when the result is reported.

## Data
- MECCANO (Ragusa et al.), 20 egocentric videos of a toy-motorbike assembly, 12 fps; official split
  11 train / 2 val / 7 test videos.
- PSR labels: the STORM-PSR authors' MECCANO annotations (`third_party/PSR-annotations`, 17 components, steps encoded
  as in IndustReal: id = 3k + {0 install, 1 incorrect install, 2 remove}); procedure definition
  `third_party/STORM-PSR/evaluation/utils/procedure_info_MECCANO.json`.
- Action labels: MECCANO's official action annotations (61 actions).

## Evaluation (as STORM-PSR)
POS, F1 and average delay on the 7 test videos with STORM-PSR's own MECCANO evaluator settings (12 fps, its
`determine_performance`), against the published rows: IndustReal baseline 0.354 / 0.545 / 99.8 s, STORM-PSR
0.377 / 0.497 / 88.6 s. Test is scored once.

## STARE-lite: what is kept and what is reduced
| Part | IndustReal STARE-TCN | STARE-lite on MECCANO |
|---|---|---|
| Action features | DINOv2 (3 views + hi-res hands), 2 fine-tuned DINOv2, VideoMAE-SSv2, 2 fine-tuned VideoMAE | **frozen only**: DINOv2-small 3 views + VideoMAE-SSv2 (causal clips) |
| Temporal model | causal verb × noun TCN ensembles | one causal TCN over the 61 actions + background, same architecture (4 stages × 10 layers, hidden 192) |
| Part verifiers | 2 × DINOv2-base, 10 parts | 1 × DINOv2-base (last 4 blocks, 4 epochs), 17 parts, trained on MECCANO train |
| Localizer | DINOv2 patch prototypes from IndustReal train boxes | same method, prototypes from MECCANO train state boxes |
| ASD | official IndustReal detector | **none** (no MECCANO detector is available) |
| Step model | naive Bayes + logistic regression on train triggers | same two models and blend, trained on MECCANO train |
| Triggers | 8 fastening actions (chosen from train statistics) | chosen from MECCANO train by the rule below |
| Filter settings | frozen IndustReal values | **the same frozen values**, not re-tuned (τ 0.05, θ 0.6, 3 stable readings, confirmation 6 readings / 5.0, view-quality 0.1 / 0.1, w_model 0.6, LR weight 4.0, action 0.6, since 0.3) |
| Short-trigger d0 | 12 frames (1.2 s at 10 fps) | 1.2 s = 14 frames at 12 fps |
| Frames | verifiers on even frames | verifiers on even frames |

**Trigger rule** (train videos only). Amended before any model was trained or scored: the first rule ("at least 30 %
of occurrences followed by a completion within 2 s") selects a single action on MECCANO train, because every step
involves several repeated fastening actions. Final rule, in the spirit of IndustReal's (a trigger marks a *possible*
completion; the step model learns "no step"): rank actions with at least 10 train occurrences by the share of their
occurrences that have a step completion within the step model's match window (84 frames before to 18 after the
action's end, IndustReal's 70 / 15 at 12 fps); add actions in that order until at least 80 % of train completions
have a trigger in that window.

**Validation** (2 videos) is used only for the TCN's epoch choice and as a sanity check; nothing else is tuned.
Everything is causal: each output at frame t uses frames up to t only.

## Report
POS / F1 / delay on test next to the two published rows, with this table of reductions. If STARE-lite does not beat
them, that is reported as is.
