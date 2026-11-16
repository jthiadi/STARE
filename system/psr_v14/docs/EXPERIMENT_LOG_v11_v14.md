# Experiment log: v11 to v14 (excerpt)

Copied unchanged from the project's EXPERIMENT_LOG.md (earlier stages, v3 to v10, are in the full log, not included). Paths: `v13/...` and `v14/...` are under `selection/` here; `industreal_psr_v11/model_v13` and `model` are `model/`.

## 2026-11-07 — v11 = v9 + live-AR robustness + mistake alerts without ASD (train+val only until the freeze)
- New dev harness `v11/` (oracle AR on train+val with LOSO step models and OOF verifiers; plus val with the teammate's
  out-of-fold live AR, the only fair live-AR dev set). v9 reproduced exactly (oracle 0.893/0.940/4.9 s; live val
  0.822/0.876/7.9 s, err 0.733/0.748).
- Live AR on val: 276 fastening triggers vs 174 GT; spurious ones are short (median 5 vs 19 frames) and low-confidence
  (median 0.57 vs 0.87). Tried: label smoothing (hysteresis) -> no gain; confidence-weighted triggers -> no gain;
  **duration-weighted triggers** (a trigger moves the belief with p = min(1, dur/d0)) -> gain.
- Grids (175 configs, criterion as v9 on oracle train+val + the same criterion on live val; full selection):
  tau 0.05, theta 0.7, d0 8 (others as v9). Oracle 0.896/0.939/4.8 s, err 0.833/0.888; live val 0.853/0.883/7.7 s,
  err 0.733/0.764. Nested 4-fold subject CV of the selection: oracle 0.896/0.936; live val 0.845/0.879/7.7 s, err
  0.712/0.753 (v9: 0.822/0.876, err 0.733/0.748) -> gain on all recordings, none shown on error recordings.
- No gain: uni 0.05 (much worse), tau 0.03/0.04, stricter confirmation / stability without a delay penalty, step models
  trained with added live-AR samples of val (worse), top-2 AR (missed triggers are not in top-2), washer/pulley recipe
  for wheel errors (2/5 at 3 FA oracle), frame-level ASR mistake detector at wheel installs (fires after correct rear
  wheels too), stricter wheel confirmation (incorrect installs read fully attached: max mean log-odds 6.9).
- Mistakes, scored with the teammate's rules and matching (assist/evaluation/eval.py; baseline on test in their README:
  official PSR on ASD + same rules 4/32, prec 21 %, 6.0 FA/h; their engine with ASD 8/32, 35 %, 6.0 FA/h):
  v9 steps replayed (no ASD) train+val oracle 11/38, prec 0.38, 5.5 FA/h. Most FAs = end-of-recording "skipped" alerts
  for parts that were handled. Rule: keep such an alert only if the filter does not believe the step done (< 0.5) and the
  last 20 readings do not show it; an install believed done (>= 0.9) but never confirmed -> possible incorrect install.
  -> train+val oracle 13/38, prec 0.59, 2.8 FA/h (2 incorrect installs found); live val 2/13, no FA (in the CV setting).
  Rejected: "trigger mass but unconfirmed after D s" (60-90 alerts), "believed >= b for D s" (mostly FAs on live val).
- Release folder industreal_psr_v11 (v9 copied; psr/v11.py, eval_v11.py, tools/causality_check.py). With d0 0 and v9
  filter settings eval_v11.py reproduces v9's val steps exactly. Causality (val, 96 cut points): 0 mismatches.
  Latency (RTX 4060 Laptop, 3,600 val frames, PSR side; live AR read from ar.jsonl, AR compute NOT included):
  median 29.0 ms, p95 72.6 ms, mean 37.6 ms, max 1134 ms (warm-up), peak GPU 423 MB; mistake check < 1 ms.
- Frozen before the test run (industreal_psr_v11/model, SHA256SUMS). Test: one run with live AR (deployable) and one with
  GT AR (oracle, diagnostic). 11th test use in this project.
- **TEST (single run of frozen v11, official evaluator)**, POS / F1 / delay:
  | AR | all | err recs | no-err | w/err labels | mistakes found / prec / FA per h |
  | live | 0.788/0.891/7.45 s | 0.724/0.833/9.0 s | 0.845/0.943/6.1 s | 0.747/0.866 | 7/32, 50 %, 2.8 |
  | GT (oracle) | 0.795/0.910/5.32 s | 0.711/0.851/5.55 s | 0.869/0.962/5.1 s | 0.754/0.884 | 6/32, 50 %, 2.4 |
  v9 live: 0.804/0.893/7.6 s; err 0.700/0.817/9.0 s. B3: 0.797/0.883/22.4 s; err 0.731/0.816/20.4 s.
  Mistake baseline (teammate, ASD + same rules): 4/32, 21 %, 6.0/h; teammate engine (ASD): 8/32, 35 %, 6.0/h.
  -> v11 beats B3 on F1 and delay everywhere but NOT on POS (all -0.009, err -0.007). Gains on error recordings
  vs v9, loses on recordings without errors. CV -> test transfer failed again for the filter thresholds (cf. v8).
  Bootstrap 95 % CI over videos: all POS [0.70, 0.87], err POS [0.60, 0.83].
- Post-hoc test ablations (after the frozen run; diagnostic only; changed nothing; results/ablations_test_posthoc):
  v9 + d0 8: live 0.805/0.894, err 0.717/0.824; v9 + tau .05/theta .7: live 0.786/0.893, GT 0.795/0.910;
  tau .05/theta .8/d0 8 (pre-test near-tie): live 0.820/0.903/7.8 s, err 0.741/0.840/9.2 s (above B3 on all six, but
  identified on test -> not a fair result, not claimed). theta 0.7 is the change that hurt on test.

## 2026-11-07 (afternoon) — v12 attempt, train+val only (test NOT run)
Dev harness v12/ (built on v11/h.py; OOF verifiers, LOSO step models, oracle AR train+val + live AR val).
Base for comparison: v9 + d0 8, tau 0.05.
- Order of steps reported in the same frame (8/168 wrong pairs oracle, 3/45 live; each costs ~0.12 POS of a video):
  visual-onset order: oracle 0.896 -> 0.833 (much worse). Learned precedence (LOSO labels) needs conditioning on the
  already-reported steps (main starts in two mirror orders); conditioned version: oracle all 0.896 = 0.896, err
  0.833 -> 0.818; live all 0.853 -> 0.861, err same. Mixed -> rejected.
- Remove + reinstall cycles (largest single POS loss on error recordings, ~0.3 per affected video): verifier dips to ~0
  for 7-20 s at true removals, but ~17 equally long dips occur on attached parts (occlusion) vs ~3 true assy removals
  -> visual removal detector not viable (consistent with earlier visual-recovery FPs). Not implemented.
- Report pending state at end of stream (posterior >= 0.5, recent mean log-odds >= 2): oracle +0.003 all / +0.007 err,
  live unchanged (one video). Too small to adopt.
- Fold-consistency check of the v11 grid (criterion delta vs v9 per subject fold): also picks the v11 configuration
  (tau .05, theta .7, d0 8; 3/4 folds up, worst fold -0.001). On CV, theta 0.7 vs 0.8 differ only by 0.2 s delay
  (POS/F1 identical). A "keep the default unless the gain exceeds noise" rule would keep theta 0.8, i.e. the post-hoc
  config D whose test result is already known (0.820/0.903/7.8 s) -> a test run of it would not be a held-out result.
- Conclusion: no new CV-visible gain; test not run (would be the 12th use and, for D, its outcome is already known).
- Round 2 (same day, train+val only, test NOT run):
  * Split composition from file names: test recordings ~25% longer (282 s vs ~225 s); procedure/error mix similar.
    Nested CV vs test: error recordings match (live 0.712 vs 0.724); the CV->test gap is on recordings WITHOUT errors
    (CV ~0.94 vs test live 0.845 / GT AR 0.869).
  * Per-recording verifier bias from the first readings (state known from the procedure): within-component
    early/late correlation 0.11 -> calibration per recording not useful.
  * Clean main recordings 11/16/21_main_0_1 lose 0.25 POS each purely by order (opening removals 11/17/20 reported in
    one frame, canonical id order). Same-frame ordering variants: never-seen-order fix (+2 videos, -2 videos),
    Bayes order = min expected POS edit distance vs training-subject orders (+0.25, +0.25, +0.12 live; -0.13, -0.13,
    -0.29: subjects 25/27 use a different habit). Net ~0 -> rejected. Absence-onset order of removals: worse.
  * Visual recovery of unreported steps (rec_n 30-150, rec_thr 3-4, expected steps only): no gain (oracle 0.896 ->
    0.867-0.896, live 0.853 -> 0.845).
  * Verifier capacity: held-out AUC already 0.96-0.99; 336 -> 448 px and wider training gave <= +0.01 AUC earlier.

## v12 Round 3: DINOv2-large verifier (fold 0 only) - REJECTED
- Same recipe as verif_base_r336 (336 px, unfreeze 4, 4 epochs), backbone DINOv2-large. Train+val fold 0, held-out subjects 01/06/15/22/27. Test untouched.
- Held-out AUC per component, large vs base: 1 0.990/0.991, 2 0.988/0.990, 3 0.976/0.974, 4 0.965/0.958, 5 0.968/0.965, 6 0.973/0.974, 7 0.991/0.992, 8 0.993/0.994, 9 0.975/0.982, 10 0.979/0.982. Mean 0.980 vs 0.980.
- Training loss matched base at every epoch (final 0.0167 vs 0.0165). Cost: 2x training time (99 min vs 49 min per fold), 38 ms/image inference.
- Decision: no gain, so folds 1-3 / final / 12th test run were not done. Verifier capacity is not the bottleneck.

## v12 Round 4: error breakdown, train+val only (test untouched)
- val (16 videos with live AR), v11 settings: oracle AR 0.862/0.904/6.2 s (FP14 FN11) vs live AR 0.853/0.883/7.7 s (FP18 FN13).
  Live AR costs only ~0.01 POS on val; AR quality is not the bottleneck.
- Remaining errors (per-video sequences, v12/diag_live.py): (a) extra "install" reports for parts that are never correctly
  installed (skipped / incorrect / recording ends), mostly the last parts 9-10; (b) missed short undo-redo episodes
  (e.g. part removed and re-installed within seconds); (c) swaps of two steps done near-simultaneously.
- Visual-confirmation margin of wrong vs correct reports (v12/diag_margin.py): oracle TP 425 p50 5.8, FP 9 p50 5.4; live
  TP 130 p50 5.7, FP 6 p50 5.4. Both saturate near the cap (6.9): wrong reports are confidently confirmed by the camera,
  so no threshold separates them. Incorrect-install detection was already tried (only 19 examples; correct-vs-incorrect
  AUC near chance for wheels). No fair lever left with the current data.

## 2026-11-07 (evening) — v11b = v11 with theta 0.8, frozen at user request. TEST-INFORMED.
- model_v11b/ (only config.json differs from v11). Train+val: identical POS/F1/mistakes to v11, +0.1-0.2 s delay.
- Causality 0/96 mismatches; latency median 32.8 / p95 79.6 / mean 40.6 ms (AR compute excluded).
- Test (reproduces post-hoc ablation D exactly): live 0.820/0.903/7.82 s, err 0.741/0.840/9.17 s, mistakes 7/32 54 % 2.4 FA/h.
  Chosen after seeing test -> not an independent confirmation. Details: industreal_psr_v11/results/v11b/README.md.

## 2026-11-07 (night) — v13 PRE-REGISTRATION (written before any v13 result; train+val only until the freeze)
Problem: train+val cannot separate near-tied filter settings (consensus of top-N configs: identical nested CV).
Only 16 val recordings have live AR. Plan: simulated live AR for all 52 train+val recordings.
- Simulator (v13/simar.py): primary true-action segments define regions [start_i, start_{i+1}); each region of a
  recording is replaced by the live-AR frame labels of a random val region with the same true action (fallback: any
  region), time-rescaled to the region length. Templates come only from val subjects other than the recording's subject.
  3 seeds.
- Simulator check (before selection): on val, simulated vs real live AR with frozen v11 settings; report the gap.
- Grid (fixed now): tau {0.04, 0.05, 0.06} x theta {0.6, 0.7, 0.8, 0.9} x d0 {0, 4, 8, 12}; all other settings = v11.
- Criterion (fixed now): v11 criterion on oracle AR (52) + real live AR val (16) + simulated live AR (52, mean of 3 seeds).
- Commitment: the argmax is frozen as v13 WHATEVER it is (also if it equals v11 = theta 0.7), nested 4-fold subject CV
  reported, then ONE test run (12th test use), reported in full. No changes after the test run.
- Disclosure: this was started after the v11 test run and post-hoc ablations showed theta 0.8 better on test, so the
  motivation (dev cannot separate settings) is test-informed; the selection itself uses train+val only.
- v13 selection result (v13/selection.py): argmax tau 0.05 / theta 0.6 / d0 12 (total 4.6509; v11 config rank 7, 4.6392).
  Full-selection: oracle 0.896/0.939/4.8 s, err 0.833/0.888; live val 0.861/0.883/7.7 s, err 0.733/0.764; sim0 0.815/0.893.
  Simulator check: sim val (3 seeds) 0.78-0.84 vs real 0.853 (harsher). Nested CV: oracle 0.896/0.939; live val
  0.845/0.879/7.7 s, err 0.712/0.753; sim 0.794-0.826 / err 0.692-0.744. Frozen as committed: industreal_psr_v11/model_v13.
- v13 TEST (single run, 12th test use): live 0.802/0.899/7.46 s, err 0.747/0.839/9.04 s, no-err 0.851/0.953/6.06 s,
  mistakes 7/32 50 % 2.8 FA/h; GT AR (diagnostic) 0.795/0.910/5.26 s, err 0.711/0.851/5.54 s. Above B3 on all six step
  metrics with live AR (POS margins +0.005 / +0.016, within bootstrap CI). Causality 0/96; latency median 27.8 / p95 83.0 ms.
  Nothing changed after the test run. Details: industreal_psr_v11/results/v13/README.md.

## 2026-11-08 — v14 attempt: more mistakes than v13 without losing other metrics (train+val only; test NOT run)
v13 release folder frozen first (psr_demo/v14/v13_folder_SHA256SUMS.txt, 356 files; verified unchanged afterwards).
Dev baseline (v13 settings, mistakes scored with the teammate's rules/matching): oracle AR train+val 14/38 (skipped 7/9,
order 5/8, install 2/21), prec 0.61, 2.8 FA/h; real live AR val 2/13; simulated live AR (v13 simulator) 7-9/38.
Everything below was tried on train+val only and REJECTED (no gain at equal precision / FA rate):
- Second ("shadow") filter for alerts only (PSR outputs unchanged), 8 settings (d0 20/30, theta .9, conf 6, stable 6,
  tau .03/.08): oracle 9-14/38, sim 5-9/38; no setting above v13. Union of all 14 variants: +0 oracle, +1-2 sim.
- Pin gate (a trigger may install a short pin only after a pin action within 30/60/120 s): identical to v13 (pin
  actions occur near every chassis step; the three short pins share one action).
- Incorrect installs from the binary verifiers (part reads not-attached after the install): GT-anchored 15/21 read low,
  but candidates anchored on triggers or on component-naming actions give AUC 0.61-0.67; 2 found costs ~22 FPs.
- 3-class verifiers (OOF): frame AUC incorrect-vs-absent 0.88 bracket, 0.77 screw, ~0.5 wheels/pins; alert rule best
  2/21 found at 25 FPs.
- Rear-wheel recipe (missing pulley / washer in the action sequence): with GT actions 14/47 correct wheels also lack a
  pulley segment in the window; not discriminative.
- Teammate's engine, actions only (no state, no ASD): val live 7/13 but precision 0.17, 33 FA/h; sim 3-5/38 at
  0.02-0.03. Kept only when v13's belief says the step is missing: no TP survives (the verifier itself misreads those
  parts). Raw union with v13: +2-3 TPs for ~140 FPs per setting.
- Root cause: perception of small / similar parts. With the rear chassis on, a missing front-rear pin reads attached
  (median log-odds +4.1, AUC 0.71; rear-rear 0.83). Upper bound with GT pin readings in the shadow filter (diagnostic):
  sim 9-12/38 but precision 0.10-0.12, 22-26 FA/h -> even a perfect pin verifier would not give a clean gain within
  the step-replay framework.
Conclusion: no fair configuration found that detects more mistakes without lowering precision / raising false alarms.
No v14 frozen, test not used (test count stays 12).

## 2026-11-08 (later) — v14 attempt 2: ASD allowed as an extra mistake cue (user decision; val only; test NOT used)
- Downloaded from data.4tu.nl (IndustReal, Apache-2.0): assembly_state_detection_model_weights.zip (used
  asd_best_IndustRealandSynthetic.pt) and ASD_results_IndustRealplusSynthetic_test.zip (only to check our run).
  ultralytics 8.3.40 installed with --no-deps (torch 2.5.1 / numpy 2.0.1 unchanged). Files in <project>/asd_download.
- Our run reproduces the released test predictions (2 recordings checked, first 900 frames): same top state on 185/185
  frames, confidences ~7 % lower, so fewer boxes pass 0.25. Stored all boxes >= 0.05 (v14/asd/val).
- Val (16 recordings, 13 mistakes, live AR; v13 = 2/13, prec 1.00, 0 FA/h). Added to v13's alerts:
  A teammate engine live AR + our ASD (hand-occlusion check off): 2/13, prec 0.20, 7.6 FA/h (alone 1/13).
  B ASD error_state persisting -> incorrect install of the last reported part: 2/13 (+0), extra alerts all false.
  C ASD stable-state replay through the rules: 3-5/13 but prec 0.08-0.14, 13-57 FA/h.
- Why: ASD detects on 9 % of val frames (state correct on 80.5 % of those) and says error_state on 1 of 2,406 GT error
  frames (0.20 % of non-error frames). It cannot see incorrect installs and its sparse states make false skip alerts.
- Conclusion: no ASD rule adds mistakes on val without lowering precision / raising false alarms. No v14 frozen; the
  test ASD run was stopped unused; test count stays 12.

## 2026-11-08 (evening) — v14 PRE-REGISTRATION: ASD as extra whole-state evidence in the v13 filter (val only)
Already seen before writing this (disclosed): 7 exploratory configs on val (W 0/0.3/1/3, CMIN 0.3/0.5/0.7, EPS 0.05,
ASD every frame); best W 1 / CMIN 0.3: live val 0.877/0.891/6.9 s (v13 0.861/0.883/7.7 s), err 0.753/0.775
(v13 0.733/0.764); oracle-AR val mistakes 4/13 vs 5/13; live mistakes 2/13 = v13. Hand-occlusion dropping: rejected.
Dev data: val only (16 recordings, 5 subjects). The ASD was trained on train (in-sample there) and its checkpoint was
picked on val mAP -> val is mildly optimistic for ASD; disclosed with any result.
- Method (v13 + ASD): at a frame where the official ASD's top box has conf >= CMIN and names an assembly state c
  (not background / error_state), every belief state s gets + W * log EPS if code(s) != c. Steps, rules, alerts, mistake
  logic otherwise exactly v13. Causal (ASD output of the current frame only).
- Grid (fixed now): W {0.3, 1, 3} x CMIN {0.3, 0.5, 0.7} x EPS {0.02, 0.05, 0.1} x ASD rate {every frame, odd frames
  only (alternating with the verifiers, for latency)} = 54 configs.
- Criterion (fixed now): v13 criterion on val live AR + v13 criterion on val oracle AR. Eligible only if live-AR val
  mistakes are not worse than v13 (TP >= 2, FA/h <= 0.0 + 1 alert allowed? NO: FA/h must be <= v13's 0.0 and TP >= 2).
  Argmax of eligible configs is frozen as v14 WHATEVER it is. Nested leave-one-subject-out CV over the 5 val subjects
  reported. Latency must stay below the 100 ms p95 budget used so far (else the odd-frame variant with the best score).
- Then ONE test run (13th test use), all metrics reported, nothing changed afterwards. If no eligible config beats v13
  on the criterion, no v14.
- v14 selection result (v14/asd_select.py, 54 configs + v13 control): argmax W 1 / CMIN 0.3 / EPS 0.05 / every frame,
  total 3.1010 (v13 2.9981), eligible (live val mistakes 2/13, 0 FA/h = v13). Val live 0.877/0.891/6.9 s, err
  0.753/0.775 (v13 0.861/0.883/7.7 s, 0.733/0.764); val oracle 0.878/0.919/5.3 s (v13 0.862/0.904/6.2 s), oracle
  mistakes 4/13 prec 0.80 (v13 5/13, 1.00). Nested LOSO over the 5 val subjects: live 0.861/0.882/7.0 s, err
  0.753/0.775; oracle 0.862/0.919/5.3 s; live mistakes 2/13 but 3.8 FA/h (fold 05 picks W 3). -> gains on delay and
  error recordings; all-recordings POS/F1 flat in CV; mistake FA not robust. Frozen as committed.
- v14 release folder <project>/industreal-psr-v14 (copy of v13 + psr/v14.py, eval_v14.py, tools/run_asd.py). With ASD
  weight 0 it reproduces v13's val steps and alerts exactly. Sanity check with the FINAL verifiers on val (in-sample for
  them, not the criterion): v14 0.921/0.931/4.1 s, err 0.852/0.861, mistakes 4/13 vs v13 0.938/0.951/4.4 s,
  0.898/0.914, 7/13 -> ASD hurts when the verifiers are near-perfect; disclosed as a risk.
- Latency rule applied: every-frame argmax p95 103.6 ms > 100 ms -> best odd-frame config w 1 / cmin 0.3 / eps 0.02 /
  rate 2 (total 3.0940, eligible): median 35.5, p95 78.6, mean 50.4 ms, peak GPU 602 MB (AR compute excluded; other
  apps were using ~30 % GPU). Causality 0/96 mismatches. Frozen 2026-11-08T14:44 (model/SHA256SUMS incl. ASD weights).
- **v14 TEST (single run, 13th test use)**, live AR: all 0.812/0.904/6.32 s, err 0.747/0.839/7.33 s, no-err
  0.870/0.960/5.42 s, w/err labels 0.769/0.878; mistakes 9/32 (skipped 2/10, order 4/4, install 3/18), prec 0.53,
  3.2 FA/h (17 alerts, 8 false). GT AR (diagnostic): 0.803/0.913/5.0 s, err 0.728/0.858/5.02 s, mistakes 7/32 50 %
  2.8 FA/h. vs v13 live: step metrics better or equal (err POS/F1 equal), +2 mistakes, precision up, but FA/h
  2.8 -> 3.2 (one more false alarm). Bootstrap all POS [0.73, 0.89]. Nothing changed after the test run.
  Details: results/README.md.
