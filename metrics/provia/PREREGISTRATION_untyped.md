# STARE vs PROVIA, untyped mistake detection on IndustReal (fixed before running PROVIA)

Written 2026-11-11, before PROVIA was run on our data and before any untyped score was computed.

## Systems
- **PROVIA** (Wen et al., arXiv 2609.20638): code `third_party/PROVIA` (commit 178e78f, unmodified), the authors'
  released IndustReal models (3 seeds, from their Google Drive link), DINOv2 ViT-S/14 features at 8 Hz, run with their
  own scripts (`scripts/single_actor/extract_features.sh`, `scripts/evaluate_checkpoints.sh`). Causal, as published.
- **STARE**: the frozen system (`system/psr_v14`), its test alerts unchanged (`results/v14_predictedAR_test.json`).
  Its alerts are used without their type.

### Amendment (2026-11-11, before any PROVIA score was computed)
The released PROVIA models could not be evaluated on our inputs: PROVIA's own provenance check refuses them because
the procedure language rebuilt from our IndustReal copy differs from the authors' in one event type for the
maintenance task (`action_12+15`, decoded by the released step segmenter on one training video; our videos are
re-encoded from the dataset frames, so the features differ slightly). Assembly matches exactly. Therefore PROVIA's
3 seeds are **retrained by us with the authors' unchanged scripts and settings** (`scripts/single_actor/train_provia.sh`,
seeds 0-2, 1 GPU) on the inputs rebuilt from our data with the **released** step segmenter (`observer.pt`), and then
evaluated with their scripts (validation thresholds, test once). It is reported as "PROVIA, retrained with the
authors' code and settings".

### Addition (2026-11-11, before Causal-TCN was run)
**Causal-TCN**, the baseline with the best alarm-level recall on IndustReal in PROVIA's paper, is added: the authors'
implementation in the same repository (`scripts/baselines/single_actor.sh industreal causal_tcn <seed>`, unchanged),
seeds 0-2, on the same features, step segmenter and splits as PROVIA, scored exactly as PROVIA below (threshold matched
on validation to STARE's false-alarm rate; test once per seed).

### Addition (2026-11-11, before Controlled PREGO was run)
**Controlled PREGO**, the PROVIA authors' re-implementation of PREGO in the same repository
(`scripts/baselines/single_actor.sh industreal prego <seed>`, unchanged), seeds 0-2, scored exactly as PROVIA.

## Protocol 1: ours, untyped
- Ground truth: the 32 annotated test mistakes of our evaluator (teammate's `assist/evaluation/eval.py`, task-graph rules).
- Matching: an alarm matches an unmatched mistake of **any** type within ±30 s, closest first, one-to-one (the
  evaluator's matching with the type condition removed). Alarms within 30 s of each other count once (its collapsing).
- Reported: found / 32, precision, false alarms per hour, and found per mistake type.
- PROVIA's alarms: its sequential alarm rule (Shiryaev-Roberts, `alarm_protocol.py`) on its anytime mistake score; the
  threshold is chosen **on validation only**, as the threshold whose validation false-alarm rate under this protocol
  is closest to STARE's validation rate without exceeding it. Also reported: PROVIA at its own validation thresholds
  for 0.1 / 0.5 / 1.0 false alarms per minute (as in its paper). Mean ± SD over the 3 seeds.
- STARE is also given the typed result (9 / 32) for reference.

## Protocol 2: PROVIA's own (`metrics/provia/score.py`, `PREREGISTRATION.md`)
- 17 incorrect-completion mistakes, recall at false-alarm budgets, step-level AP / AUROC. STARE's numbers there are
  already computed (0 / 17); PROVIA's are its published values plus our re-run.

Both protocols are reported. Test is scored once per system and seed.
