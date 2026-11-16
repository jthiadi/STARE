# STARE vs PROVIA: mistake detection on IndustReal (paper Table 8)

PROVIA (Wen et al., arXiv 2609.20638, 2026) is, to our knowledge, the only published mistake detector evaluated on
IndustReal. It raises untyped alarms, so both systems are compared **without the mistake type**. Rules fixed before
running: [`PREREGISTRATION_untyped.md`](PREREGISTRATION_untyped.md) (with an amendment, written before any PROVIA
score: the released models could not be evaluated on our inputs, see below).

## PROVIA, as run here
- Code: `third_party/PROVIA`, commit 178e78f, unchanged (verified file by file). IndustReal conversion with their own
  converter (`make_manifest.sh`): 501 completions, test 194 steps / 17 mistakes, as in their paper.
- Features: their DINOv2 ViT-S/14 extractor at 8 Hz (`run_provia.sh`).
- Models: the released IndustReal models were refused by PROVIA's own provenance check (one maintenance event type
  differs when the procedure language is rebuilt from our IndustReal copy, whose videos are re-encoded from the
  dataset frames). The 3 seeds were therefore **retrained with the authors' unchanged scripts and settings**, with the
  released step segmenter (`run_provia_train.sh`). Their test step-level AP / AUROC: 0.150 / 0.636 (mean of 3 seeds),
  paper 0.153 / 0.625.

## Result: our 32 test mistakes, untyped (±30 s, any type, one-to-one; alarms within 30 s merged)

| Setting | Found / 32 | Precision | False alarms / h | Found by type (install / order / skipped) |
|---|---|---|---|---|
| **STARE** | **8** | **0.67** | **1.6** | 4/18 · 3/4 · 1/10 |
| PROVIA, threshold matched to STARE's validation false-alarm rate (1.89 / h) | 3.3 ± 1.5 | 0.19 | 5.5 ± 1.4 | |
| Causal-TCN, threshold matched as PROVIA | 3.0 ± 1.0 | 0.23 | 4.0 ± 0.0 | |
| Controlled PREGO, threshold matched as PROVIA | 1.7 ± 0.6 | 0.16 | 3.6 ± 0.0 | |
| PROVIA at its budget 0.1 / min (validation) | 4.0 | 0.18 | 7.3 | |
| PROVIA at 0.5 / min | 12.0 | 0.16 | 24.5 | |
| PROVIA at 1.0 / min | 15.7 | 0.15 | 35.2 | |

PROVIA and Causal-TCN: mean (± SD) over 3 seeds. Causal-TCN is the baseline with the best alarm-level recall on
IndustReal in PROVIA's paper; it was run with the authors' implementation in the same repository (`run_causal_tcn.sh`,
same features, step segmenter and splits). Its step-level test AP / AUROC in our run, 0.123 / 0.496, are below the
paper's 0.148 ± 0.040 / 0.616 ± 0.055 (PROVIA's own reproduce: 0.150 / 0.636 vs 0.153 / 0.625). Controlled PREGO
is the PROVIA authors' re-implementation of PREGO (Flaborea et al., CVPR 2024; `run_prego.sh`); its step-level test
AP / AUROC, 0.135 / 0.596, match the paper's 0.133 ± 0.010 / 0.612 ± 0.008. Typed, STARE finds 9 / 32 (the untyped rule merges STARE's simultaneous
end-of-recording alerts, hence 8). At comparable false-alarm rates STARE finds more mistakes with higher precision;
PROVIA finds more only at 15-20 times more false alarms.

**In PROVIA's own protocol** (17 incorrect installs, alarm must fall on the faulty step's decision): STARE finds
0 / 17 (`score.py`, `results/provia_protocol.json`): its incorrect-install alerts come at the end of the recording.
PROVIA's published recall there: 0.098 / 0.392 / 0.667 at 0.1 / 0.5 / 1.0 false alarms per minute.

## Reproduce
CPU, from the saved PROVIA outputs (`third_party/PROVIA/runs/industreal/model_seed*/*_candidates.npz`), part of
`metrics/reproduce_all.py`:
```bash
python metrics/provia/untyped.py
```
From scratch (GPU): `make_manifest.sh`, `run_provia.sh` (features), `run_provia_train.sh` (3 seeds + test evaluation).
