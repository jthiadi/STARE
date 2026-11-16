# STARE on MECCANO: procedure step recognition (paper Table 5)

A second dataset: MECCANO (toy-motorbike assembly, 20 egocentric videos, 12 fps; official split 11 / 2 / 7), with the
STORM-PSR authors' PSR annotations and evaluator.

## Result (7 MECCANO test videos, STORM-PSR's evaluator)

| Method | POS ↑ | F1 ↑ | Delay ↓ |
|---|---|---|---|
| IndustReal baseline (as reported by STORM-PSR) | 0.354 | 0.545 | 99.8 s |
| STORM-PSR (Schoonbeek et al., CVIU 2025) | 0.377 | 0.497 | 88.6 s |
| **STARE (ours)** | **0.615** | **0.609** | **26.6 s** |

Per-video details: [`results/retuned/meccano_psr.json`](results/retuned/meccano_psr.json).

## MECCANO configuration of STARE
Same architecture and code as on IndustReal (causal action model, part verifier, causal state filter with step model
and triggers), trained on MECCANO's 11 train videos:
- Action recognition: causal TCN over the 61 MECCANO actions on frozen DINOv2-small (3 views) + VideoMAE-SSv2 (causal
  clips) features.
- One DINOv2-base part verifier for MECCANO's 17 parts; localizer prototypes from MECCANO's training state boxes.
  No assembly state detector (none is released for MECCANO).
- Step model and triggers trained on MECCANO train (trigger rule: `PREREGISTRATION.md`).
- Reporting settings selected on the MECCANO validation split (`PREREGISTRATION_retune.md`, `results/retune.json`):
  confirmation threshold 5.0, emission temperature 0.1, drift prior 1e-4, visual recovery after 12 contradicting
  readings. All other filter and step-model settings as on IndustReal.

Record: the fixed rules of every stage and the full selection grid are in the two `PREREGISTRATION*.md` files; the
run with IndustReal's reporting settings, made before the validation selection, is kept in `results/archive/`.

## Reproduce
CPU, from the saved model outputs (`work/ar`, `work/vprobs`, `work/steps`), part of `metrics/reproduce_all.py`:
```bash
python metrics/meccano/stare_lite.py run          # uses the validation-selected settings (results/retune.json)
```
From scratch (GPU; `data/MECCANO`: videos, annotations, `frames_960` frame cache):
```bash
python metrics/meccano/features.py 0/2 & python metrics/meccano/features.py 1/2   # frozen features
python metrics/meccano/stare_lite.py steps        # triggers + step model (train only)
bash metrics/meccano/run_vision.sh                # localizer prototypes, crops, verifier
python metrics/meccano/stare_lite.py tcn          # causal TCN
python metrics/meccano/stare_lite.py vprobs 0/1   # verifier readings, val + test
python metrics/meccano/stare_lite.py retune       # select the reporting settings on validation
python metrics/meccano/stare_lite.py run          # tracker + STORM-PSR evaluator
```
