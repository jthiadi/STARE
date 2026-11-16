# STARE: State Tracking and Action Recognition for Error Detection in Procedural Assembly Tasks

J. N. Hartono, J. Thiadi, S. Indrawan. Department of Computer Science, National Tsing Hua University.

STARE watches an egocentric assembly or maintenance video and, as it plays, recognises the worker's actions, the
completed procedure steps and the mistakes. Every output at time *t* uses only frames up to *t*.

```
video, 10 fps ─┬─► Action recognition    causal verb × noun TCN on DINOv2 + VideoMAE features (every frame)
               │                         + final label per action: 3 fine-tuned V-JEPA 2 + VideoMAE (shown to the worker)
               ├─► Visual state verifier  even frames: localizer + 2 DINOv2-base part verifiers; odd frames: IndustReal ASD
               └─► State tracker          probability over assembly states; a step is reported once it is likely,
                                          stable and visually confirmed
                   Mistake detection      task graph + procedure engine: wrong order, skipped step, wrong part,
                                          incorrect installation, early warning
```

Formulas and frozen settings: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Results (paper Tables 1-8)

All IndustReal numbers are on the 32 official test recordings; settings were chosen on train / validation and each
test score was computed once. One command recomputes every number below on a CPU and checks it (see
[Reproduce the paper's numbers](#reproduce-the-papers-numbers)).

**Table 1. Action recognition on IndustReal** (top-1 %, 3,678 test actions). *Official*: true start and end given
(the IndustReal protocol). *Live*: the system finds the actions itself from past frames; a true action counts if the
detection that overlaps it most has the right label.

| Method | Official | Live |
|---|---|---|
| SlowFast (IndustReal paper) | 60.39 | – |
| MViTv2 (IndustReal paper, RGB) | 65.25 | – |
| Fine-tuned V-JEPA 2 ViT-L (hand crop, single run) | 70.20 | – |
| STARE, causal TCN | 55.74 | 53.72 |
| **STARE, final label (ensemble)** (val 73.81) | **75.07** | **55.52** |

**Table 2. What each part of the final label adds** (official protocol, test; all settings fixed on validation)

| Step | Top-1 |
|---|---|
| Fine-tuned VideoMAE alone | 62.18 |
| Fine-tuned V-JEPA 2 (best single run) | 70.20 |
| Mean of 3 V-JEPA 2 runs | 72.76 |
| + VideoMAE | 73.25 |
| + TCN (w = 0.95) | 73.74 |
| + order prior (final label) | **75.07** |

**Table 3. Action recognition on MECCANO** (official protocol, top-1 %, 2,823 test actions, 61 classes, RGB)

| Method | Top-1 |
|---|---|
| SlowFast (MECCANO paper, CVIU 2023) | 45.16 |
| Swin3D-B (MECCANO challenge winner) | 50.80 |
| **STARE, final label, retrained on MECCANO** (val 66.12) | **62.45** |

**Table 4. Procedure step recognition on IndustReal** (official evaluator). STARE's delay is end to end: every action
label reaches the tracker only after its measured compute on one RTX 2080 Ti.

| Method | POS ↑ | F1 ↑ | Delay ↓ |
|---|---|---|---|
| B3 (IndustReal paper) | 0.797 | 0.883 | 22.4 s |
| STORM-PSR (CVIU 2025) | 0.812 | 0.901 | 15.5 s |
| **STARE** | **0.819** | **0.907** | **6.5 s** |

**Table 5. Procedure step recognition on MECCANO** (7 test videos, STORM-PSR's evaluator)

| Method | POS ↑ | F1 ↑ | Delay ↓ |
|---|---|---|---|
| IndustReal B3 (as reported by STORM-PSR) | 0.354 | 0.545 | 99.8 s |
| STORM-PSR | 0.377 | 0.497 | 88.6 s |
| **STARE** | **0.615** | **0.609** | **26.6 s** |

**Table 6. Ablations, end to end** (each setting fixed before its single test run; mistakes found / 32, typed)

| Setting | POS / F1 / delay | Mistake recall |
|---|---|---|
| STARE (full) | 0.819 / 0.907 / 6.5 s | 0.28 (9 / 32) |
| w/o action recognition | 0.772 / 0.893 / 7.9 s | 0.25 (8 / 32) |
| w/o state detector | 0.809 / 0.903 / 7.7 s | 0.22 (7 / 32) |

**Table 7. Step recognition by action-recognition input, end to end**

| AR input | AR top-1 (official) | AR delay | POS / F1 / delay |
|---|---|---|---|
| STARE, causal TCN | 55.74 | 0.8 s | 0.819 / 0.907 / 6.5 s |
| STARE, final label (ensemble) | 75.07 | 2.5 s | 0.764 / 0.879 / 7.2 s |
| Oracle (annotated actions) | 100 | 0.1 s | 0.814 / 0.915 / 5.0 s |

**Table 8. Mistake detection without the mistake type** (32 test mistakes, ±30 s; baselines run with the PROVIA
authors' code, 3 seeds, alarm threshold matched to STARE's false-alarm rate on validation)

| Method | Recall | Precision | False alarms / h |
|---|---|---|---|
| Causal-TCN (PROVIA's implementation) | 0.09 | 0.23 | 4.0 |
| Controlled PREGO (PROVIA's re-implementation) | 0.05 | 0.16 | 3.6 |
| PROVIA | 0.10 | 0.19 | 5.5 |
| **STARE** | **0.25** | **0.67** | **1.6** |

Compute on one RTX 2080 Ti: action recognition 87.1 ms per frame, state tracking 31.8 ms (95th percentile 67.8 ms),
final label 1.23 s per action. Online but not real time: 119 ms per frame, above the 100 ms that 10 fps needs.

Notes on the selection:
- The final-label recipe was chosen on validation (73.81 %) and fixed before its test score. Earlier offline variants
  of it had been scored on the test set during development, so 75.07 % is not the first look at that test set,
  although nothing in the recipe was chosen on it.
- The deployed TCN was trained on train + validation, so its validation scores are not held out; every test number is.
- `metrics/*/PREREGISTRATION*.md` are the rules written before each test run. They are kept as written and use the
  table numbers of their time, not the paper's.

## Repository layout

```
system/
  assistant/           action recognition, final labels, mistake engine, live pipeline and card UI
                       (the teammate's code, branch verb-noun-tcn, plus our changes listed in OUR_CHANGES.md)
    live/              runs every model on a video (GPU) and serves results to the UI
    meccano_action_end/  MECCANO action recognition (Table 3)
  psr_v14/             the frozen state tracker (verifiers, ASD evidence, step model, settings) with its selection code
  v14_demo/            frame-by-frame demo page; frames_compat.py reads frames from .mp4
  server/run_frozen.py run a psr_v14 script when the dataset has videos but no JPEG frames
metrics/
  reproduce_all.py     recompute and check every number in Tables 1-8 (CPU)
  ar/                  Tables 1-3
  e2e/                 Tables 4, 6, 7: measured compute, delayed label streams, validation re-tune, one test score
  meccano/             Table 5
  provia/              Table 8
  psr/                 settings files for the ablations and older PSR checks
demo_cards/            card UI for the 32 test recordings (build_v14_tcn.py, render_cards.py, serve.py)
tools/                 fetch third-party code, pack / unpack the large files
env.sh                 environment variables for a portable Python in env/ (optional)
```

## Setup

1. **Python 3.12** with PyTorch (CUDA build for the GPU steps; the CPU reproduction also imports it), then:
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
   pip install -r system/assistant/requirements.txt scikit-learn==1.7.2 weighted-levenshtein ultralytics==8.3.40 fastapi uvicorn
   ```
   On a shared server, `source env.sh` keeps Python, packages and caches inside `env/` (create it there first).
2. **Datasets** (from their authors, under their terms):
   - IndustReal (data.4tu.nl): the `train/`, `validation/`, `test/` annotation folders, and the 84 recordings as
     `system/assistant/data/generated/industreal/videos/<rid>.mp4` (`python -m scripts.data.download_industreal` from
     `system/assistant` downloads them).
   - IndustReal `part_geometries` (CAD states, only for the UI's 3D inset) in
     `system/assistant/output/assist/cad/part_geometries/`.
   - MECCANO: annotations and `MECCANO_RGB_Videos` in `data/MECCANO/` (or set `MECCANO_ROOT`).
3. **Third-party code** at the commits we used (PROVIA, STORM-PSR, MECCANO step labels):
   ```bash
   bash tools/fetch_third_party.sh
   ```
4. **Saved model outputs and weights** (too large for Git): download `stare_outputs.tar` (0.3 GB, needed to reproduce
   the tables) and `stare_weights.tar` (8.5 GB, needed only to rerun the models) from
   <https://huggingface.co/jthiadiii/STARE>, then unpack them inside this folder:
   ```bash
   hf download jthiadiii/STARE stare_outputs.tar stare_weights.tar --local-dir .   # or download from the page
   python tools/unpack_artifacts.py stare_outputs.tar stare_weights.tar      # checks every file's SHA-256
   ```
5. **Public backbones** for the GPU steps (DINOv2-small, VideoMAE-SSv2, V-JEPA 2 ViT-L):
   `python -m scripts.data.download_models` from `system/assistant`.

## Reproduce the paper's numbers

CPU only, about 30 minutes on 8 cores; needs steps 1-4 above (`stare_outputs.tar`, no weights):

```bash
python metrics/reproduce_all.py --data <IndustReal root with train/ validation/ test/>
```

It prints `OK` or `MISMATCH` for every number of Tables 1-8 and ends with `ALL NUMBERS REPRODUCED`. Per-table scripts and
details: [`metrics/ar/README.md`](metrics/ar/README.md), [`metrics/e2e/README.md`](metrics/e2e/README.md),
[`metrics/meccano/README.md`](metrics/meccano/README.md), [`metrics/provia/README.md`](metrics/provia/README.md).

## Rerun the models (GPU)

- **Whole system on a video** (actions, final labels, verifiers, ASD, tracker, mistakes), from `system/assistant`:
  `python -m live.run_video --test 0/2` (one shard per GPU). Outputs go to `system/assistant/output/assist/<rid>/`.
- **Measured compute and the validation ensemble** (Tables 4, 6, 7): `bash metrics/e2e/run_gpu.sh`.
- **MECCANO** step recognition: `metrics/meccano/README.md`; action recognition:
  `system/assistant/meccano_action_end/README.md`.
- **PROVIA and baselines**: `metrics/provia/README.md`.

## Retrain from scratch

- **Part verifiers, localizer, step model, tracker selection**: `system/psr_v14/README.md`
  (`train_verifier.py`, `run_pipeline.py`, `selection/`).
- **V-JEPA 2 and VideoMAE fine-tuning** (final label): `system/assistant/README.md`, "Label given action segments with
  the V-JEPA 2 ensemble" (`scripts.features.vjepa2_finetune`, `scripts.features.finetune_videomae_ar`).
- **Features and cross-fitted backbones** (from `system/assistant`): `python -m scripts.segmentation.run_official_split_eval`
  downloads the data, extracts the DINOv2 features and fine-tunes the two cross-fitted DINOv2 backbones (it then also
  trains two earlier, non-causal TCNs that STARE does not use). The VideoMAE side:
  `python -m scripts.features.finetune_videomae_ar --cache`, then `--fold 0` and `--fold 1` (cross-fitted backbones),
  `--causal-fold 0` and `--causal-fold 1` (their past-only features), and
  `python -m scripts.features.extract_videomae_features --causal --output output/videomae_ssv2_features_causal`.
- **Causal TCNs**. The settings below are read from the saved runs
  (`output/official_split_eval/tcn_{verb,noun}_causal/ar_object_paper_metrics.json`); this exact command line was not
  rerun. From `system/assistant`, with `TRAIN_VAL` = the 36 train + 16 validation recordings and `TEST` = the 32 test
  recordings of `scripts/segmentation/run_official_split_eval.py`:
  ```bash
  E=output/official_split_eval
  for t in verb noun; do
    python -m scripts.segmentation.train_ar_object_tcn --model causal_tcn --target $t $([ $t = noun ] && echo --verb-aux) \
      --train-recordings $TRAIN_VAL --test-recordings $TEST \
      --features output/ar_frame_features --extra-features output/ar_frame_features_hires output/videomae_ssv2_features_causal \
      --crossfit $E/finetune/feats_fold0 $E/finetune/feats_fold1 \
      --crossfit $E/vmae_finetune/feats_fold0_causal $E/vmae_finetune/feats_fold1_causal \
      --crossfit-meta $E/finetune/meta.json --subject-folds --folds 6 --seeds 3 --epochs 80 --cosine --warp \
      --hidden 192 --layers 10 --stages 4 --full-nouns --host-features --output $E/tcn_${t}_causal
  done
  ```

## Demo

**Rendered demo videos from IndustReal test set**: https://drive.google.com/drive/folders/1gOOiDWF7yPbYWn1aHRhmxn8siRY1gwmt?usp=drive_link

## Credits and licenses

- IndustReal dataset, its PSR evaluator and assembly state detector: Schoonbeek et al. (Apache-2.0 code and weights;
  dataset under its own terms). MECCANO: Ragusa et al., under its own terms.
- STORM-PSR (Apache-2.0) and the MECCANO step labels; PROVIA (MIT): fetched by `tools/fetch_third_party.sh`.
- Backbones: DINOv2 (Apache-2.0), V-JEPA 2 and VideoMAE (see their licenses; the released VideoMAE weights are
  CC BY-NC 4.0, so the fine-tuned VideoMAE in `stare_weights.tar` is for non-commercial use).
- `system/psr_v14/NOTICE.md` lists the files copied into that package.
