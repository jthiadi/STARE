# IndustReal action-end recipe on MECCANO

Question: does the IndustReal action-end recognition recipe (75.07 % test top-1 on IndustReal, past only) generalise to
MECCANO, with only the necessary dataset-specific changes?

Protocol (same as IndustReal): ground-truth segment boundaries, each segment labelled from its own frames and earlier
actions only; trained on the 11 official train videos, the one free choice (which MECCANO VideoMAE run) made on the 2
val videos, the 7 test videos scored once. 61 actions, top-1 over all annotated test segments (2,823).

## Result

| | IndustReal | MECCANO |
|---|---|---|
| Val top-1, full action-end recipe | 73.81 | 66.12 |
| **Test top-1, full action-end recipe** | **75.07** | **62.45** |
| Best single V-JEPA 2 run, test | 70.2 | 61.6 |
| Offline recipe (sees later frames), test | 77.2 | 62.24 |

Component breakdown on MECCANO (`ablation.json`, diagnostic only, nothing chosen from it):

| Test top-1 | |
|---|---|
| VideoMAE (off_s3) alone | 51.82 |
| V-JEPA 2 area k12 / hand k12 / hand k12 s1 alone | 57.88 / 60.86 / 61.60 |
| Mean of the 3 V-JEPA runs | 62.98 |
| Mean of all 4 members | 62.98 |
| + past-only TCN (0.05) | 62.98 |
| + order prior (0.2) | 62.88 |
| Full recipe (TCN + prior) | 62.45 |
| Past-only TCN alone | 32.66 |

## What is reused, what is new, what differs from IndustReal

Reused from `output/meccano` (trained earlier with the IndustReal settings: 12 layers, 10 epochs, batch 4 x 4, lr 2e-5,
no context): V-JEPA 2 `vjft_area_k12`, `vjft_hand_k12`; MECCANO VideoMAE runs `off_s2`, `off_s3` (`vmae_probs.npz`);
frame caches and hand landmarks (symlinked).

Trained here: V-JEPA 2 `vjft_hand_k12_s1` (seed 1, same settings, epoch 9 chosen on val); DINOv2 + past-only
VideoMAE-SSv2 per-frame features; past-only TCN (3 seeds, 80 epochs, no video-progress input).

Dataset-specific differences from IndustReal:
- 61 MECCANO actions (class 0 unused background slot), 12 fps, official MECCANO split.
- One 62-class past-only TCN on MECCANO instead of IndustReal's verb x noun pair of past-only TCNs.
- The VideoMAE member is the MECCANO fine-tuned VideoMAE (two runs, the better on val used).

Code changes vs `scripts/meccano`: package name, output folder (`output/meccano_action_end`), dataset path
(STARE's `data/MECCANO`), and `--prefix 0.0` added to `vjepa2.py` (needed by the current `vjepa2_finetune`).
`ablation.py` is new. Nothing under `scripts/` or `assist/` was edited.

## Reproduce

```bash
source env.sh && cd system/assistant    # from the STARE folder
python -m meccano_action_end.action_end --features      # GPU
python -m meccano_action_end.action_end --tcn           # GPU
python -m meccano_action_end.vjepa2 --crop hand --layers 12 --batch 4 --accum 4 --seed 1   # GPU
python -m meccano_action_end.action_end --val           # CPU
python -m meccano_action_end.action_end --test          # CPU, once
python -m meccano_action_end.ablation                   # CPU, diagnostic
```

Logs: `output/meccano_action_end/logs/`. Results: `action_end_val.json`, `action_end_test.json`, `ablation.json`.
