# IndustReal action segmentation: verb x noun TCN

> **In the STARE repository.** This is the teammate's working README, kept as the record of how the action models were
> built and what was tried. The STARE paper and its reproduction are described in the top-level `README.md`. Scripts
> for experiments that feed none of the paper's results are not in this repository, so their commands below will not
> run here: `scripts/evaluate_release.py`, `scripts/run_release_pipeline.py` (the earlier offline release),
> `scripts/efficiency/`, `scripts/zeroshot/`, `assist/assembly101/`, the replay dashboard app (`python -m assist.dashboard`,
> `assist.tools.transcode_videos`), and the mistake experiments `scripts/mistakes/{render_wrong, render_train, tar_images,
> val_baseline, crop_prototypes, dual_qformer, execution_qformer, warn_vlm}.py`, plus the recognition experiments listed
> under "Other experiments". `OUR_CHANGES.md` lists them.

Frame-wise action recognition for full, untrimmed egocentric assembly videos
([IndustReal](https://github.com/TimSchoonbeek/IndustReal)). For every frame (10 fps) the model
predicts one of the 74 official IndustReal actions (`verb_noun`, e.g. `take_nut`, `tighten_nut`)
or `background`. No start/end boundaries are needed at inference time.
Optionally, SAM 3.1 hand and object masks are drawn on the overlay video.

```
video (10 fps)
 ├─ DINOv2-small, frozen ─────────── 3 views (full / centre / hands) + hi-res hands view
 ├─ DINOv2-small, fine-tuned ──────── hands crop, 2 cross-fitted backbones (noun + verb heads)
 ├─ VideoMAE-base (Something-Something-v2) ─ 1.6 s motion clips around every frame
 └─ VideoMAE-base, fine-tuned on IndustReal action clips ─ 2 cross-fitted backbones
        │
        ├─ verb TCN ensemble  (10 verbs + background)
        └─ noun TCN ensemble  (18 nouns + background)
                │
   P(verb) x P(noun) over the 74 valid verb_noun actions
   + Viterbi decoding (switch penalty, action-order prior from the training labels)
                │
   frame-wise actions  ->  actions.csv / segments.csv / overlay video (+ SAM 3.1 masks)
```

## Results (official IndustReal split)

Trained on the 36 official train videos, decoding chosen on the 16 val videos, scored once on the
32 test videos (full untrimmed videos, background counted as a class):

**Action segmentation** (the model finds the action boundaries itself):

| Test subset | Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| All 32 videos | 54.1 | 55.0 | 60.2 | 55.0 | 40.8 |
| All 32, background frames excluded | 52.3 | 54.5 | 61.4 | 59.0 | 46.6 |
| Assembly videos | 53.5 | 55.2 | 59.6 | 54.7 | 40.2 |
| Maintenance videos | 55.8 | 54.6 | 61.7 | 56.0 | 42.5 |

Metrics are the standard temporal action segmentation ones (`scripts/segmentation/tas_metrics.py`): frame
accuracy, segmental Edit score and F1 at 10/25/50% IoU. For comparison, FACT (CVPR 2024) retrained on
exactly the same features, with its learning rate and frame step chosen on val, reaches 46.7 / 48.2 /
51.1 / 44.2 / 29.4 on the same 32 videos.

**Action recognition with ground-truth boundaries** (the IndustReal paper's protocol): each of the 3,678
annotated test segments is cut at its true start and end and classified among the 74 actions
(background excluded):

| Model | Val top-1 | Test top-1 |
|---|---|---|
| Verb x noun TCN (mean action probability inside the segment) | 56.7 | 62.2 |
| VideoMAE fine-tuned on the 36 train videos, one seed (16 frames, 3 temporal samplings) | 60.7 | 62.1 |
| TCN + VideoMAE fused (one seed, VideoMAE weight 0.7) | 66.7 | 69.5 |
| TCN + VideoMAE fused (two seeds, plus multi-window scoring of long segments) | 67.3 | 70.0 |
| **+ action-order prior** (VideoMAE weight 0.8, prior strength 0.3, Viterbi) | **70.6** | **72.6** |
| + VideoMAE refit on train + val videos (*not validated*, see below) | | *74.8* |
| V-JEPA 2 ViT-L, last 12 of 24 layers fine-tuned, hand-centred crop (one run alone) | 68.3 | 70.2 |
| **V-JEPA 2 fine-tuned (4 runs) + VideoMAE + action-order prior** (research, not in `release/`, see below) | **74.8** | **77.2** |
| V-JEPA 2 fine-tuned (8 runs) + VideoMAE + action-order prior (*second test look*, see below) | 75.3 | *77.6* |
| V-JEPA 2 fine-tuned (12 runs) + VideoMAE + action-order prior (*third test look*, see below) | 75.2 | *78.1* |
| *IndustReal paper: SlowFast, RGB* | | *60.4* |
| *IndustReal paper: MViTv2, RGB* | | *65.3* |
| The live pipeline's models: past-only verb x noun TCN alone | 48.65 | 55.74 |
| The live pipeline's models: action-beginning V-JEPA 2 alone | 62.34 | 64.57 |
| **The live pipeline's models: TCN + V-JEPA 2 + order prior over earlier actions** | **67.32** | **69.63** |

**The live pipeline's models on the paper's test** (`scripts/recognition/pipeline_on_segments.py`). True start and end given, label
fixed at each segment's end, only the segment's own frames and earlier actions used. The blend weight (0.7) and prior (strength
0.2, smoothing 0.01) were chosen on the 16 val videos and test was scored once. V-JEPA 2 alone reproduces its saved run (64.57
against 64.55). This shows what the models can do with true boundaries, not what the assistant shows live: it gets no true
boundaries and uses weight 0.1, which gives 60.90 on this test even with true boundaries.

```powershell
python -m scripts.recognition.pipeline_on_segments --score    # GPU, about 45 min on an RTX 4050; needs the 960 x 540 frame cache
python -m scripts.recognition.pipeline_on_segments --select   # CPU, one look at test -> output/assist/pipeline_segments/result.json
```

**The action-end label (75.07 on the paper's test).** Scoring the models as above still differs from what runs live, so the assistant now has a second output with one shared function (`assist/extract/action_end.py`). The running
label stays close to the TCN while an action is in progress, because the clip models have seen little of it. When the action ends they have seen all of it, so they lead: three fine-tuned V-JEPA 2 runs (`area_k12`, `hand_k12`, `hand_k12_s1`) and the
fine-tuned VideoMAE, 3 temporal samplings each, weight 0.95 against the past-only TCN, order prior over earlier actions (strength 0.2, smoothing 0.01). It uses nothing after the action's end, and it runs once per action,
so it does not have to fit the running label's 0.5 s budget. The recipe was the best of the val candidates (73.81) and was locked before test was scored once. Test has been used before for related models (77.16, 77.6, 78.1), so
this is not a first look at that test set, though nothing here was chosen on it. Its output is written per video to `action_end.jsonl`; the engine and the dashboard do not use it yet, so mistake results are unchanged.

| Test, the same function | Boundaries | True actions labelled right | F1@10 / 25 / 50 |
|---|---|---|---|
| **Action-end label, the paper's test** (3,678 segments, top-1) | **given** | **75.07** | |
| Action-end label, as the assistant detects actions (3,736 detections) | detected | 55.46 | 57.37 / 53.73 / 39.78 |
| The TCN's own event label | detected | 53.75 | 55.52 / 51.13 / 36.95 |

The gap between 75.07 and 55.46 is mostly finding each action's start and end: better labels barely help when the boundaries are wrong. On detected actions a true action counts as right if the detection overlapping it most
has its label, and as wrong if none overlaps it. Speed on one RTX 2080 Ti (`--benchmark`): 1.21 s per action on a free GPU, 1.34 s sharing it with the running label's V-JEPA 2 updates; neither run includes the TCN pipeline
or all four models loaded at once. Replaying the test videos' detected actions through a one-at-a-time queue, a label arrives a median of 1.21 s after the action's end (95th percentile 2.12 s, slowest 3.85 s), or 1.34 / 2.73 /
5.09 s on a shared GPU. This is still not a true live stream.

```powershell
python -m scripts.recognition.action_end_benchmark         # CPU, one look at test, refused unless val gives 73.81 -> output/assist/action_end/paper_test.json
python -m assist.extract.run_action_end --spans            # CPU -> spans_test.json
python -m assist.extract.run_action_end --score --part 0/2 # GPU, one part per GPU, about 25 min on two RTX 2080 Ti -> probs/
python -m assist.extract.run_action_end --benchmark        # GPU; add --running in another process on the same GPU for the shared figure
python -m assist.extract.run_action_end --labels           # CPU, one look at test -> detected_test.json, <video>/action_end.jsonl
```


Every setting (which VideoMAE models, fusion weight, prior strength and smoothing, Viterbi or
per-segment posterior) is chosen on the 16 val videos; test is scored once. The action-order prior
counts, on the 36 train videos, how often each action follows each other one; each video's segments
are then labelled jointly instead of one at a time. The refit backbone is trained on train + val with
the epoch count fixed beforehand, so val cannot check it; it is reported separately. Unlike the paper's
models, the TCN also sees the video around each segment, and the order prior uses the neighbouring
segments of the same video.

**Tried, not adopted: a bigger frozen backbone.** V-JEPA 2 ViT-L (300M parameters, frozen, with an attentive
probe trained on the 36 train videos; `scripts/features/vjepa2_recognition.py`) reaches 54.8 val / 56.6 test
on its own, below the fine-tuned VideoMAE-base (60.7 / 62.1): adapting the backbone to IndustReal matters more
than its size. Offered to the fusion as an extra stream (`sequence_recognition --vjepa`), val keeps the sets
without it, so the result stays 72.6 (without the order prior val does pick it: 70.2 vs 70.0 test).

**Fine-tuned V-JEPA 2 (research result, 77.2).** Fine-tuning the last 8 or 12 of V-JEPA 2's 24 layers on the 36
train videos (`scripts/features/vjepa2_finetune.py`, 10 epochs, epoch chosen on val, 1.5-2 h per run on one
RTX 2080 Ti) lifts it from 54.8 to 66-68 val on its own, above VideoMAE. Four runs: work-area or hand-centred
crop (a square around both MediaPipe hands, so small parts get about twice the pixels), 8 or 12 layers; the
hand crop and 12 layers are each slightly better (val 68.3 best). Offered to the same val selection as the
VideoMAE streams, val picks the four runs averaged with one VideoMAE seed, VideoMAE / V-JEPA weight 1.0 (the TCN
drops out), order prior strength 0.5: **74.8 val / 77.2 test** (74.3 test without the order prior), scored on
test once. Train only, every setting chosen on val, so it compares with the paper's 65.3. It is not in the
released package or the live pipeline: four ViT-L passes per segment cost far more than VideoMAE-base.

After seeing 77.2, four more runs were trained (hand crop 12 layers seeds 1 and 2, area crop 12 layers seed 1,
hand crop 16 layers; val 66.9-68.0 each, none above the best first run). Val then picks all eight with both
VideoMAE seeds and their window scores, order prior 0.3: 75.3 val / 77.6 test. Test had already been seen
once, so 77.6 is a second look; 77.2 stays the clean number. More seeds give little (+0.4), so 80 % needs
something other than more of the same runs.

A third round tried three changes on the hand 12-layer run, each judged on val alone: 25 % extra context
around each segment (`--context 0.25`, val 69.3), 15 epochs (69.8), both together (two seeds, 69.5 / 69.7),
and a 320-pixel crop (56.3, left out of the fusion for that val score; V-JEPA 2 was pre-trained at 256).
With all twelve runs val picks VideoMAE + every V-JEPA 2 stream, order prior 0.3 (posterior decoding):
75.2 val / 78.1 test, a third test look. Val did not rise (75.3 before), so the +0.5 on test is not
confirmed by val. Overlay of these test predictions (offline: each video's segments decoded together):
`python -m scripts.recognition.render_online_overlay --recording 18_assy_2_5 --predictions
output/actions/vmaeft/sequence_recognition_vjepa_test_predictions.npz --model "offline (whole video)"
--output output/overlays/18_assy_2_5_vjepa_recognition.mp4`. Commands:

```powershell
python -m scripts.features.vjepa2_finetune --cache            # training downloads V-JEPA 2 (1.3 GB) on first use
python -m scripts.features.vjepa2_finetune --crop area --layers 8
python -m scripts.features.vjepa2_finetune --crop hand --layers 8
python -m scripts.features.vjepa2_finetune --crop area --layers 12 --batch 4 --accum 4
python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4
# second look: --seed 1 / --seed 2 of the hand 12-layer run, --seed 1 of the area 12-layer run, and:
python -m scripts.features.vjepa2_finetune --crop hand --layers 16 --batch 4 --accum 4
python -m scripts.recognition.sequence_recognition --vjepa   # -> output/actions/vmaeft/sequence_recognition_vjepa.json
```

**Tried, not adopted: a richer procedure prior** (`scripts/recognition/procedure_recognition.py`). Separate
next-action counts per task (assembly / maintenance) change nothing on val (70.6), and a prior on where in the
video each action happens lowers it (68.3 at full weight); val keeps the plain order prior.

## Setup

Requirements: Python 3.10, an NVIDIA GPU with at least 6 GB, ~2 GB disk for models
(~25 GB more to retrain: 4.6 GB of videos plus crops, features and TCN caches).

```powershell
python -m venv .venv
.venv\Scripts\activate                      # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m scripts.data.download_models           # DINOv2-small + VideoMAE-SSv2 into models/
```

Download the trained weights (`industreal_verb_noun_release.zip`) from this repository's
**Releases** page and unzip it into the repository root, so that you have:

```
release/
  verb_model.pt         verb TCN ensemble (18 networks)
  noun_model.pt         noun TCN ensemble (18 networks)
  decoder.json          action vocabulary + decoding settings chosen on val
  finetune/fold0.pt     fine-tuned DINOv2 hand-crop backbones
  finetune/fold1.pt
  finetune/meta.json
  vmae_finetune/fold0.pt  IndustReal-fine-tuned VideoMAE backbones (TCN features)
  vmae_finetune/fold1.pt
  vmae_finetune/full.pt   IndustReal-fine-tuned VideoMAEs on all train videos, two seeds (recognition)
  vmae_finetune/full_s3.pt
  vmae_finetune/refit_trainval.pt  VideoMAE refit on train + val (reported separately)
  recognition.json      recognition settings chosen on val (models, fusion weight, order prior)
  action_order_counts.npz  first-action and next-action counts on the train videos
  test_metrics.json     the numbers in the tables above
```

## Check the reported accuracy

With only `release/` (no training), this downloads the 32 official test videos and annotations, runs the
same code as `predict_actions` on each (every feature computed from scratch) and prints the segmentation
and recognition numbers above:

```powershell
python -m scripts.evaluate_release                         # all 32 test videos, ~4-5 h on a laptop GPU
python -m scripts.evaluate_release --videos 13_assy_0_1    # quick check on one video (~10 min)
```

Results are written to `output/evaluate_release/results.json`. The recognition numbers are
`recognition_top1_fused_with_order` (72.6, the headline) and
`recognition_top1_with_refit_not_validated` (74.8); `recognition_top1` is the share of the 3,678
annotated segments whose predicted action (verb and noun) is exactly right.

Run every command from the repository root (it is the import root: `python -m scripts.<folder>.<name>`,
see [Repository layout](#repository-layout)).

## Use the model on a video

```powershell
python -m scripts.predict_actions --video path\to\video.mp4 --overlay
```

Results go to `output/predict/<video name>/`:

| File | Content |
|---|---|
| `segments.csv` | one row per predicted action: `start_s, end_s, action, mean_confidence` |
| `actions.csv` | one row per frame (10 fps): `frame, time_s, action, verb, noun, confidence` |
| `predictions.npz` | `pred` / `conf` per frame, plus the raw verb and noun probabilities |
| `<video>_actions.mp4` | overlay video (with `--overlay`) |

Notes:
- Any frame rate works; the video is resampled to 10 fps first.
- The model was trained on IndustReal (HoloLens 2 head-mounted view, hands in the lower centre of
  the image, the IndustReal toy-car assembly). Other setups need retraining.
- A 5-minute video takes a few minutes on a laptop GPU. Intermediate features are cached in
  `output/predict/<video name>/features/`, so re-running is fast.

### With SAM 3.1 masks (optional, slow)

SAM 3.1 draws the hands and the object the predicted noun refers to. It is gated on Hugging Face:
request access at <https://huggingface.co/facebook/sam3.1>, then:

```powershell
pip install -r requirements-sam.txt
huggingface-cli login
python -m scripts.data.download_models --sam
python -m scripts.predict_actions --video path\to\video.mp4 --sam
```

This runs SAM 3.1 on every frame (about 10 s per frame on a 6 GB laptop GPU, i.e. hours per video;
it can be stopped and resumed).

### Label given action segments with the V-JEPA 2 ensemble (77.2 %)

The most accurate recognition model here: four fine-tuned V-JEPA 2 ViT-L runs (work-area or hand crop,
last 8 or 12 layers trained) plus the fine-tuned VideoMAE, probabilities averaged, then each video's
segments decoded together with the action-order prior; 77.2 % top-1 on the IndustReal test segments
(settings chosen on val, test scored once). It **labels** actions whose start and end you give; it does not
find them. Use annotated segments, or another model's, e.g. `segments.csv` from `predict_actions` above
(expect less than 77.2 % with predicted boundaries). It decodes a whole video at once, so it is for
recorded videos, not live use.

Download `industreal_vjepa2_recognition_part1.zip` and `industreal_vjepa2_recognition_part2.zip` from this
repository's **Releases** page and unzip both into the repository root:

```
release_vjepa/
  vjft_area_k8.pt  vjft_hand_k8.pt  vjft_area_k12.pt  vjft_hand_k12.pt   fine-tuned V-JEPA 2 layers + heads
  vmae_full.pt                                                           fine-tuned VideoMAE
  action_order_counts.npz                                                action-order prior (train videos)
  settings.json                                                          everything chosen on val
```

The base V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`, 1.3 GB, MIT) and VideoMAE models are downloaded from
Hugging Face on first use. Needs a GPU with about 8 GB.

```powershell
python -m scripts.recognition.predict_vjepa --video path\to\video.mp4 --segments path\to\segments.csv
```

`segments.csv` needs `start_s` and `end_s` columns (seconds; other columns are ignored). Results go to
`output/predict_vjepa/<video name>/labels.csv` (`start_s, end_s, action, confidence`), next to the frame
caches and hand landmarks made on the way. Each segment is read as 16 frames spread over it, so the frame
rate does not matter. Per segment: 4 models x 3 samplings of V-JEPA 2 ViT-L plus 3 of VideoMAE.
Checked: on the 3,678 annotated test segments this script gives 77.16 %, the same top action as the saved
results for every segment (probabilities within 0.0006); 54 minutes on one RTX 2080 Ti (~0.9 s per segment).

## Evaluate on / retrain with IndustReal

One script reproduces everything from scratch: downloads the 84 official videos and annotations
from 4TU.ResearchData, extracts all features, fine-tunes DINOv2, trains both TCNs, chooses decoding on
val, scores the test set and writes `release/`. Each step is skipped when its output exists, so it
can be interrupted and restarted.

```powershell
python -m scripts.run_release_pipeline
```

It takes roughly two days on a laptop GPU (feature extraction, ~3 h of VideoMAE fine-tuning per fold,
~2 h for each of the two full-data seeds and ~2.5 h for the train + val refit, each TCN ~2 h). Results:
`output/actions/vmaeft/results.json` (segmentation metrics, per-video accuracy),
`output/actions/vmaeft/recognition_fusion.json` (fusion without order prior),
`output/actions/vmaeft/sequence_recognition.json` (recognition with order prior, full val grid) and
`output/actions/vmaeft/test_predictions.npz`.

To redo only the recognition scoring from existing outputs (CPU, ~20 min):

```powershell
python -m scripts.recognition.fuse_recognition --actions-dir output/actions/vmaeft
python -m scripts.recognition.sequence_recognition --actions-dir output/actions/vmaeft
```

Overlay of a test video against the ground truth (and optionally SAM 3.1 detections made with
`python -m scripts.kit_sop.run_kit_card_detect --video <video> --output <folder>`):

```powershell
python -m scripts.segmentation.render_action_overlay --predictions output/actions/vmaeft/test_predictions.npz `
    --recording 13_assy_0_1 [--sam <folder>]
```

### Repository layout

Run every script as a module from the repository root, e.g.
`python -m scripts.recognition.sequence_recognition`.

```
scripts/
  run_release_pipeline.py   retrain everything and write release/
  evaluate_release.py       reproduce the test numbers from release/ alone
  predict_actions.py        run the released model on any video
  data/           downloads, AR_labels.csv -> frame labels / intervals
  features/       hand crops, DINOv2 + VideoMAE features and fine-tuning (shared by both tasks)
  segmentation/   verb and noun TCNs, verb x noun decoding, metrics, split lists, overlay video
  recognition/    TCN + VideoMAE fusion and the action-order prior (GT boundaries given)
  kit_sop/        SAM 3.1 kit / hand detection (used by --sam overlays) and the earlier
                  SOP step-checking experiments
monitor/          paths (settings.py) and the SOP / kit logic used by kit_sop/
tests/            unit tests (python -m unittest discover -s tests)
```

Useful pieces on their own:

| Script | Purpose |
|---|---|
| `scripts/data/tas_labels.py` | IndustReal start/end annotations -> one label per frame (later-starting action wins on overlaps) |
| `scripts/data/ar_labels.py` | `AR_labels.csv` rows as time intervals, and the kit object each action refers to |
| `scripts/segmentation/tas_metrics.py` | frame accuracy, Edit, F1@{10,25,50} |
| `scripts/segmentation/train_ar_object_tcn.py` | TCN training (`--target verb` or noun), k-fold ensemble, saves `model.pt` |
| `scripts/segmentation/combine_verb_noun_actions.py` | verb x noun combination, decoding tuned on val, writes `decoder.json` |
| `scripts/features/finetune_videomae_ar.py` | VideoMAE fine-tuning on IndustReal clips (cross-fitted folds, full-data model, clip scoring) |
| `scripts/recognition/fuse_recognition.py` | TCN + VideoMAE fusion for the recognition protocol, weight chosen on val |
| `scripts/recognition/sequence_recognition.py` | action-order prior on top of the fusion, every setting chosen on val |
| `scripts/evaluate_release.py` | reproduce the test numbers from `release/` alone |
| `scripts/segmentation/run_official_split_eval.py` | official train/val/test video lists |
| `scripts/segmentation/render_action_overlay.py` | overlay video of predicted vs ground-truth actions |

### Other experiments (not used by the results above)

Kept so the "tried" statements can be checked; none feeds the released model or the live system. Numbers from
the train + val pooled runs (72.81 / 75.83) are not headline results, since val then no longer chooses settings.

| Script | What it tried |
|---|---|
| `scripts/recognition/cv_trainval.py` | recognition with train + val pooled (subject 4-fold CV for settings) |
| `scripts/recognition/online_eval.py` | the pooled model labelled online (each action at its end, past-only order prior) |
| `scripts/recognition/early_eval.py` | how accurate the causal model is while an action is still under way |
| `scripts/recognition/boost_accuracy.py` | error analysis and an assembly-state prior on the action label |
| `scripts/recognition/distill_base.py` | distilling the fused teacher into one VideoMAE-Base |
| `scripts/recognition/compare_finals.py`, `score_extra_final.py` | 2 vs 3 VideoMAE refits; scoring an extra refit |
| `scripts/efficiency/` (`cost`, `ladder`, `student`) | accuracy vs inference cost, down to one distilled VideoMAE-Small |
| `scripts/zeroshot/zeroshot_recognition.py` | training-free recognition (no IndustReal training) |
| `scripts/segmentation/fuse_videomae_segmentation.py`, `relabel_segments.py` | VideoMAE scores fused into segmentation; relabelling predicted segments |
| `scripts/mistakes/detect.py`, `mistsense.py`, `dual_qformer.py`, `execution_qformer.py`, `crop_prototypes.py` | learned heads for incorrectly installed parts on frozen features (best test F1 0.24) |
| `scripts/mistakes/warn.py`, `warn_vlm.py`, `order_check.py`, `render_hands_overlay.py` | pre-action warnings from procedure rules, a local VLM (Qwen3-VL-2B) and hand overlays |

### Memory

All long jobs run at below-normal priority under a memory guard
(`scripts/segmentation/run_subject_holdout_eval.py`): a step is stopped when free RAM drops below 0.35 GB or free
commit below 1 GB, and restarted (resuming) once memory recovers. On a 16 GB laptop, close browsers
and other large apps while training.

## Online assembly assistant (`assist/`)

Online (causal) procedure monitoring on top of the action model: what the worker is doing, which steps are
done, and plain-language alerts for skipped steps, wrong order and bad installs, with early warnings when a
part is picked up too early.

```
video -> causal AR (verb x noun TCN + V-JEPA 2) | MediaPipe hands | IndustReal ASD (released predictions)
      -> event builder -> procedure engine (task graph) -> alerts {title, why, fix_steps, rule, evidence}
```

Dataset-specific pieces are config only: `assist/configs/industreal.yaml` (paths, thresholds),
`assist/configs/industreal_catalog.yaml` (which actions belong to which step, hand-written) and
`assist/configs/industreal_graph.yaml` (the task graph the engine reads, drafted by `mine_graph`).
The released ASD predictions are read from the sibling STORM-PSR checkout (`../STORM-PSR-IndustReal`).

**What the live system uses** (the demo, the dashboard's "Live system" and every `pred_vjepap_asd` number below;
`pred_asd` is the earlier live system, with TCN actions only):

| Input | What | Where |
|---|---|---|
| Actions | Causal verb x noun TCN ensemble: DINOv2-small (3 views + hi-res hands), 2 fine-tuned DINOv2 hand-crop backbones, VideoMAE-SSv2 and 2 IndustReal-fine-tuned VideoMAEs, every clip ending at the current frame; `P(verb) x P(noun)` per frame, no Viterbi. Every 0.5 s, each label is pulled 10% towards V-JEPA 2 retrained on action beginnings (`w` 0.1, see "Action-beginning V-JEPA 2" below). 48.8 % of test frames right, 43.2 % at action end. Together on one RTX 2080 Ti: 14.6 fps (68.6 ms / frame), 164 ms per V-JEPA 2 update; details under "Action-beginning V-JEPA 2" | `assist/extract/run_ar.py` -> `ar.jsonl`, `assist/extract/run_ar_vjepa.py --variant prefix` -> `ar_vjepa_prefix.jsonl` |
| State | The released IndustReal state-detector (ASD) predictions (test videos; on val, our run of its published weights) | `assist/extract/run_asd.py` -> `asd.jsonl` |
| Hands | MediaPipe hand landmarks (only to tell when the hands cover the assembly) | `assist/extract/run_hands.py` -> `hands.jsonl` |
| Engine | Task graph + rules; thresholds in `assist/configs/industreal.yaml` | `assist/run_engine.py` |

Not used live: the offline segmentation TCN (54.1, decodes the whole video) and the 72.6 / 77.2 recognisers (they
need each action's true start and end), the earlier online V-JEPA 2 labels (an optional run, `pred_vjepa_asd`,
below), our own YOLO state detector, and the anomaly-detection and per-part state experiments (below).

```powershell
# 1. inputs for the 32 test videos (CPU: from cached causal TCN outputs and MediaPipe landmarks)
python -m assist.extract.run_ar --video all
python -m assist.extract.run_asd --video all
python -m assist.extract.run_hands --video all
#    on the GPU server instead: recompute AR from the video and log fps / ms per frame
python -m assist.extract.run_ar --video 18_assy_2_5 --compute --force

# 2. task graph draft from the correct train + val recordings (then review by hand)
python -m assist.tools.mine_graph

# 3. engine + evaluation (oracle = annotated actions and state; live system = TCN + V-JEPA 2 actions + ASD)
#    needs ar_vjepa_prefix.jsonl, from the "Action-beginning V-JEPA 2" commands below
python -m assist.run_engine --split test --actions gt --state gt
python -m assist.run_engine --split test --actions pred_vjepap --state asd
python -m assist.evaluation.eval --split test --runs gt_gt pred_vjepap_asd  # -> output/assist/eval/eval_test.md
python -m assist.evaluation.psr_eval --runs pred_vjepap_asd gt_gt         # -> output/assist/eval/psr_test_release.md

# 4. overlay videos (1920x1080 H.264; needs `pip install imageio-ffmpeg`) and the demo reel
python -m assist.tools.render_overlay --video 13_main_2_1 --run pred_vjepap_asd
python -m assist.tools.render_overlay --video 18_assy_2_5 --run gt_gt --snapshot 331   # PNG still only
python -m assist.tools.demo_reel --force                                  # -> output/assist/overlays/demo_reel.mp4

# 5. replay dashboard (reads the saved engine results; never runs a model)
python -m assist.run_engine --split test --actions pred --state asd       # the earlier live system, for comparison
python -m assist.run_engine --split test --actions pred_vjepap --state gt   # the "perfect state" run
python -m assist.run_engine --split test --actions pred_vjepap --state none # the "no state" run
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_asd pred_vjepa_asd pred_vjepap_asd pred_vjepap_gt pred_vjepap_none pred_gt pred_none gt_none
python -m assist.evaluation.psr_eval --runs pred_asd pred_vjepa_asd pred_vjepap_asd pred_vjepap_gt gt_asd pred_gt gt_gt
python -m assist.tools.transcode_videos                                   # once: H.264 copies, ~7 min
python -m assist.dashboard                                                # open http://localhost:8000

# unit tests (engine rules on synthetic event sequences)
python -m unittest discover -s tests -p "test_assist_*.py"
```

**Sound on the replay.** The station page has a Sound button, off by default, and a voice menu that remembers its choice.
An early warning plays two rising notes and one short line ("Wait. The front pin of the rear chassis is not next."); a mistake
plays three low, harsh falling notes and then a firm command ("Stop. Wrong order. Front pin of the rear chassis. First do:
short rear chassis."), because a factory is noisy. An uncertain install says "Check now" and not "Stop". "Cannot verify"
stays silent. The firm wording also applies to false alarms, which are about 6 per hour on the live system. Sound plays
when playback reaches an alert, not when dragging; alerts within 2 s after a mistake stay silent, so the screen can show
more than is said. The browser makes the tones and the speech (David by default, a little slower and lower), so there are
no audio files, and the engine results are unchanged (`spoken_text` in `assist/dashboard/data.py`, `static/sound.js`).
This is sound on a replay; it has not been tried with workers or in real factory noise.

The overlay videos can carry the same sounds (Windows only, for the speech). There the voice is also processed to carry
over noise: rumble removed, the 2-4 kHz range lifted, quiet syllables brought up. The picture is copied, unless the last
sentence runs past the video's end; then the last frame is held and that video is re-encoded.

```bash
python -m assist.tools.add_sound --video 13_main_2_1 --run pred_vjepap_asd --voice David   # or --voice Zira; -> overlays/<video>_<run>_sound.mp4
```

**The card overlay** (`assist/tools/overlay_cards.py`) is a second design for the same results: one large card that is green
for what is next, amber for a warning or check, and red for a mistake. The action pill, progress ring and timeline labels
take the same colour, and steps done correctly fade so the next one stands out. "Cannot verify" stays a mark on the
timeline. It draws the hands, what is in them and the state box; `--no-tracking` leaves them out. Oracle runs show no state
box, since they do not use the detector. `assist/tools/reel_cards.py` builds the demo reel in this design with sound, from
the same title cards and clips as `demo_reel.py`, whose older reel is kept.

```bash
python -m assist.tools.overlay_cards --video 13_main_2_1 --run pred_vjepap_asd   # -> overlays/cards/<video>_<run>.mp4
python -m assist.tools.add_sound --video 13_main_2_1 --run pred_vjepap_asd --design cards
python -m assist.tools.reel_cards --force                                        # -> overlays/cards/demo_reel.mp4
```

`assist/tools/six_cases.py` builds one clip per kind of alert, every one from a live run, chosen by matching alerts to the
labelled mistakes. Skipped step, wrong order, check the install and early warning are correct alerts on test videos. The
live system has no correct "installed wrong" on test (5 raised, all false alarms) or val (0 of 5 wrong installs), and no test
video has a real wrong part (1 raised, false), so those two clips come from training videos, and their title cards say so:

- Installed wrong: `11_assy_3_3`, the rear wheel, 3 s after the label. Wrong part: `22_assy_2_3`, the short chassis, correct
  but over 2 min after it went on.
- The state detector trained on these videos, so they show the alert working, not how well it finds new mistakes. They are
  not scored and do not change any result above.
- Their actions come from the TCN alone, from the folds that did not train on the video. V-JEPA 2 is not run on training
  videos.
- Across the 13 training videos with a real wrong install or wrong part, "installed wrong" fired on the right part at about
  the right time in 2 (`11_assy_3_3`, `27_main_1_3`). Elsewhere it often fired at about the right moment on another part, so
  the detector notices something is wrong more often than it knows which part.

```bash
# the released detector's weights are in assembly_state_detection_model_weights.zip on the dataset page
python -m assist.extract.train_asd --predict --conf 0.01 --weights <folder>/asd_best_IndustRealandSynthetic.pt --predict-out output/assist/asd_released/predictions --videos 11_assy_3_3 22_assy_2_3
python -m assist.extract.run_asd --source weights --split train --video 11_assy_3_3     # and 22_assy_2_3, for each of these
python -m assist.extract.run_ar --split train --video 11_assy_3_3
python -m assist.extract.run_hands --split train --video 11_assy_3_3 --compute
python -m assist.run_engine --video 11_assy_3_3 --actions pred --state asd --split train
python -m assist.tools.six_cases      # -> overlays/cards/six_cases.mp4 and six_cases_<n>_<alert>.mp4 for slides
python -m assist.tools.six_cases --v9 # -> six_cases_v9.mp4 and six_cases_v9_<n>_<alert>.mp4: our engine with v9's steps
```

The `--v9` clips cover skipped step, wrong order and early warning, all correct alerts on `13_main_2_1`. They need
`engine_pred_vjepap_asdv9.json` from `scripts.mistakes.compare_v9`. On test that run raised no "installed wrong" or "wrong
part", and its one "check the install" on `23_assy_1_2` names another part at another time, so those three clips have no v9 version.

`--cad` makes all six with a CAD picture of the recognised state in the corner of the video (`assist/tools/cad_state.py`): the
state the engine was given at that moment, not what is really on the table. Cases 1, 2 and 6 use the v9 run; cases 3, 4 and 5 use
the released detector. The dataset models correct states only, so an error reading keeps the last correct state with a red
"Wrong state seen" label. It needs `part_geometries.zip` (17 MB, [IndustReal's data page](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224)), unzipped so that
`output/assist/cad/part_geometries/state1.fbx` exists:

```powershell
python -m assist.tools.six_cases --cad   # -> overlays/cards/six_cases_cad.mp4 and six_cases_cad_<n>_<alert>.mp4
```

`--final` adds the action-end label (the 75.07 one, see "The action-end label" above) to the same six clips: a "Final label" chip in
the video's bottom-left corner and a thin "Final label" strip on the timeline. Each label appears 1.21 s after its action ends (the
pass measured on a free RTX 2080 Ti), or later while the previous pass still runs, and the chip names the action that just
ended, not the one in progress. The two training videos' labels come from models that also trained on those videos; their cards
say so. Training-video labels are written without scoring:

```powershell
python -m assist.extract.run_action_end --spans --videos 11_assy_3_3 22_assy_2_3 --plan spans_reel.json
python -m assist.extract.run_action_end --score --plan spans_reel.json   # GPU (needs both frame caches for these videos)
python -m assist.extract.run_action_end --write --videos 11_assy_3_3 22_assy_2_3
python -m assist.tools.six_cases --final  # -> overlays/cards/six_cases_final.mp4 and six_cases_final_<n>_<alert>.mp4
python -m assist.tools.six_cases --replay # replay made after every final label -> overlays/cards/six_cases_replay*.mp4
```

`--replay` is a replay made after every final label is ready: the action pill and the actions row (renamed "Final labels") show
each action's final label over its whole span, with no separate chip or strip. Live, a final label only exists 1.21 s or more
after its action ends, so this does not show what the system knew at each moment; every title card says the labels were filled in
after each action ended, and slides using the clips without their cards must say it too. Alerts, checklist and CAD picture stay
those of the live run.

Results on the 32 official test videos (every setting chosen on the 16 val videos, except the hidden-step
rule described in `assist/engine/engine.py`, which was found while inspecting test errors):

| Procedure step recognition (IndustReal protocol) | All: F1 | POS | Delay | With errors: F1 | POS | Delay |
|---|---|---|---|---|---|---|
| IndustReal B3 (WACV'24), as published | 0.883 | 0.797 | 22.4 s | 0.816 | 0.731 | 20.4 s |
| IndustReal, as reported in STORM-PSR (CVIU'25) | 0.891 | 0.797 | 21.0 s | | | |
| STORM-PSR (CVIU'25), as published | 0.901 | 0.812 | 15.5 s | | | |
| IndustReal ASD + official PSR (released predictions) | 0.878 | 0.815 | 16.9 s | 0.807 | 0.731 | 14.8 s |
| This engine, live system: TCN + V-JEPA 2 actions + the same ASD predictions | **0.892** | 0.795 | 15.9 s | 0.804 | 0.704 | **13.6 s** |
| This engine: annotated actions + annotated state (oracle) | 0.964 | 0.954 | 0.7 s | 0.923 | 0.902 | 0.7 s |

"With errors" is the paper's group, using the official `video_contains_errors` rule: an expected step never
completes, or a part is ever installed wrongly. That is 35 of the 84 videos, as the paper states, and 15 of the
32 test videos. Most of our F1 lead over the official PSR comes from the other 17 videos (0.969 against 0.941 on the
same predictions). On videos with errors our F1 is level with it, and our POS is lower. The live system does not beat
STORM-PSR: it is below it on F1 and POS, and about the same on delay.

**A correction.** Our copy of the official scoring (`assist/evaluation/psr_eval.py`) used to differ from IndustReal's
`determine_performance`, which raised our F1: we reported the live system at F1 0.902 and 15.6 s, and said that matched
STORM-PSR. It now matches the official code exactly: compared on 224 video and system combinations, with no differences.
The numbers here are from the corrected scoring. No pass or fail decision in this README changed.

**A teammate's system, v9** (branch `psr`, folder `PSR`), scored with the same official code. It uses per-part DINOv2
verifiers and actions in a state filter, uses past frames only, and has its own state input, so unlike the live system it
does not read pre-computed detector results. Its README notes it was the best of three versions tested on test. The last
row feeds v9's saved steps into our engine as the state, with the live actions and nothing tuned; it is one look at
test, and our hidden-step rule was found on test errors, so treat it as promising, not as a clean result.

| Step recognition, test | All: F1 | POS | Delay | With errors: F1 | POS | Delay | Mistakes found | False alarms / h |
|---|---|---|---|---|---|---|---|---|
| Live system | 0.892 | 0.795 | 15.9 s | 0.804 | 0.704 | 13.6 s | 8/32 | 6.0 |
| v9 alone | 0.893 | 0.804 | 7.6 s | 0.817 | 0.700 | 9.0 s | reports steps only | |
| This engine + v9's steps | 0.907 | 0.833 | 8.3 s | 0.837 | 0.740 | 9.6 s | 8/32 | 9.2 |

Faster, better step recognition did not find more mistakes here: order 4/4 against 3/4, skipped 2/10 against 3/10,
installed wrong 2/18 for both. v9's saved steps give the engine a reading every 0.5 s, so it always "sees" the assembly,
which may account for some of the extra skipped-step false alarms. The live system stays on the released detector.

```bash
git fetch origin psr
mkdir -p output/assist/v9 && git archive origin/psr PSR/results | tar -x -C output/assist/v9
python -m scripts.mistakes.compare_v9     # -> output/assist/v9/compare.json
```

**v9's visual checks as an install check: failed, test not scored.** After an install, if v9's two checks say the part is
absent for several good views in a row, the engine says "installed wrong". Rule fixed before any result, with the live
actions and the released detector: at least 2 more wrong installs, no more new false alarms than new finds, step F1 and POS
not lower. v9's checks were trained on val, so val could only choose a setting, and failing there means test is not scored.
On val (5 wrong installs, baseline 0 found and 5 false alarms) it found 5 in 17 of 18 settings, but always with too many
false alarms: the closest were 5 finds with 6 new false alarms (11 in all) and 4 finds with 5 new (10 in all). The finds
overstate it: at the first of those settings, 9 of the 11 check alerts fired while the labels still had the part off, so
action recognition had called the install finished too early and the part really did look absent; only 2 fired on a part
labelled wrong. Reading the videos instead of the original frames changes little: v9's own steps with annotated actions
score F1 0.921, POS 0.835 on test against its saved 0.918, 0.833.

```bash
# his code and our label files; PSR_CACHE holds his two cache files. Frames are read from our videos
git archive origin/psr PSR | tar -x -C output/assist/v9
export V9_DIR=output/assist/v9/PSR INDUSTREAL_DATA=output/assist/v9/dataset PSR_CACHE=output/assist/v9_checks
python -m scripts.mistakes.v9_checks --link
python -m scripts.mistakes.v9_checks --crops --shard 0 --shards 2      # and --shard 1; about 26 min on a laptop RTX 4050
VBACK=base VRES=336 python -m scripts.mistakes.v9_checks --verify val weights/verif_base_r336.pt                         # and test
VBACK=base VRES=336 VEPOCHS=6 VUNFREEZE=6 python -m scripts.mistakes.v9_checks --verify val weights/verif_base_r336_e6u6.pt   # and test
python -m scripts.mistakes.v9_checks --reproduce test                  # his step scores from our videos
python -m scripts.mistakes.v9_checks --select                          # -> output/assist/v9_checks/select_val.json
```

**The action-end label and a stricter confidence for engine + v9: failed on test.** Two alert-only ideas, both off by
default (`Engine.on_final_label`, `cfg["final_label"]`): (1) the action-end label (75.07) reaches the engine when the GPU
has finished it (action end + a 1.34 s queue per action). An install it names is checked against v9's state (no expected
state within 30 s: possible incorrect install), and a later-subgoal action with an earlier step not done is a skipped
step after a grace period. (2) A higher `low_conf` (0.6 instead of 0.45): weaker actions only give "cannot verify". Rule
fixed before any result: at least 2 more finds with no more new false alarms than new finds, or no fewer finds with at
least a quarter fewer false alarms. Steps never move, so step F1 0.907, POS 0.833 and delay 8.3 s stay the same.

Val (13 mistakes, optimistic: v9 was trained on val) already found 10 of 13. Idea 1 found nothing new in any of its 30
settings and only added false alarms. Idea 2 passed (10 found, false alarms 8 to 6), so it was scored on test once:

| Engine + v9, test (32 mistakes) | Found | Skipped | Order | Installed wrong | Precision | False alarms / h | Early warnings / h |
|---|---|---|---|---|---|---|---|
| Live settings (`low_conf` 0.45) | 8 | 2/10 | 4/4 | 2/18 | 26% (8/31) | 9.2 | 21.6 |
| `low_conf` 0.6, chosen on val | 7 | 2/10 | 4/4 | 1/18 | 32% (7/22) | 6.0 | 12.4 |

It cut false alarms by a third but lost one install, so it fails the rule and the live settings stay. Why the action
label cannot help with installs: of val's 3 missed wrong installs, v9 reported 2 as normal finished steps, so the expected
state did appear and the check passed. A wrong install needs a state input that can tell wrong from right.

```bash
python -m assist.extract.run_action_end --spans --videos 05_assy_0_1 05_assy_2_2 05_main_0_1 14_assy_0_1 14_main_0_1 14_main_2_2 14_main_2_3 20_assy_0_1 20_assy_3_6 20_main_0_1 24_assy_0_1 24_assy_2_4 24_main_0_1 26_assy_0_1 26_assy_1_5 26_main_0_1 --plan spans_val.json
python -m assist.extract.run_action_end --score --plan spans_val.json --part 0/2      # GPU; and --part 1/2 on the second GPU
python -m assist.extract.run_action_end --write --videos 05_assy_0_1 05_assy_2_2 05_main_0_1 14_assy_0_1 14_main_0_1 14_main_2_2 14_main_2_3 20_assy_0_1 20_assy_3_6 20_main_0_1 24_assy_0_1 24_assy_2_4 24_main_0_1 26_assy_0_1 26_assy_1_5 26_main_0_1
python -m scripts.mistakes.end_label_check --states     # v9's saved val steps -> <video>/asd_v9.jsonl
python -m scripts.mistakes.end_label_check --select     # -> output/assist/end_label_check/select_val.json
python -m scripts.mistakes.end_label_check --test       # once -> output/assist/end_label_check/test.json
```

**A state model trained on correct states only (Schoonbeek et al., RA-L 2024): failed on val, test not scored.**
`scripts/mistakes/state_match.py` trains their ResNet34 (supervised contrastive loss with their intermediate-state change,
real + synthetic images, their code and settings, 100 epochs, seed 123) on their train images, all from our 36 train
recordings. After an install, if several views in a row are too unlike the reference image of the state the procedure
expects, the engine says "installed wrong"; steps never move. Rule fixed on 2026-11-05, before any result, with the live
actions and the released detector on val: at least 2 more of the 5 wrong installs, no more new false alarms than new
finds, step F1 and POS not lower.

The model itself is about as good as the paper's on their error images (balanced AP, chance 50):

| Error type | Train recordings | Val recordings | Paper, read from its chart |
|---|---|---|---|
| Part missing | 99.7 | 99.1 | ~100 |
| Wrong place | 99.3 | (none) | ~70 |
| Turned the wrong way | 94.9 | (none) | ~95 |
| Part-level | 60.9 | 86.3 | ~73 |

The train column is optimistic: those recordings were used for training, though their error frames were not.

On val no setting raised a single alert (baseline 0 of 5 wrong installs, 5 false alarms; all 12 settings the same). The 5
wrong installs show why: 3 (front bracket and its screw) come at a state the dataset defines no reference for, so they
cannot be checked; the wrongly mounted front wheel looks like the right state to the model; only the rear wheel reads as
wrong, and the detector's confidence on those views is below the engine's 0.5, so they are not used. At most 1 of 5 was
reachable. Training the model is not the problem; a wrong install has to be compared with a reference of exactly the
state that should be there, and the dataset has references for 22 states only.

The images are read straight from their tar files (`scripts/mistakes/tar_images.py`): the shared disk creates small
files at about 5 per second, so unpacking 300,000 images would take more than a day. Their per-epoch val check runs on
the CPU (`--metric-on-cpu`), as it does not fit next to training on an 11 GB GPU; the numbers are the same.

```bash
# GPU server; their code in $CODE (github.com/TimSchoonbeek/AssemblyStateRecognition), data (doi 10.4121/611adbc7-7935-43a6-8c3f-b2260a508e73) in $B
python -m scripts.mistakes.tar_images --index $B           # synth.tar val.tar train.tar -> tar_index.pkl, label files
python -m scripts.mistakes.tar_images --metric-on-cpu --run $B $CODE/train.py resnet34_supcon_isil --data_path $B/data --run_path $B/runs --seed 123 --loss supcon --n_real 8 --n_synth 8 --n_bg 64 --exclude_bg --img_h 224 --img_w 224 --model resnet34.a1_in1k --workers 8
python -m scripts.mistakes.state_match --embed             # -> output/assist/state_match/sims_{val,test}.npz
python -m scripts.mistakes.state_match --errors            # with tar_images installed -> errors_check.json
python -m scripts.mistakes.state_match --select            # CPU -> select_val.json (failed: test is not scored)
```

| Mistake detection (32 real mistakes) | Precision | Recall | False alarms / h | Warned before |
|---|---|---|---|---|
| Baseline: official IndustReal PSR on the same ASD + the same mistake rules | 21% (4/19) | 12% (4/32) | 6.0 | 0/32 |
| This engine, live system: TCN + V-JEPA 2 actions + ASD | **35% (8/23)** | **25% (8/32)** | 6.0 | 10/32 |
| Oracle (annotated actions + state), upper bound | 94% (32/34) | 100% (32/32) | 0.8 | 10/32 |

No published method reports live mistake detection on IndustReal, so the baseline is built from the paper's
own pipeline (`assist/evaluation/baseline.py`): the official PSR's completed steps are replayed into a state
sequence and checked with exactly the rules that define the ground truth. It cannot report an incorrectly
installed part (the official PSR only completes correct steps), and gives no early warnings. On order and
skipped mistakes alone: baseline 4/14 found with 15 false alarms, this engine 6/14 with 4.

```powershell
python -m assist.evaluation.baseline                                     # -> engine_baseline_psr.json per video
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_vjepap_asd
```

The gap to the oracle is perception: 18 of the 32 mistakes are incorrectly mounted parts, and the released
ASD detector almost never reports them.

**Our own state detector** (tried, not used): the released ASD predictions cover test only and we took its weights as
unreleased (they are published; see "A fairer val baseline" below), so `assist/extract/train_asd.py` trains a YOLOv8-M on the 36 real train videos (no synthetic data), with frames of
wrongly mounted parts labelled `error_state`. Val mAP50 0.75, but `error_state` only 0.12. It has no
"background" output, so it guesses on blocked views; detections below 0.85 are dropped (`asd_min_conf`, chosen
on val). It stays well below the released detector, which was also trained on synthetic images:

| Test, official PSR | F1 | POS | Delay | Mistake recall (install), with our engine | False alarms / h |
|---|---|---|---|---|---|
| Released ASD | 0.878 | 0.815 | 16.9 s | 11% (2/18) | 6.0 |
| Our ASD (real train videos only) | 0.784 | 0.742 | 19.5 s | 11% (2/18) | 22.0 |

```bash
# on the GPU server (~1.7 h training, 25 epochs on one RTX 2080 Ti, ~10 min prediction), then copy predictions/ back
python -m assist.extract.train_asd --build --train --predict
python -m assist.extract.run_asd --source ours --split val     # and --split test
python -m assist.run_engine --split val --actions pred --state asdours
python -m assist.evaluation.psr_eval --asd ours --tune         # fusion settings on val
python -m assist.evaluation.psr_eval --asd ours --runs pred_asdours gt_gt
```

**Where the gap is.** The same engine and rules on test, from no state input at all to a perfect one (mistakes
found / annotated; step recognition with the IndustReal protocol):

| Engine inputs | Skipped | Order | Installed wrong | All | False alarms / h | Step F1 | POS |
|---|---|---|---|---|---|---|---|
| Our live actions only, no state (`pred_vjepap_none`) | 2/10 | 4/4 | 0/18 | 6/32 | 33.2 | | |
| Annotated actions only, no state (`gt_none`) | 1/10 | 4/4 | 0/18 | 5/32 | 20.8 | | |
| Annotated actions + released ASD (`gt_asd`) | 3/10 | 3/4 | 0/18 | 6/32 | 4.0 | 0.892 | 0.795 |
| Earlier live system: TCN actions + released ASD (`pred_asd`) | 3/10 | 3/4 | 2/18 | 8/32 | 6.0 | 0.892 | 0.795 |
| Our live actions + released ASD (`pred_vjepap_asd`, the live system) | 3/10 | 3/4 | 2/18 | 8/32 | 6.0 | 0.892 | 0.795 |
| Our live actions + annotated state (`pred_vjepap_gt`) | 10/10 | 4/4 | 18/18 | 32/32 | 2.8 | 0.964 | 0.954 |
| Oracle: annotated actions + annotated state (`gt_gt`) | 10/10 | 4/4 | 18/18 | 32/32 | 0.8 | 0.964 | 0.954 |

Perfect actions change nothing in step recognition and find fewer mistakes than ours; a perfect state takes
both to the oracle. The engine and the action model are enough; the state detector is the limit. Without any
state (what a dataset without state labels gets) order mistakes are still found, at many more false alarms, and
wrongly mounted parts cannot be seen at all. IndustReal is the only dataset here with assembly-state labels;
step-completion labels (IndustReal, MECCANO) give the state over time and are the route to state recognition
elsewhere (not done).

```powershell
python -m assist.run_engine --split test --actions pred_vjepap --state none   # and --state gt; --actions gt
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_asd pred_vjepa_asd pred_vjepap_asd pred_vjepap_gt pred_vjepap_none pred_gt pred_none gt_none
python -m assist.evaluation.psr_eval --runs pred_asd pred_vjepa_asd pred_vjepap_asd pred_vjepap_gt gt_asd pred_gt gt_gt
```

**Tried, no gain: anomaly detection for wrongly mounted parts** (`scripts/mistakes/state_anomaly.py`, a 1-day
spike). There are only 16 wrongly mounted parts in train and 5 in val, so instead of learning them, each
detected assembly box is compared with DINOv2-small embeddings of the annotated correct states in the train
videos (no error examples); a crop far from every reference of its reported state, several detections in a row
with the hands not covering it, becomes `error_state`. Stop rule fixed beforehand: on val, find at least 2 of
the 5 with no more false alarms than without it. It finds 2-3 of 5 but at 28-60 false alarms / h against 25.6,
so it stopped and test was not scored. Half of all ordinary detections already score as "unusual" (median 1.5x
the typical distance between train videos): viewpoint, hands and box placement change the crop more than a
wrongly mounted part does.

```powershell
python -m scripts.mistakes.state_anomaly --embed     # GPU server: DINOv2 crops of train references, val + test boxes
python -m scripts.mistakes.state_anomaly --select    # CPU: val grid + stop rule -> output/assist/anomaly/select_val.json
```

**Tried, worse on test: reading each part's state** (`scripts/mistakes/part_state.py`). A fine-tuned backbone
looks at the assembly crop and calls each of the 11 parts absent, mounted or wrong, trained on the 12,483
annotated boxes of the 36 train videos (1,072 show a wrong part). DINOv2-small and V-JEPA 2 were trained, and val
chose among each model and their average, with a threshold for calling a part wrong. Stop rule fixed beforehand:
on val, more wrongly mounted parts found than our YOLO detector, with false alarms / h and step F1 no worse. The
average at threshold 0.5 passed, so test was scored once:

| | Compared with | Mistakes found | Installed wrong | False alarms / h | Step F1 | POS | Delay |
|---|---|---|---|---|---|---|---|
| Val, our YOLO detector | | 1/13 | 0/5 | 25.6 | 0.476 | 0.313 | 15.9 s |
| Val, per-part average | our YOLO | 3/13 | 1/5 | 6.6 | 0.621 | 0.479 | 14.5 s |
| Test, released ASD (live system) | | 8/32 | 2/18 | 6.0 | 0.892 | 0.795 | 15.9 s |
| Test, per-part average | released ASD | 5/32 | 0/18 | 5.2 | 0.749 | 0.688 | 10.8 s |

The released detector stays. The two comparisons differ because the released detector has no val predictions: val
could only show a gain over our weaker YOLO, not over the released one (also trained on synthetic images). On
test the per-part model reads the released detector's boxes. Finding wrongly mounted parts is still the unsolved
part, with 1 of 5 on val and none of 18 on test.

```bash
# on the GPU server (V-JEPA 2: about 1.5 h on one RTX 2080 Ti), then copy probs_<model>.npz back
python -m scripts.mistakes.part_state --train --model dino
python -m scripts.mistakes.part_state --train --model vjepa --batch 4 --accum 4
python -m scripts.mistakes.part_state --select     # CPU: val grid + stop rule -> output/assist/part_state/select_val.json
python -m scripts.mistakes.part_state --test       # CPU: only if val passed -> output/assist/part_state/test.json
```

**Tried, failed on val: the per-part model as an extra check** (`scripts/mistakes/part_check.py`). The state
detector keeps completing steps, and the per-part average only adds "installed wrong" alerts: just after a part's
install, on looks with a good detection and clear hands, after several high readings in a row. Rule and 12 settings
fixed beforehand (threshold 0.5 / 0.7 / 0.9, window 10 / 30 s, 3 / 8 looks): on val, with the live actions and our
YOLO, at least 2 of the 5 wrong installs newly found, no more new false alarms than new finds, step F1 and POS not
lower. No setting raised a single alert, so the rule failed and test was not scored.

| Val, live actions + our YOLO | Part alerts | Installed wrong | False alarms / h | Step F1 | POS |
|---|---|---|---|---|---|
| Without the check | 0 | 0/5 | 25.6 | 0.476 | 0.313 |
| The 12 fixed settings (all the same) | 0 | 0/5 | 25.6 | 0.476 | 0.313 |
| After the rule failed: threshold 0.1, 10 s, 3 looks | 20 | 1/5 | 42.6 | 0.476 | 0.313 |

The last row was added afterwards to see whether the settings were just too strict, and could not be chosen. The
lower thresholds trade one find for many false alarms. The trigger is not the problem: the model's readings are. On
looks where a part really is wrong, its median P(wrong) is 0.08, against 0.05 on all other parts, and only 1.6 % of
those looks pass 0.5. Step status never changes with this check (the script stops if it does). The engine hook
(`on_part_wrong`) does nothing unless `part_check` is set, and the live system does not set it.

```bash
python -m scripts.mistakes.part_check   # CPU, val only -> output/assist/part_state/part_check_val.json
```

**A fairer val baseline: IndustReal's own detector.** The dataset page also publishes the detector's weights
(`assembly_state_detection_model_weights.zip`), which we had taken as unreleased. So val can now compare against the
real baseline. Our run reads the compressed videos and only approximates the published predictions, so a confidence
cutoff (0.12) and shift (+0.08) were set on 5 test videos, with no labels, to match them: 99 % of the detections, 93 %
the same frames, 98 % the same class, 88 % of the engine's step completions within 10 s. On val with the live actions:

| Val, live actions | Mistakes found | Installed wrong | False alarms / h | Step F1 | POS | Delay |
|---|---|---|---|---|---|---|
| Released detector (our run) | 1/13 | 0/5 | 4.7 | 0.790 | 0.621 | 18.1 s |
| Per-part average | 3/13 | 1/5 | 6.6 | 0.621 | 0.479 | 14.5 s |
| Our YOLO | 1/13 | 0/5 | 25.6 | 0.476 | 0.313 | 15.9 s |

The per-part model's test loss shows up here too, so this comparison would have stopped it before test. New state
models should be judged against the first row.

**IndustReal's synthetic images hold no wrong installs.** `assembly_state_detection_synthetic_data.zip.001/.002`
(16.1 GB) has 104,750 images. Its label file covers 100,000 of them, all with correct states; there is no `error_state`
class. The 22 models in `part_geometries.zip` and `Overview of states.pdf` also show correct states only. These were
read from the end of the zip, without downloading the images. So synthetic data explains the released detector's step
scores, not wrong-install detection (0/5 on val), and wrong installs would have to be modelled by hand.

```bash
# on the GPU server: the released weights on val (about 10 min on one RTX 2080 Ti), then copy the CSVs back
python -m assist.extract.train_asd --predict --conf 0.01 --weights <folder>/asd_best_IndustRealandSynthetic.pt --predict-out output/assist/asd_released/predictions --videos <the 16 val videos>
python -m assist.extract.run_asd --source weights --split val
python -m scripts.mistakes.val_baseline   # CPU -> output/assist/part_state/val_baseline.json
```

**Tried, failed its first check: training on rendered wrong installs** (`scripts/mistakes/render_wrong.py`,
`render_train.py`). Eleven kinds of error on wheels and pins were built from the 3D models, after comparing sharp frames
of the real wrong installs in the train videos with the correct model: a pin the other way round, the rear spacer,
a wheel, a hub or the front plate missing, the axle turned round or in another hole. Each of the 10,000 wrong images has
a correct twin with the same camera, light, background and blur, so the model cannot just learn "rendered" against
"real". DINOv2-small was fine-tuned on these only, for a fixed 6 epochs. Real wrong installs were never used in training,
so they could judge it. Rule fixed beforehand: AUC of at least 0.90 on real annotated crops, and at least half of the 12
real cases flagged (3 readings in a row above a threshold that lets through at most 1 % of correct train crops).

| Real crops, never trained on | AUC | Median reading, wrong / correct | Cases flagged |
|---|---|---|---|
| Front wheels | 0.745 | 0.07 / 0.01 | 1/4 |
| Rear wheels | 0.767 | 0.09 / 0.01 | 2/5 |
| Front rear-chassis pin | 0.981 | 0.99 / 0.00 | 1/1 |
| Rear rear-chassis pin | 0.720 | 0.28 / 0.02 | 1/2 |
| **All (rule: 0.90 and 6/12)** | **0.751** | | **5/12 (val 0/2)** |

It failed both bars, so it stops here and test was not scored. The per-part model's median on wrong parts was 0.08, so
this is more signal than before on some parts, but the wins are all on train videos used to design the errors, and the
clean val cases were missed. The 0.981 comes from one recording. Two of the misses (`11_assy_3_3`, `16_main_3_3`) were
ones whose error could not be seen by eye, so they were never modelled.

```bash
# on the GPU server. pyrender needs numpy's old name, handled in the script; packages kept apart from the main venv
python -m pip install --no-deps --target <folder> pyrender trimesh PyOpenGL pyglet==1.5.29 networkx freetype-py imageio
PYTHONPATH=.:<folder> python -m scripts.mistakes.render_wrong --pairs 10000     # about 25 min, 0.25 GB
python -m scripts.mistakes.render_train --train --gate                          # about 15 min -> output/assist/render_wrong/gate.json
PYTHONPATH=.:<folder> python -m scripts.mistakes.render_wrong --round 2 --pairs 10000
python -m scripts.mistakes.render_train --round 2 --train --gate                # -> output/assist/render_wrong/round2/gate.json
```

**Round 2, worse.** Same rule, new renders (real hands over the car, wider views, loose parts, softer images), and real
correct crops added to training from 13 recordings with no wrong part, counted only when read above 0.2. Thresholds and
AUC leave those 13 out. It was designed after seeing round 1's scores, so a pass would only have earned one run on test.

| Real crops, never trained on | Round 1 AUC | Round 2 AUC | Round 2 median reading, wrong / correct |
|---|---|---|---|
| Front wheels | 0.745 | 0.547 | 0.013 / 0.013 |
| Rear wheels | 0.767 | 0.545 | 0.007 / 0.006 |
| Front rear-chassis pin | 0.981 | 0.519 | 0.028 / 0.028 |
| Rear rear-chassis pin | 0.720 | 0.698 | 0.018 / 0.013 |
| All | 0.751 | 0.398 | |
| Cases flagged | 5/12 | 3/12 (val 1/2) | |

The model stopped reacting to wrong parts on real images: it reads them the same as correct ones. Most likely it learned
that real images are never wrong, the risk the 0.2 cut-off was meant to reduce, but both the renders and the training
changed in this round, so this run cannot separate the two. The pooled 0.398 is below the per-part values because the
parts' readings sit on different scales. Stopped; test was not scored.

**Looked at, not used: depth and the side cameras.** IndustReal also publishes depth and two side cameras. Frames from 8
train recordings were fetched file by file and compared by eye (no model). Depth is stored as a 320 x 288 colour-coded
JPEG; the car is a blurred shape with holes in it, and two nuts side by side that are obvious in colour (`27_main_3_1`)
cannot be seen. The side cameras are 640 x 480, grey and wide-angle, with the car small in the frame. Neither adds detail
to the colour view, so nothing was trained on them.

**Online V-JEPA 2 labels (the 77.2 models without the future).** The 77.2 result labels segments whose true
start and end are given and decodes each video as a whole, so it cannot run live. Two online versions:

- *Online at action end* (true boundaries, label fixed the moment each action ends, order prior over past
  actions only, `scripts/recognition/online_vjepa_end.py`): **74.5 test** (val 72.5), against 77.2 offline.
- *Live label* (no boundaries given, `assist/extract/online_vjepa.py`): the live model's events give each
  action's start; every 0.5 s the ensemble scores the clip [start, now], and each frame's live label is pulled
  towards the latest result (weight `w`), shown 0.5 s later (compute budget). One update must fit in 0.5 s on
  the RTX 2080 Ti: all five members take 523 ms, so val chooses among faster groups.

Every setting is chosen on the 16 val videos. `w` is picked by what the assistant does with the labels: the
engine on val (our ASD, the only state source with val predictions), most mistakes found, then most warned
before, then fewest early warnings per hour. Every weight finds the same mistakes on val (mistake alerts come
from the state detector), so the early warnings decide: VideoMAE + V-JEPA 2 hand k12 (149 ms), `w` 0.6, order
prior 1.5 (val: early warnings 44.5 -> 36.9 / h, same 5 of 13 warned). Test, scored once:

| Test (32 videos) | Frames right | Label right at each action's end | Mistakes: precision / recall | False alarms / h | Warned before | Early warnings / h | PSR F1 |
|---|---|---|---|---|---|---|---|
| TCN alone (`pred_asd`) | **48.4%** | 39.8% | 35% (8/23) / 25% | 6.0 | 10/32 | **19.2** | 0.892 |
| Online V-JEPA 2 labels (`pred_vjepa_asd`) | 44.9% | **46.2%** | 38% (8/21) / 25% | **5.2** | 10/32 | 22.8 | 0.892 |

The ensemble is better once most of an action has been seen and worse early on (its window is then a few
frames, while it was trained on whole actions), so the label at the action's end gains 6.4 points while
per-frame accuracy loses 3.5. The assistant barely changes: the same 8 mistakes found with 2 fewer false
alarms, 3.6 more early warnings per hour (fewer on val), and identical step recognition, because step
completion and most mistakes come from the state detector. Better action labels alone do not move the
assistant; the gap to the oracle is the state detector (see above).

```powershell
python -m scripts.recognition.online_vjepa_end                    # online-at-end number (server or with torch)
python -m assist.extract.run_ar_vjepa --plan --split val          # and --split test (CPU, from the live model)
python -m assist.extract.run_ar_vjepa --benchmark                 # GPU server: ms per update and member
python -m assist.extract.run_ar_vjepa --windows --split val --part 0/2   # GPU server, one part per GPU (also 1/2,
                                                                         # and --split test); copy the npz back
python -m assist.extract.run_ar_vjepa --select                    # CPU: per-frame grid on val
python -m assist.extract.run_ar_vjepa --select-mistakes           # CPU: final weight by the val engine run
python -m assist.extract.run_ar_vjepa --write --split test        # test scored once -> ar_vjepa.jsonl per video
python -m assist.run_engine --split test --actions pred_vjepa --state asd
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_asd pred_vjepa_asd
python -m assist.evaluation.psr_eval --runs pred_asd pred_vjepa_asd gt_gt
python -m assist.tools.render_overlay --video 13_main_2_1 --run pred_vjepa_asd
```

The `--windows` step needs `release_vjepa/` and the frame caches of `vjepa2_finetune --cache` and
`finetune_videomae_ar` (val and test videos) on the server; val took 26 min and test 59 min on two RTX 2080 Ti.

**Action-beginning V-JEPA 2 (`--variant prefix`).** The table above shows the ensemble losing early in an action,
so V-JEPA 2 (hand crop, last 12 layers) was retrained on clips cut at a random 10-100% of each action. Rule fixed
beforehand: on val, beat the live model on frames right and on the label at action end, find at least as many
mistakes, raise no more false alarms; then test is scored once and reported either way. Val's pick by mistakes,
`w` 0.8, failed (39.7% frames against 44.5%), so the setting comes from those that pass: `w` 0.1, V-JEPA 2 alone,
125 ms per update.

| | Frames right | Label right at each action's end | Mistakes found | False alarms / h | Warned before | Early warnings / h | PSR F1 |
|---|---|---|---|---|---|---|---|
| Val, TCN alone | 44.5% | 37.8% | 1/13 | 25.6 | 5/13 | 44.5 | |
| Val, `w` 0.1 | 44.9% | 40.2% | 1/13 | 25.6 | 5/13 | 40.7 | |
| Test, TCN alone (`pred_asd`) | 48.4% | 39.8% | 8/32 | 6.0 | 10/32 | 19.2 | 0.892 |
| Test, `w` 0.1 (`pred_vjepap_asd`) | **48.8%** | **43.2%** | 8/32 | 6.0 | 10/32 | 19.2 | 0.892 |

It passes and beats the TCN on both label measures, but it improves the causal TCN and does not replace it: `w`
0.1 keeps each label close to the TCN's, and the TCN still gives every action's start. V-JEPA 2 at full weight
stays below the TCN per frame on val (37.2%). The assistant's mistake, warning and step numbers do not change. It
is now the live system, for its better labels.

**Speed, measured together on one RTX 2080 Ti** (`18_assy_2_5`, 4,318 frames, 432 s of video). The TCN pipeline
recomputed every feature from the video while V-JEPA 2 ran one update every 0.5 s on the same GPU:

| | Result | Needed to keep up |
|---|---|---|
| TCN pipeline | 14.6 fps (68.6 ms / frame), 296 s for the 432 s video | at least 10 fps |
| V-JEPA 2 update, median / 95th percentile | 164 ms / 227 ms (148 / 160 ms with the GPU otherwise idle) | under 500 ms |
| V-JEPA 2 updates over 500 ms | 7 of 586 (1.2%), slowest 2.3 s | none would be ideal |

Both keep up, with about 1 update in 80 running late. The TCN pipeline's 14.6 fps against 13.7 fps in the earlier
run is run-to-run variation, not a speed-up. This is not yet a true live stream: the pipeline runs one model at a
time over the whole video, V-JEPA 2 reads frames from a cache, and neither the state detector (only its released
predictions exist) nor MediaPipe hands are included.

```powershell
python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4 --prefix 0.1  # GPU server, ~2 h
cp output/official_split_eval/vjepa2/vjft_hand_k12_p10.pt release_vjepa_prefix/
python -m assist.extract.run_ar_vjepa --variant prefix --windows --split val    # and --split test; GPU, ~13 + ~29 min
python -m assist.extract.run_ar_vjepa --variant prefix --select                 # CPU, as are the steps below
python -m assist.extract.run_ar_vjepa --variant prefix --select-mistakes
python -m assist.extract.run_ar_vjepa --variant prefix --apply-rule             # the rule above
python -m assist.extract.run_ar_vjepa --variant prefix --write --split test     # test scored once
python -m assist.run_engine --split test --actions pred_vjepap --state asd
python -m assist.evaluation.eval --split test --runs gt_gt baseline_psr pred_asd pred_vjepa_asd pred_vjepap_asd pred_gt pred_none gt_none
python -m assist.evaluation.psr_eval --runs pred_asd pred_vjepa_asd pred_vjepap_asd pred_vjepap_gt gt_asd pred_gt gt_gt
# the speed test, on one GPU: remove that video's output/assist/_features files first, or they are reused
python -m assist.extract.run_ar_vjepa --variant prefix --benchmark-live   # prints ms per update until stopped
python -m assist.extract.run_ar --video 18_assy_2_5 --compute --force     # in a second shell -> ar_timing.json
```

**Second dataset: Assembly101-O** (PREGO protocol, annotations only). The engine's idea in a separate, smaller
rule module (`assist/assembly101/mistakes.py`, `assist/configs/assembly101.yaml`; the benchmark gives step
sequences, so there is no perception, state or event layer): order rules mined from the 138 correct training procedures, settings
chosen on synthetic mistakes in held-out training procedures (no test data used), scored on the 182 test
sequences in all three metrics the papers use:

```powershell
python -m assist.assembly101.mistakes        # -> output/assist/assembly101/assembly101_o.md
```

| Mistake detection | Steps | F1 (PREGO) | F1_b (TI-PREGO) | Avg F1 (TGML) |
|---|---|---|---|---|
| PREGO (CVPR'24) | oracle | 46.3 | 69.5 | |
| TI-PREGO | oracle | | 74.7 | |
| TGML-DO (NeurIPS'24) | oracle | | 83.6 | 75.9 |
| Ours | oracle | 46.9 | 62.3 | 57.6 |
| PREGO / TI-PREGO / TGML-DO | MiniROAD | 41.8 / – / – | – / 67.4 / 44.1 | – / – / 53.5 |
| Ours | MiniROAD | 32.9 | 46.7 | 43.4 |

Simple mined rules match the original PREGO but stay below the learned task graphs of TGML-DO and the
LLM-based TI-PREGO; 39 of the 182 mistakes are "wrong position" attaches, which look like correct steps at the
benchmark's verb + part granularity. Requires `../datasets/assembly101_mistakes` (Assembly101 mistake labels,
CC BY-NC 4.0) and `../datasets/assembly101_prego` (PREGO's released split and sequences). The causal AR pipeline with V-JEPA 2 updates runs at 14.6 fps (68.6 ms per frame) on one
RTX 2080 Ti, faster than the 10 fps video.

**Overlay on an Assembly101 video.** Assembly101-O is egocentric (PREGO uses one head-mounted view). Its files
give the step order only; the step times come from the annotations (30 fps frame numbers; the videos run at 60
fps), so the overlay shows the oracle setting: annotated steps, each judged online when it ends. Test sequences
end at their first mistake. The example (selected: a wrong-order mistake caught with no false alarm) needs the
four ~0.1 GB egocentric views of one recording from the official Hugging Face copy (sign in and accept the
CC BY-NC 4.0 terms first, then `hf auth login`):

```powershell
hf download cvml-nus/assembly101 --repo-type dataset --local-dir ..\datasets\assembly101_videos --include "recordings/nusar-2021_action_both_9063-c14a_9063_user_id_2021-02-17_101116/HMC_*"
python -m assist.assembly101.render_overlay --recording nusar-2021_action_both_9063-c14a_9063_user_id_2021-02-17_101116 --camera HMC_21179183 --start 220 --stop 292
# -> output/assist/assembly101/overlays/9063-c14a_HMC_21179183.mp4
```

## Second dataset: MECCANO action recognition

The IndustReal action-end recipe (75.07 % on IndustReal), run on MECCANO with only the dataset-specific changes:
20 egocentric videos of a toy-motorbike assembly, 61 actions, official split 11 train / 2 val / 7 test. Same protocol as
the published numbers: annotated segment boundaries, top-1 over all 2,823 test segments, RGB only, and each segment is
labelled from its own frames and earlier ones only (past only). The one free choice (which MECCANO VideoMAE run) was made
on the 2 val videos; test was scored once. Run by the teammate on the lab server (2026-11-10); their notes are in
`meccano_action_end/README.md`, results in `output/meccano_action_end/`.

| MECCANO test top-1 | Top-1 |
|---|---|
| SlowFast (MECCANO paper, CVIU 2023) | 45.16 |
| Swin3D-B (challenge winner) | 50.80 |
| **Ours, IndustReal action-end recipe, past only** (val 66.12) | **62.45** |

Four video models each classify the action's frames (16 frames, at 3 temporal samplings). Their mean counts 0.95 against
0.05 of a past-only TCN, then the action-order prior (strength 0.2, smoothing 0.01, forward only, counts from the 11 train
videos) picks the label:

| Model | Test top-1 alone | Checkpoint |
|---|---|---|
| V-JEPA 2, hand crop, 12 layers, seed 1 | 61.60 | `output/meccano_action_end/vjepa/vjft_hand_k12_s1.pt` |
| V-JEPA 2, hand crop, 12 layers, seed 0 | 60.86 | `output/meccano_action_end/vjepa/vjft_hand_k12.pt` |
| V-JEPA 2, work-area crop, 12 layers | 57.88 | `output/meccano_action_end/vjepa/vjft_area_k12.pt` |
| VideoMAE fine-tuned on MECCANO (`off_s3`, better than `off_s2` on val) | 51.82 | not saved; only its predictions (`output/meccano_action_end/vmae_probs.npz`) |
| Past-only TCN (DINOv2 + past-only VideoMAE-SSv2 features, 3 seeds, no video-progress input) | 32.66 | predictions in `output/meccano_action_end/tcn/official_causal/` |

Mean of the four: 62.98; full recipe: 62.45 (`output/meccano_action_end/ablation.json`, diagnostic only, nothing chosen
from it). Each model's predictions per segment are saved (`vjepa/clip_probs_*.npz`, `vmae_probs.npz`), so the number is
recomputed on a CPU.

Data: STARE's `data/MECCANO` (annotations, `MECCANO_RGB_Videos`), or `$MECCANO_ROOT`. On the server
`output/meccano_action_end/vjepa_frames_960` is a link to `data/MECCANO/frames_960` (only V-JEPA 2 training reads it; the
Windows copy has no link). From `system/assistant`:

```bash
python -m meccano_action_end.action_end --features      # GPU: DINOv2 + past-only VideoMAE-SSv2 per-frame features
python -m meccano_action_end.action_end --tcn           # GPU: past-only TCN
python -m meccano_action_end.vjepa2 --crop hand --layers 12 --batch 4 --accum 4 --seed 1   # GPU, 2-2.5 h per run
python -m meccano_action_end.action_end --val           # CPU -> output/meccano_action_end/action_end_val.json
python -m meccano_action_end.action_end --test          # CPU, once (refuses if action_end_test.json exists)
python -m meccano_action_end.ablation                   # CPU, diagnostic
```

The earlier offline MECCANO recipe (62.24 %, it read later frames; `scripts/meccano/`) was removed from this folder.

## Data and model credits

- IndustReal dataset: Schoonbeek et al., *IndustReal: A Dataset for Procedure Step Recognition
  Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting*, WACV 2024.
  Videos and labels are not included in this repository; follow the dataset's license.
- DINOv2 (`facebook/dinov2-small`), VideoMAE (`MCG-NJU/videomae-base-finetuned-ssv2`),
  SAM 3.1 (`facebook/sam3.1`) are downloaded from Hugging Face under their own licenses.
