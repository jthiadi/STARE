"""The IndustReal action-end label (75.07 on the paper's test) on MECCANO, recipe unchanged and past only.

MECCANO's 62.24 (``meccano_action_end.recognition``) uses later frames in three ways: VideoMAE per-frame features centred on
each frame, a "video progress" input that needs the video's length, and a TCN that is not causal. This run removes all
three, so its number sits next to IndustReal's 75.07 on the same terms: true segment boundaries, each segment labelled
from its own frames and the earlier ones only.

Recipe, as on IndustReal (``assist/extract/action_end.py``): the mean of VideoMAE and three fine-tuned V-JEPA 2 runs (work
area k12, hand k12, hand k12 seed 1), each over 3 samplings of the segment; 0.95 of that against 0.05 of a past-only TCN;
the action-order prior with strength 0.2 and smoothing 0.01, decoded forward only. The prior counts come from the 11 train
videos. The one choice, which of the two MECCANO VideoMAE runs, is made on the 2 val videos. Test is scored once.

    python -m meccano_action_end.action_end --features      # GPU: DINOv2 (if missing) + past-only VideoMAE-SSv2 per-frame features
    python -m meccano_action_end.action_end --tcn           # GPU: past-only TCN, no progress input, train -> val + test
    python -m meccano_action_end.vjepa2 --crop hand --layers 12 --batch 4 --accum 4 --seed 1   # GPU: the third V-JEPA 2 run
    python -m meccano_action_end.action_end --val           # CPU -> output/meccano/action_end_val.json
    python -m meccano_action_end.action_end --test          # CPU, once -> output/meccano/action_end_test.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from assist.extract.action_end import SMOOTHING, STRENGTH, WEIGHT, end_labels
from meccano_action_end import data

VJ_MEMBERS = ("vjft_area_k12", "vjft_hand_k12", "vjft_hand_k12_s1")
VMAE_RUNS = ("off_s2", "off_s3")
FEATS_CAUSAL = data.OUT / "feats/vmae_ssv2_causal"
TCN_RUN = "official_causal"
VAL_RESULT = data.OUT / "action_end_val.json"
TEST_RESULT = data.OUT / "action_end_test.json"


def pick_vmae(val_top1: dict[str, float]) -> str:
    """The VideoMAE run with the best val top-1 on its own; a tie goes to the first in name order."""
    return max(sorted(val_top1), key=lambda name: val_top1[name])


def top1(pred: np.ndarray, labels: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(labels)).mean() * 100)


def per_video(clips: list[tuple]) -> list[np.ndarray]:
    """Indices of each video's segments, videos in first-seen order, segments in the given (start) order."""
    out: dict[str, list[int]] = {}
    for i, c in enumerate(clips):
        out.setdefault(c[0], []).append(i)
    return [np.array(v) for v in out.values()]


def check_can_score_test(val_result: Path, test_result: Path) -> None:
    if not val_result.is_file():
        raise FileNotFoundError(f"{val_result} is missing: run --val first, so the VideoMAE run is chosen on val")
    if test_result.is_file():
        raise FileExistsError(f"{test_result} exists: test has already been scored")


# ---------------------------------------------------------------------------------------------- GPU
def features() -> None:
    """DINOv2 per-frame features if missing (one frame each, so past only already), then past-only VideoMAE-SSv2."""
    import torch
    from transformers import AutoModel

    from monitor.settings import DINO_V2_DIR
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as vmae_x
    from meccano_action_end.recognition import FEATS

    vids = data.split("train") + data.split("val") + data.split("test")
    dino = FEATS["dino"]
    dino.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not (dino / f"{v}.npz").is_file()]
    if todo:
        model = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
        for vid in todo:
            start = time.time()
            feats = dino_x.extract(model, data.video_path(vid), 16)
            np.savez(dino / f"{vid}.npz", feats=feats, views=np.array([v[0] for v in dino_x.VIEWS]))
            print(json.dumps({"dino": vid, "frames": int(len(feats)), "seconds": round(time.time() - start, 1)}), flush=True)
        del model
        torch.cuda.empty_cache()
    FEATS_CAUSAL.mkdir(parents=True, exist_ok=True)
    model = vmae_x.load_model()
    for vid in vids:
        if (FEATS_CAUSAL / f"{vid}.npz").is_file():
            continue
        start = time.time()
        feats, logits = vmae_x.extract(model, data.video_path(vid), 4, 4, causal=True)
        np.savez(FEATS_CAUSAL / f"{vid}.npz", feats=feats[:, None, :], video_logits=logits)
        print(json.dumps({"features": vid, "frames": int(len(feats)), "seconds": round(time.time() - start, 1)}), flush=True)
    del model
    torch.cuda.empty_cache()


def _tcn_input(vid: str) -> np.ndarray:
    from meccano_action_end.recognition import FEATS
    from scripts.segmentation.train_ar_object_tcn import load_features

    return load_features(vid, FEATS["dino"], False, [FEATS_CAUSAL])          # no progress input: it needs the length


def tcn(seeds: int = 3) -> None:
    """Same TCN, epochs and seeds as ``recognition --train-tcn`` (official split), but causal and without progress."""
    import torch

    from meccano_action_end.recognition import TCN_DIR
    from scripts.segmentation.train_ar_object_tcn import _resident, predict_proba, train_model

    out = TCN_DIR / TCN_RUN
    pred_vids = data.split("val") + data.split("test")
    if all((out / f"{v}.npy").is_file() for v in pred_vids):
        return
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    classes = [str(i) for i in range(len(data.actions()))]
    train = []
    for v in data.split("train"):
        f = _tcn_input(v)
        train.append((_resident(f, "cuda"), [str(x) for x in data.frame_labels(v, len(f))]))
    models = [train_model(train, classes, epochs=80, seed=100 + seed, device="cuda", cosine=True, warp=True,
                          arch=(192, 10, 4), model_type="causal_tcn")[0] for seed in range(seeds)]
    del train
    torch.cuda.empty_cache()
    for v in pred_vids:
        f = _tcn_input(v)
        np.save(out / f"{v}.npy", np.mean([predict_proba(m, f, "cuda") for m in models], axis=0).astype(np.float16))
    print(json.dumps({"tcn": TCN_RUN, "seconds": round(time.time() - start, 1)}), flush=True)


# ---------------------------------------------------------------------------------------------- CPU
def _members(split: str, vmae_run: str) -> np.ndarray:
    """(segments, actions) mean of the four members, in ``data.clips`` order."""
    from meccano_action_end.recognition import PROBS, _vjepa_probs

    vj = _vjepa_probs()
    missing = [m for m in VJ_MEMBERS if m not in vj]
    if missing:
        raise FileNotFoundError(f"V-JEPA 2 runs not found: {missing} (train them with meccano_action_end.vjepa2)")
    probs = [np.load(PROBS)[f"{vmae_run}/{split}/whole"]] + [vj[m][split] for m in VJ_MEMBERS]
    return sum(probs) / len(probs)


def predict(split: str, vmae_run: str) -> tuple[np.ndarray, np.ndarray]:
    from meccano_action_end.recognition import TCN_DIR

    clips = data.clips(data.split(split))
    clip = _members(split, vmae_run)
    first, pairs = data.transition_counts(data.split("train"))
    pred = np.zeros(len(clips), np.int64)
    for idx in per_video(clips):
        frames = np.load(TCN_DIR / TCN_RUN / f"{clips[idx[0]][0]}.npy").astype(np.float32)
        spans = [(clips[i][1], clips[i][2]) for i in idx]
        pred[idx] = end_labels(frames, spans, clip[idx], WEIGHT, STRENGTH, SMOOTHING, first, pairs)
    return pred, np.array([c[3] for c in clips])


def val() -> None:
    from meccano_action_end.recognition import PROBS

    labels = np.array([c[3] for c in data.clips(data.split("val"))])
    alone = {r: round(top1(np.load(PROBS)[f"{r}/val/whole"].argmax(1) + 1, labels), 2) for r in VMAE_RUNS}
    chosen = pick_vmae(alone)
    pred, labels = predict("val", chosen)
    report = {"protocol": "MECCANO, true segment boundaries, 61 actions, past only; IndustReal action-end recipe unchanged",
              "videomae_alone_val_top1": alone, "videomae_run": chosen, "val_top1": round(top1(pred, labels), 2),
              "segments": len(labels)}
    VAL_RESULT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


def test() -> None:
    check_can_score_test(VAL_RESULT, TEST_RESULT)
    chosen = json.loads(VAL_RESULT.read_text(encoding="utf-8"))["videomae_run"]
    pred, labels = predict("test", chosen)
    report = {"protocol": "MECCANO test, true segment boundaries, 61 actions, past only; scored once",
              "videomae_run": chosen, "test_top1": round(top1(pred, labels), 2), "segments": len(labels),
              "published_rgb_test_top1": {"SlowFast (CVIU 2023)": 45.16, "challenge winner 2023 (multimodal)": 52.82}}
    TEST_RESULT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("features", "tcn", "val", "test"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.features:
        features()
    if args.tcn:
        tcn()
    if args.val:
        val()
    if args.test:
        test()


if __name__ == "__main__":
    main()
