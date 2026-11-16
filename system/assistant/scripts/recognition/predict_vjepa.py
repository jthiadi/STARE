"""Action labels for given segments of a video: the fine-tuned V-JEPA 2 ensemble (IndustReal test top-1 77.2 %).

Four V-JEPA 2 ViT-L runs (work-area or hand crop, last 8 or 12 of 24 layers fine-tuned) and the fine-tuned
VideoMAE each give 74 action probabilities per segment (mean over 3 temporal samplings); they are averaged and
the video's segments are decoded together with the action-order prior. All settings are the ones chosen on
the IndustReal val videos (``settings.json`` next to the weights).

The model labels actions, it does not find them: the segments come from annotations or from another model,
e.g. ``predict_actions``' ``segments.csv``. Accuracy is lower when the boundaries are not the annotated ones.

    python -m scripts.recognition.predict_vjepa --video my_video.mp4 --segments segments.csv
    -> output/predict_vjepa/<video>/labels.csv   start_s, end_s, action, confidence

``segments.csv`` needs ``start_s`` and ``end_s`` columns (seconds). Weights: ``release_vjepa/`` (the two
``industreal_vjepa2_recognition_*.zip`` release assets unzipped together); the base V-JEPA 2 and VideoMAE
models are downloaded from Hugging Face on first use. Needs a GPU with about 8 GB and MediaPipe (hand crop).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping
from scripts.features import finetune_videomae_ar as vft
from scripts.features import vjepa2_finetune as ft
from scripts.features.finetune_videomae_ar import TTA_FRACS
from scripts.recognition.sequence_recognition import decode_segments, videos

WEIGHTS = ROOT / "release_vjepa"
OUT = ROOT / "output/predict_vjepa"


def member_probs(segments: list[tuple[str, int, int]], weights: Path, caches: dict[str, Path], batch: int = 8,
                 workers: int = 4, fracs: tuple = TTA_FRACS) -> dict[str, np.ndarray]:
    """Per ensemble member (run name, e.g. ``vjft_hand_k12``, and ``vmae_full``): (segments, 74) action
    probabilities for (video name, start frame, end frame) segments, mean over the temporal samplings ``fracs``;
    ``caches`` holds the ``vjepa`` / ``vmae`` frame caches and the ``hands`` landmarks folder (a file per video)."""
    cfg = json.loads((weights / "settings.json").read_text(encoding="utf-8"))
    out = {}
    for run in cfg["vjepa_runs"]:
        model = ft.Finetuned(run["layers"], cfg["size"]).cuda()
        model.load_state_dict(torch.load(weights / run["file"], map_location="cuda", weights_only=True), strict=False)
        items = [(name, s, e, 1, f) for name, s, e in segments for f in fracs]
        args = SimpleNamespace(crop=run["crop"], size=cfg["size"], context=cfg["context"], workers=workers)
        p = ft.predict(model, items, args, batch, caches["vjepa"], caches["hands"])
        out[Path(run["file"]).stem] = p.reshape(-1, len(fracs), p.shape[1]).mean(1)
        del model
        torch.cuda.empty_cache()
    model = vft.load_model(len(load_mapping()), weights / cfg["videomae"])
    out[Path(cfg["videomae"]).stem] = vft.tta_probs(model, [(name, s, e, 1) for name, s, e in segments], batch,
                                                    fracs, cache=caches["vmae"])
    del model
    torch.cuda.empty_cache()
    return out


def ensemble_probs(segments: list[tuple[str, int, int]], weights: Path, caches: dict[str, Path],
                   batch: int = 8, workers: int = 4) -> np.ndarray:
    """(segments, 74) action probabilities of the ensemble: the mean of ``member_probs``."""
    probs = member_probs(segments, weights, caches, batch, workers)
    return sum(probs.values()) / len(probs)


def decode(probs: np.ndarray, segments: list[tuple[str, int, int]], weights: Path) -> np.ndarray:
    """1-based action per segment: each video's segments in time order, decoded with the action-order prior."""
    cfg = json.loads((weights / "settings.json").read_text(encoding="utf-8"))
    counts = np.load(weights / cfg["order_counts"])
    groups = videos([f"{name}|{s}|{e}" for name, s, e in segments])
    o = cfg["order"]
    return decode_segments(np.log(probs + 1e-8), groups, counts["first"], counts["pairs"], o["strength"],
                           o["smoothing"], o["viterbi"])


def prepare(name: str, video: Path, work: Path, need_hands: bool) -> dict[str, Path]:
    """Frame caches (960 x 540 for V-JEPA 2, the 224 x 224 work-area crop for VideoMAE) and hand landmarks."""
    from scripts.mistakes.mistsense import extract_video

    caches = {"vjepa": work / "frames_960", "vmae": work / "frames_224", "hands": work / "hands"}
    for key in ("vjepa", "vmae"):
        caches[key].mkdir(parents=True, exist_ok=True)
    ft.build_cache(name, video, caches["vjepa"])
    vft.build_cache(name, video, caches["vmae"])
    if need_hands:
        extract_video(name, video, caches["hands"])
    return caches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True, help="CSV with start_s, end_s columns (seconds).")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out", type=Path, help="Default: output/predict_vjepa/<video>/labels.csv")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    name = args.video.stem
    cap = cv2.VideoCapture(str(args.video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    with args.segments.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    spans = [(max(0, int(round(float(r["start_s"]) * fps))), min(n - 1, int(round(float(r["end_s"]) * fps))))
             for r in rows]
    spans = [(s, e) for s, e in spans if e >= s]
    cfg = json.loads((args.weights / "settings.json").read_text(encoding="utf-8"))
    work = OUT / name
    caches = prepare(name, args.video, work, any(r["crop"] == "hand" for r in cfg["vjepa_runs"]))
    segments = [(name, s, e) for s, e in spans]
    probs = ensemble_probs(segments, args.weights, caches, args.batch)
    pred = decode(probs, segments, args.weights)
    actions = load_mapping()
    out = args.out or work / "labels.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_s", "end_s", "action", "confidence"])
        for (s, e), p, row in zip(spans, pred, probs):
            writer.writerow([round(s / fps, 2), round(e / fps, 2), actions[p], round(float(row[p - 1]), 3)])
    print(json.dumps({"video": str(args.video), "segments": len(spans), "written": str(out)}), flush=True)


if __name__ == "__main__":
    main()
