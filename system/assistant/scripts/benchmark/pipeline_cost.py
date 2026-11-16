"""Cost of the online recognition pipeline (72.81 % top-1) on this machine.

Inference: every component is timed on one test video, end to end from the video file (or the JPEG frame cache
where the pipeline reads it), with its peak GPU memory. Speed is reported as frames per second of video
processed; the videos are 10 fps, so >= 10 fps is real time for that component.
Training: GPU time of every trained model, from the run folders (VideoMAE) and fold reports (TCN).

    python -m scripts.benchmark.pipeline_cost --recording 18_assy_2_5
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from monitor.settings import DINO_V2_DIR, ROOT

EVAL = ROOT / "output/official_split_eval"
VMAE_FT, DINO_FT = EVAL / "vmae_finetune", EVAL / "finetune"
VIDEOS = ROOT / "data/generated/industreal/videos"
OUT = ROOT / "output/benchmark"
VIDEO_FPS = 10.0


def gpu_name() -> str:
    return torch.cuda.get_device_name(0)


class Timer:
    """Wall time and peak GPU memory of one component."""

    def __init__(self, name: str, frames: int, rows: list):
        self.name, self.frames, self.rows = name, frames, rows

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.start
        self.rows.append({"component": self.name, "seconds": round(seconds, 2),
                          "fps": round(self.frames / seconds, 1),
                          "realtime_factor": round(self.frames / seconds / VIDEO_FPS, 2),
                          "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)})
        print(json.dumps(self.rows[-1]), flush=True)


def finetuned_dino(video: Path, checkpoint: Path, n_obj: int, n_verb: int, batch: int = 32) -> np.ndarray:
    """Hands crop of every frame through a fine-tuned DINOv2 (as dump_ar_crops + finetune_dino_ar.extract)."""
    from scripts.features.dump_ar_crops import CROP_SIZE, HANDS_BOX
    from scripts.features.finetune_dino_ar import INPUT_HW, Net, _to_tensor, embed_batch

    model = Net(n_obj, n_verb).to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval()
    cap, parts, buf = cv2.VideoCapture(str(video)), [], []
    while True:
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            crop = frame[int(HANDS_BOX[1] * h):int(HANDS_BOX[3] * h), int(HANDS_BOX[0] * w):int(HANDS_BOX[2] * w)]
            crop = cv2.resize(cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA), (INPUT_HW[1], INPUT_HW[0]),
                              interpolation=cv2.INTER_AREA)
            buf.append(_to_tensor(crop))
        if buf and (len(buf) >= batch or not ok):
            parts.append(embed_batch(model, torch.from_numpy(np.stack(buf)).to("cuda"))["feats"])
            buf = []
        if not ok:
            break
    cap.release()
    del model
    return np.concatenate(parts)


def tcn_inference(run: Path, rid: str, rows: list, frames: int) -> None:
    """All fold x seed networks of one causal TCN on both test feature variants (as train_ar_object_tcn)."""
    from scripts.segmentation.train_ar_object_tcn import TCN, load_features, predict_proba

    bundle = torch.load(run / "model.pt", map_location="cuda", weights_only=False)
    report = json.loads((run / "ar_object_paper_metrics.json").read_text(encoding="utf-8"))
    base = Path(report["features"][0])
    extras = [Path(p) for p in report["features"][1:]]
    pairs = [[Path(a), Path(b)] for a, b in report["crossfit"]]
    variants = [load_features(rid, base, False, extras + [p[i] for p in pairs]) for i in (0, 1)]
    models = []
    for sd in bundle["models"]:
        m = TCN(bundle["dim"], len(bundle["classes"]), hidden=bundle["hidden"], layers=bundle["layers"],
                stages=bundle["stages"], aux=bundle["aux"], causal=True).to("cuda")
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append(m.eval())
    with Timer(f"tcn_{bundle['target']} ({len(models)} networks x 2 feature variants)", frames, rows):
        np.mean([predict_proba(m, v, "cuda") for m in models for v in variants], axis=0)
    del models


def inference(rid: str) -> list[dict]:
    from transformers import AutoModel

    from scripts.data.tas_labels import load_mapping
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as ssv2_x
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features.finetune_dino_ar import VERBS

    video = VIDEOS / f"{rid}.mp4"
    frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    rows: list[dict] = []
    actions = load_mapping()

    dino = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
    with Timer("dinov2_small_3_views (decode + embed)", frames, rows):
        dino_x.extract(dino, video, 16)
    with Timer("dinov2_small_hires_hands (decode + embed)", frames, rows):
        dino_x.extract(dino, video, 16, views=dino_x.HIRES_VIEWS, grid=True)
    del dino

    meta = json.loads((DINO_FT / "meta.json").read_text(encoding="utf-8"))
    for k in (0, 1):
        with Timer(f"finetuned_dinov2_fold{k} (decode + crop + embed)", frames, rows):
            finetuned_dino(video, DINO_FT / f"fold{k}.pt", len(meta["objects"]), len(VERBS))
    torch.cuda.empty_cache()

    ssv2 = ssv2_x.load_model()
    with Timer("videomae_ssv2_causal (decode + 16-frame clip every 4 frames)", frames, rows):
        ssv2_x.extract(ssv2, video, 4, 4, causal=True)
    del ssv2
    torch.cuda.empty_cache()

    for k in (0, 1):
        model = vft.load_model(len(actions), VMAE_FT / f"fold{k}" / "best.pt")
        with Timer(f"finetuned_videomae_fold{k}_causal (frame cache + clip every 4 frames)", frames, rows):
            vft.extract(model, rid, 4, 8, causal=True)
        del model
        torch.cuda.empty_cache()

    for target in ("verb", "noun"):
        tcn_inference(EVAL / f"tcn_{target}_causal", rid, rows, frames)
    torch.cuda.empty_cache()

    clips, _ = vft.segment_clips([rid], actions)
    for name in ("refit_trainval", "refit_trainval_s8"):
        model = vft.load_model(len(actions), VMAE_FT / name / "best.pt")
        with Timer(f"videomae_classifier_{name} ({len(clips)} actions x 3 samplings)", frames, rows):
            vft.tta_probs(model, clips, 8)
        del model
        torch.cuda.empty_cache()
    return rows


def run_hours(path: Path) -> float:
    """Training time of a VideoMAE run: folder creation -> done.json."""
    start = datetime.fromtimestamp(path.stat().st_ctime)
    end = datetime.fromtimestamp((path / "done.json").stat().st_mtime)
    return round((end - start).total_seconds() / 3600, 2)


def training() -> dict:
    vmae = {n: run_hours(VMAE_FT / n) for n in ("fold0", "fold1", "cv52_0", "cv52_1", "cv52_2", "cv52_3",
                                                 "refit_trainval", "refit_trainval_s8")}
    tcn = {}
    for t in ("verb", "noun"):
        report = json.loads((EVAL / f"tcn_{t}_causal/ar_object_paper_metrics.json").read_text(encoding="utf-8"))
        tcn[t] = round(sum(f.get("seconds", 0) for f in report["folds"]) / 3600, 2)
    return {"videomae_finetune_hours": vmae, "videomae_total_hours": round(sum(vmae.values()), 2),
            "causal_tcn_hours": tcn, "causal_tcn_total_hours": round(sum(tcn.values()), 2),
            "not_included": "DINOv2 fine-tuning (2 backbones, no timing recorded), frame-cache building and "
                            "feature extraction for the 84 videos"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default="18_assy_2_5")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = inference(args.recording)
    frames = int(cv2.VideoCapture(str(VIDEOS / f"{args.recording}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    total = sum(r["seconds"] for r in rows)
    report = {"gpu": gpu_name(), "recording": args.recording, "frames": frames,
              "video_seconds": round(frames / VIDEO_FPS, 1),
              "inference": rows,
              "inference_total_seconds": round(total, 1),
              "pipeline_fps": round(frames / total, 1),
              "pipeline_realtime_factor": round(frames / total / VIDEO_FPS, 2),
              "peak_vram_mb": max(r["vram_peak_mb"] for r in rows),
              "training": training()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pipeline_cost_{args.recording}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "inference"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
