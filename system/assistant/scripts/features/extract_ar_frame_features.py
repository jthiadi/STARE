"""Extract per-frame DINOv2-small features (3 views) for AR-labeled recordings.

Views: full frame, centre region, lower-centre (hands) region. Each view stores
CLS + mean patch token (768 dims, fp16). One ``<recording>.npz`` per video;
existing files are skipped so the job can be resumed after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from monitor.settings import DINO_V2_DIR, ROOT

# (x1, y1, x2, y2) as fractions of the frame, output (h, w) multiple of 14.
VIEWS = (
    ("full", (0.0, 0.0, 1.0, 1.0), (224, 392)),
    ("centre", (0.25, 0.25, 0.75, 0.75), (224, 392)),
    ("hands", (0.2, 0.35, 0.8, 1.0), (266, 322)),
)
HIRES_VIEWS = (
    ("hands_hi", (0.2, 0.3, 0.8, 1.0), (364, 448)),
)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _set_low_priority() -> None:
    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _view_tensor(frame: np.ndarray, box, size) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    patch = cv2.resize(frame[y1:y2, x1:x2], (size[1], size[0]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


@torch.inference_mode()
def _embed(model, batch: np.ndarray, grid: bool = False) -> np.ndarray:
    x = torch.from_numpy(batch).to("cuda", dtype=torch.float16)
    hidden = model(pixel_values=x).last_hidden_state
    parts = [hidden[:, 0], hidden[:, 1:].mean(dim=1)]
    if grid:
        gh, gw = batch.shape[2] // 14, batch.shape[3] // 14
        patches = hidden[:, 1:].reshape(hidden.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patches, 2).flatten(2).permute(0, 2, 1)
        parts.extend(pooled[:, i] for i in range(4))
    feats = torch.cat(parts, dim=1)
    return feats.float().cpu().numpy().astype(np.float16)


def extract(model, video: Path, batch: int, views=VIEWS, grid: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dim = 384 * (6 if grid else 2)
    out = np.zeros((n, len(views), dim), dtype=np.float16)
    buffers: list[list[np.ndarray]] = [[] for _ in views]
    indices: list[int] = []
    index = 0

    def flush() -> None:
        for v in range(len(views)):
            feats = _embed(model, np.stack(buffers[v]), grid=grid)
            out[indices, v] = feats
            buffers[v].clear()
        indices.clear()

    while index < n:
        ok, frame = cap.read()
        if not ok:
            break
        for v, (_, box, size) in enumerate(views):
            buffers[v].append(_view_tensor(frame, box, size))
        indices.append(index)
        index += 1
        if len(indices) >= batch:
            flush()
    if indices:
        flush()
    cap.release()
    return out[:index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/ar_frame_features")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hires", action="store_true",
                        help="High-resolution hands view with 2x2 spatial pooling.")
    args = parser.parse_args()
    views = HIRES_VIEWS if args.hires else VIEWS

    _set_low_priority()
    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))
    from transformers import AutoModel

    model = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True)
    model = model.to("cuda").half().eval()

    args.output.mkdir(parents=True, exist_ok=True)
    recordings = sorted(
        p.name for p in (ROOT / "data/evaluation/official").iterdir()
        if (p / "AR_labels.csv").is_file()
    )
    for rid in recordings:
        target = args.output / f"{rid}.npz"
        video = ROOT / "data/generated/industreal/videos" / f"{rid}.mp4"
        if target.is_file() or not video.is_file():
            continue
        start = time.time()
        feats = extract(model, video, args.batch, views=views, grid=args.hires)
        np.savez(target, feats=feats, views=np.array([v[0] for v in views]))
        torch.cuda.empty_cache()
        print(json.dumps({
            "recording": rid,
            "frames": int(feats.shape[0]),
            "seconds": round(time.time() - start, 1),
            "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20),
        }), flush=True)


if __name__ == "__main__":
    main()
