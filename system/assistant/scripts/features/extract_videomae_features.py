"""Per-frame motion features from VideoMAE-base fine-tuned on Something-Something-v2 (hand-object actions).

A 16-frame clip (1.6 s at the videos' 10 fps) of the hands region is centred every ``--stride`` frames;
each frame takes the nearest clip's pooled embedding (768) and its 174 SSv2 action logits.
One ``<recording>.npz`` per video (``feats`` (T, 1, 768), ``video_logits`` (T, 174)); resumable.
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

from monitor.settings import ROOT

MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
MODEL_DIR = ROOT / "models/videomae-base-ssv2"
BOX = (0.1, 0.15, 0.9, 1.0)  # (x1, y1, x2, y2) fractions: hands and work area
CLIP = 16
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prep(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(BOX[0] * w), int(BOX[1] * h), int(BOX[2] * w), int(BOX[3] * h)
    patch = cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def _embed(model, clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.stack(clips)).to("cuda").float() / 255.0  # (B, T, H, W, C)
    mean = torch.tensor(_MEAN, device="cuda")
    std = torch.tensor(_STD, device="cuda")
    x = ((x - mean) / std).permute(0, 1, 4, 2, 3).half()  # (B, T, C, H, W)
    seq = model.videomae(pixel_values=x).last_hidden_state
    pooled = model.fc_norm(seq.mean(dim=1))
    logits = model.classifier(pooled)
    return pooled.float().cpu().numpy(), logits.float().cpu().numpy()


def extract(model, video: Path, stride: int, batch: int, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``causal``: each clip ends at its anchor frame and a frame takes the latest clip ending at or before it,
    so no frame's feature uses later frames."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    centres = list(range(0, n, stride))
    back = CLIP - 1 if causal else CLIP // 2
    frames: dict[int, np.ndarray] = {}
    read = 0
    pooled, logits, pending = [], [], []

    def flush():
        p, lg = _embed(model, pending)
        pooled.append(p)
        logits.append(lg)
        pending.clear()

    last = 0
    for c in centres:
        idx = [min(max(i, 0), n - 1) for i in range(c - back, c - back + CLIP)]
        while read <= idx[-1] and read < n:
            ok, frame = cap.read()
            if not ok:
                n = read
                break
            frames[read] = _prep(frame)
            last = read
            read += 1
        idx = [min(i, last) for i in idx]
        pending.append(np.stack([frames[i] for i in idx]))
        for k in [k for k in frames if k < c + stride - back]:
            del frames[k]
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    cap.release()
    pooled_c, logits_c = np.concatenate(pooled), np.concatenate(logits)
    nearest = np.arange(n) // stride if causal else (np.arange(n) + stride // 2) // stride
    nearest = np.minimum(nearest, len(pooled_c) - 1)
    return pooled_c[nearest].astype(np.float16), logits_c[nearest].astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/videomae_ssv2_features")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--recordings", nargs="*", default=None)
    parser.add_argument("--causal", action="store_true", help="Clips end at each frame (no future frames).")
    args = parser.parse_args()

    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    model = load_model()

    args.output.mkdir(parents=True, exist_ok=True)
    recordings = args.recordings or sorted(
        p.name for p in (ROOT / "data/evaluation/official").iterdir() if (p / "AR_labels.csv").is_file())
    for rid in recordings:
        target = args.output / f"{rid}.npz"
        video = ROOT / "data/generated/industreal/videos" / f"{rid}.mp4"
        if target.is_file() or not video.is_file():
            continue
        start = time.time()
        feats, logits = extract(model, video, args.stride, args.batch, args.causal)
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, feats=feats[:, None, :], video_logits=logits)
        tmp.replace(target)
        torch.cuda.empty_cache()
        print(json.dumps({"recording": rid, "frames": int(feats.shape[0]),
                          "seconds": round(time.time() - start, 1),
                          "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)


def load_model():
    """VideoMAE-SSv2 on the GPU in fp16; downloaded on first use."""
    download()
    from transformers import VideoMAEForVideoClassification

    return VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to("cuda").half().eval()


def download() -> None:
    """Fetch VideoMAE-SSv2 into models/ and convert its .bin weights to safetensors."""
    if not (MODEL_DIR / "config.json").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_ID, local_dir=MODEL_DIR,
                          allow_patterns=["*.json", "*.safetensors", "*.bin"])
    if not (MODEL_DIR / "model.safetensors").is_file():
        from safetensors.torch import save_file

        state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in state.items()}, MODEL_DIR / "model.safetensors")
        (MODEL_DIR / "pytorch_model.bin").unlink()


if __name__ == "__main__":
    main()
