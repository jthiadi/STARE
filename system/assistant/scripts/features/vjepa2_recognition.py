"""IndustReal action recognition from a frozen V-JEPA 2 ViT-L video encoder plus a small attentive probe.

Same protocol and sampling as the fine-tuned VideoMAE (``finetune_videomae_ar --score-clips``): every annotated
segment gives 16 frames spread over its ground-truth start/end, of the hands / work-area crop (here at 256 px,
V-JEPA 2's resolution). The encoder is never trained, so no cross-fitting is needed: one probe is trained on the
36 train videos, its epoch is chosen on the 16 val videos, and test is scored once.

    python -m scripts.features.vjepa2_recognition --extract --shard 0/2   # GPU 0 (and --shard 1/2 on GPU 1)
    python -m scripts.features.vjepa2_recognition --train                 # -> clip_probs_vjepa.npz

Extraction stores, per sampling, the encoder tokens average-pooled to a 4 x 4 grid (8 x 4 x 4 = 128 tokens of
1024), one ``<recording>.npz`` per video (resumable). Training videos get 4 jittered samplings per segment,
val / test videos the 3 fixed samplings of ``TTA_FRACS``; their probabilities are averaged.
``clip_probs_vjepa.npz`` has the layout of the VideoMAE ``clip_probs.npz`` (``vjepa/<split>``, ``keys/<split>``,
``labels/<split>``), so ``fuse_recognition`` can weigh it against the other streams on val.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping, read_segments
from scripts.features.extract_videomae_features import BOX
from scripts.features.finetune_videomae_ar import TTA_FRACS, sample_indices
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
MODEL_DIR = ROOT / "models/vjepa2-vitl-fpc64-256"
VIDEOS = ROOT / "data/generated/industreal/videos"
OUT = ROOT / "output/official_split_eval/vjepa2"
SIZE, GRID, TRAIN_SAMPLINGS = 256, 4, 4
SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ---------------------------------------------------------------- extraction (GPU)
def download() -> None:
    """Only the safetensors weights and configs (the repo also holds a 5 GB original checkpoint)."""
    if not (MODEL_DIR / "model.safetensors").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_ID, local_dir=MODEL_DIR, allow_patterns=["*.json", "model.safetensors"])


def video_frames(rid: str) -> np.ndarray:
    """Every frame's hands / work-area crop, (n, SIZE, SIZE, 3) RGB uint8."""
    cap, out = cv2.VideoCapture(str(VIDEOS / f"{rid}.mp4")), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(BOX[0] * w), int(BOX[1] * h), int(BOX[2] * w), int(BOX[3] * h)
        out.append(cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out)


def segments(rid: str, n: int, actions: list[str]) -> list[tuple[str, int, int, int]]:
    """(key ``rid|start|end`` as annotated, clamped start, clamped end, 1-based action) per annotated segment."""
    index = {a: i for i, a in enumerate(actions)}
    return [(f"{rid}|{s}|{e}", max(0, s), min(n - 1, e), index[a]) for a, s, e in read_segments(rid)
            if min(n - 1, e) >= max(0, s) and index[a] > 0]


@torch.inference_mode()
def encode(model, clips: np.ndarray) -> np.ndarray:
    """(B, 16, SIZE, SIZE, 3) uint8 -> (B, 128, 1024) float16 encoder tokens, pooled to a GRID x GRID grid."""
    x = torch.from_numpy(clips).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
    x = ((x - _MEAN.cuda()) / _STD.cuda()).half()
    tokens = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state   # (B, 8 * 16 * 16, 1024)
    b, t, s = len(clips), clips.shape[1] // 2, SIZE // 16
    grid = tokens.view(b, t, s, s, -1).permute(0, 1, 4, 2, 3).flatten(0, 1)        # (B * 8, 1024, 16, 16)
    pooled = nn.functional.adaptive_avg_pool2d(grid.float(), GRID)                   # (B * 8, 1024, 4, 4)
    return pooled.view(b, t, -1, GRID * GRID).permute(0, 1, 3, 2).reshape(b, -1, tokens.shape[-1]).half() \
        .cpu().numpy()


def extract(shard: int, shards: int, batch: int) -> None:
    from transformers import AutoModel

    download()
    model = AutoModel.from_pretrained(MODEL_DIR, attn_implementation="sdpa").cuda().half().eval()
    actions = load_mapping()
    jobs = [(rid, split) for split, rids in SPLITS.items() for rid in rids][shard::shards]
    OUT.mkdir(parents=True, exist_ok=True)
    for rid, split in jobs:
        target = OUT / f"{rid}.npz"
        if target.is_file():
            continue
        start = time.time()
        frames = video_frames(rid)
        np.random.seed(zlib.crc32(rid.encode()))                  # reproducible training jitter
        keys, labels, clips = [], [], []
        for key, s, e, label in segments(rid, len(frames), actions):
            draws = ([sample_indices(s, e, True) for _ in range(TRAIN_SAMPLINGS)] if split == "train"
                     else [sample_indices(s, e, False, f) for f in TTA_FRACS])
            for idx in draws:
                keys.append(key)
                labels.append(label)
                clips.append(idx)
        tokens = np.concatenate([encode(model, frames[np.array(clips[i:i + batch])])
                                 for i in range(0, len(clips), batch)])
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, tokens=tokens, keys=np.array(keys), labels=np.array(labels))
        tmp.replace(target)
        print(json.dumps({"recording": rid, "split": split, "samples": len(clips),
                          "seconds": round(time.time() - start, 1),
                          "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)


# ---------------------------------------------------------------- probe
class AttentiveProbe(nn.Module):
    """One learned query cross-attends over the clip's tokens (V-JEPA's frozen-evaluation recipe), then a
    linear classifier."""

    def __init__(self, dim: int = 1024, tokens: int = 128, classes: int = 74, heads: int = 16):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x + self.pos)
        h = self.query.expand(len(x), -1, -1)
        h = h + self.attn(h, x, x, need_weights=False)[0]
        return self.head((h + self.mlp(h))[:, 0])


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tokens, keys, 1-based labels) of every sampling of a split's videos."""
    parts = [np.load(OUT / f"{rid}.npz") for rid in SPLITS[split]]
    return (np.concatenate([p["tokens"] for p in parts]), np.concatenate([p["keys"] for p in parts]),
            np.concatenate([p["labels"] for p in parts]))


@torch.inference_mode()
def segment_probs(model, tokens: np.ndarray, batch: int = 256) -> np.ndarray:
    """Per segment, in stored order: softmax over the 74 actions, averaged over its len(TTA_FRACS) consecutive
    samplings (by position: two segments can share the same start and end)."""
    model.eval()
    probs = np.concatenate([torch.softmax(model(torch.from_numpy(tokens[i:i + batch]).cuda().float()), 1)
                            .cpu().numpy() for i in range(0, len(tokens), batch)])
    return probs.reshape(-1, len(TTA_FRACS), probs.shape[1]).mean(1)


def train(epochs: int, seeds: int, batch: int) -> None:
    tr_x, _, tr_y = load_split("train")
    evals = {s: load_split(s) for s in ("val", "test")}
    per = len(TTA_FRACS)
    keys = {s: evals[s][1][::per] for s in evals}
    labels = {s: evals[s][2][::per] for s in evals}
    report, summed = {"protocol": "IndustReal action recognition, GT segments, 74 actions; frozen V-JEPA 2 ViT-L, "
                                  "attentive probe trained on the 36 train videos, epoch chosen on val",
                      "seeds": []}, {}
    for seed in range(seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = AttentiveProbe().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
        steps = epochs * (len(tr_x) // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=steps, pct_start=0.1)
        best = {"val_top1": -1.0}
        for epoch in range(epochs):
            model.train()
            perm = rng.permutation(len(tr_x))
            for i in range(0, len(perm) - batch + 1, batch):
                idx = np.sort(perm[i:i + batch])
                x = torch.from_numpy(tr_x[idx]).cuda().float()
                y = torch.from_numpy(tr_y[idx] - 1).cuda()
                loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            top1 = float((segment_probs(model, evals["val"][0]).argmax(1) + 1 == labels["val"]).mean() * 100)
            print(json.dumps({"seed": seed, "epoch": epoch + 1, "loss": round(loss.item(), 3),
                              "val_top1": round(top1, 2)}), flush=True)
            if top1 > best["val_top1"]:
                best = {"epoch": epoch + 1, "val_top1": round(top1, 2),
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        model.load_state_dict(best.pop("state"))
        torch.save(model.state_dict(), OUT / f"probe_seed{seed}.pt")
        for split in evals:
            p = segment_probs(model, evals[split][0])
            summed[split] = summed.get(split, 0) + p / seeds
            best[f"{split}_top1"] = round(float((p.argmax(1) + 1 == labels[split]).mean() * 100), 2)
        report["seeds"].append({"seed": seed, **best})
        print(json.dumps(report["seeds"][-1]), flush=True)
    arrays = {}
    for split, p in summed.items():
        arrays[f"keys/{split}"] = keys[split]
        arrays[f"labels/{split}"] = labels[split]
        arrays[f"vjepa/{split}"] = p.astype(np.float32)
        report[f"{split}_top1_seed_mean"] = round(float((p.argmax(1) + 1 == labels[split]).mean() * 100), 2)
    np.savez(OUT / "clip_probs_vjepa.npz", **arrays)
    (OUT / "recognition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "seeds"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th video from i.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.extract:
        shard, shards = (int(v) for v in args.shard.split("/"))
        extract(shard, shards, args.batch)
    if args.train:
        train(args.epochs, args.seeds, 64)


if __name__ == "__main__":
    main()
