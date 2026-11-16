"""IndustReal action recognition: V-JEPA 2 ViT-L with its last ``--layers`` transformer layers fine-tuned.

Same protocol as ``vjepa2_recognition`` (16 frames inside each ground-truth segment, 74 actions, official split):
trained on the 36 train videos, epoch chosen on the 16 val videos, then val and test are scored with the 3
``TTA_FRACS`` samplings. The first layers stay frozen and run without gradients, so it fits an 11 GB GPU.

Two crops (``--crop``):
  area   the hands / work-area box used by every other model here (``BOX``)
  hand   a square around both hands over the clip's 16 frames (MediaPipe landmarks, ``output/mistakes/hands``),
         1.6 x their extent, at least 40 % of the frame height; the area box when no hand is seen. Small parts
         (nut vs tooth washer) get about twice the pixels.

    python -m scripts.features.vjepa2_finetune --cache                     # 960 x 540 JPEG frames, all 84 videos
    python -m scripts.features.vjepa2_finetune --crop area --layers 8      # -> clip_probs_vjft_area_k8.npz
    python -m scripts.features.vjepa2_finetune --crop hand --layers 8
    python -m scripts.features.vjepa2_finetune --crop area --layers 12 --batch 4 --accum 4
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4 --seed 1  # -> ..._k12_s1
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 2 --accum 8 --size 320  # -> ..._r320
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4 --context 0.25  # _c25
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4 --epochs 15  # _e15
    python -m scripts.features.vjepa2_finetune --crop hand --layers 12 --batch 4 --accum 4 --prefix 0.1  # _p10

``--prefix f`` trains on beginnings of actions: each training clip runs from the action's start to a random point
10-100 % (for 0.1) of the way through it, the question the live assistant asks (``assist/extract/online_vjepa.py``
scores the clip [start, now]). Val and test are still scored on whole segments.

The probabilities file has the ``clip_probs.npz`` layout (``<name>/<split>``, ``keys/<split>``, ``labels/<split>``)
and is offered to ``sequence_recognition --vjepa`` next to the other streams.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn

from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping
from scripts.features.extract_videomae_features import BOX
from scripts.features.finetune_videomae_ar import TTA_FRACS, Frames, sample_indices
from scripts.features.vjepa2_recognition import (MODEL_DIR, OUT, SIZE, SPLITS, VIDEOS, AttentiveProbe, _MEAN, _STD,
                                                 download, segments)

CACHE = ROOT / "output/vjepa_frames_960"
HANDS = ROOT / "output/mistakes/hands"
W, H = 960, 540


# ---------------------------------------------------------------- frame cache
def build_cache(rid: str, video: Path | None = None, cache: Path = CACHE) -> str:
    """Every frame at W x H as JPEG in one ``<rid>.bin`` with byte offsets in ``<rid>.idx.npy``."""
    if (cache / f"{rid}.idx.npy").is_file():
        return rid
    cap, offsets = cv2.VideoCapture(str(video or VIDEOS / f"{rid}.mp4")), [0]
    tmp = cache / f"{rid}.bin.tmp"
    with tmp.open("wb") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = cv2.imencode(".jpg", cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            out.write(raw)
            offsets.append(offsets[-1] + len(raw))
    cap.release()
    tmp.replace(cache / f"{rid}.bin")
    np.save(cache / f"{rid}.idx.npy", np.array(offsets, np.int64))
    print(json.dumps({"cached": rid, "frames": len(offsets) - 1}), flush=True)
    return rid


# ---------------------------------------------------------------- clips
def crop_box(points: np.ndarray | None) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) in cache pixels: the work-area box, or a square around ``points`` (normalised x, y)."""
    if points is None or not len(points):
        return int(BOX[0] * W), int(BOX[1] * H), int(BOX[2] * W), int(BOX[3] * H)
    (x1, y1), (x2, y2) = points.min(0) * (W, H), points.max(0) * (W, H)
    side = int(np.clip(1.6 * max(x2 - x1, y2 - y1), 0.4 * H, H))
    cx = int(np.clip((x1 + x2) / 2, side / 2, W - side / 2))
    cy = int(np.clip((y1 + y2) / 2, side / 2, H - side / 2))
    return cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side


class Clips(torch.utils.data.Dataset):
    """(16 x size x size x 3 uint8 clip, 0-based action) per (rid, start, end, label, frac) item. ``context``
    widens the sampled span by that fraction of the segment's length on each side (clipped to the video)."""

    def __init__(self, items: list[tuple], crop: str, train: bool, size: int = SIZE, context: float = 0.0,
                 cache: Path = CACHE, hands_dir: Path = HANDS, prefix: float = 0.0):
        self.items, self.crop, self.train, self.hands, self.prefix = items, crop, train, {}, prefix
        self.size, self.context, self.cache, self.hands_dir = size, context, cache, hands_dir

    def __len__(self) -> int:
        return len(self.items)

    def _points(self, rid: str, idx: list[int]) -> np.ndarray | None:
        if self.crop != "hand":
            return None
        if rid not in self.hands:
            self.hands[rid] = np.load(self.hands_dir / f"{rid}.npz")["landmarks"][..., :2]
        lm = self.hands[rid][np.minimum(idx, len(self.hands[rid]) - 1)].reshape(-1, 2)
        return lm[(lm > 0).any(1)]

    def __getitem__(self, i: int):
        rid, s, e, label, frac = self.items[i]
        video = Frames(rid, self.cache)
        if self.train and self.prefix:                    # a beginning of the action: start .. 10-100 % of it
            e = s + int(round(np.random.uniform(self.prefix, 1.0) * (e - s)))
        pad = int(round(self.context * (e - s + 1)))
        idx = sample_indices(max(0, s - pad), min(len(video) - 1, e + pad), self.train, frac)
        frames = video.get(idx)
        x1, y1, x2, y2 = crop_box(self._points(rid, idx))
        if self.train:                                    # mild zoom / shift
            side = x2 - x1
            new = int(side * np.random.uniform(0.85, 1.0))
            x1 += np.random.randint(0, side - new + 1)
            y1 += np.random.randint(0, (y2 - y1) - int((y2 - y1) * new / side) + 1)
            x2, y2 = x1 + new, y1 + int((y2 - y1) * new / side)
        clip = np.stack([cv2.resize(f[y1:y2, x1:x2], (self.size, self.size), interpolation=cv2.INTER_AREA)
                         for f in frames])
        return torch.from_numpy(clip), label - 1


def items_for(split: str, fracs: tuple) -> tuple[list[tuple], list[str]]:
    actions = load_mapping()
    items, keys = [], []
    for rid in SPLITS[split]:
        for key, s, e, label in segments(rid, len(Frames(rid, CACHE)), actions):
            keys.append(key)
            items += [(rid, s, e, label, f) for f in fracs]
    return items, keys


def _seed_worker(_):
    np.random.seed(torch.initial_seed() % 2**32)


# ---------------------------------------------------------------- model
class Finetuned(nn.Module):
    def __init__(self, layers: int, size: int = SIZE, classes: int = 74):
        super().__init__()
        from transformers import AutoModel

        self.enc = AutoModel.from_pretrained(MODEL_DIR, attn_implementation="sdpa").encoder
        self.split = len(self.enc.layer) - layers
        for p in self.enc.parameters():
            p.requires_grad = False
        for m in list(self.enc.layer[self.split:]) + [self.enc.layernorm]:
            for p in m.parameters():
                p.requires_grad = True
        self.probe = AttentiveProbe(tokens=(16 // 2) * (size // 16) ** 2, classes=classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = clips.permute(0, 1, 4, 2, 3).float() / 255.0
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        with torch.no_grad():
            h = self.enc.embeddings(x)
            for layer in self.enc.layer[:self.split]:
                h = layer(h)[0]
        for layer in self.enc.layer[self.split:]:
            h = layer(h)[0]
        return self.probe(self.enc.layernorm(h))


def trained_state(model: Finetuned) -> dict:
    """The weights that change: the probe, the final norm and the fine-tuned layers (copied to the CPU)."""
    def trained(k: str) -> bool:
        if k.startswith("enc.layer."):
            return int(k.split(".")[2]) >= model.split
        return not k.startswith("enc.") or k.startswith("enc.layernorm")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if trained(k)}


@torch.inference_mode()
def predict(model, items: list[tuple], args, batch: int, cache: Path = CACHE, hands_dir: Path = HANDS) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(Clips(items, args.crop, False, args.size, args.context, cache, hands_dir),
                                         batch_size=batch, num_workers=args.workers)
    out = []
    for clips, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(clips.cuda()).float(), 1).cpu().numpy())
    return np.concatenate(out)


# What ``train`` needs from a dataset: segment items per split (``items(split, fracs) -> (items, keys)``, items
# as (video, start, end, 1-based label, frac)), the frame cache, the hand landmarks folder (hand crop), the class
# count and the output folder.
INDUSTREAL = SimpleNamespace(items=items_for, cache=CACHE, hands=HANDS, classes=74, out=OUT)


def train(args, ds: SimpleNamespace = INDUSTREAL) -> None:
    download()
    name = (f"vjft_{args.crop}_k{args.layers}" + (f"_r{args.size}" if args.size != SIZE else "")
            + (f"_c{round(args.context * 100)}" if args.context else "")
            + (f"_e{args.epochs}" if args.epochs != 10 else "") + (f"_s{args.seed}" if args.seed else "")
            + (f"_p{round(args.prefix * 100)}" if args.prefix else ""))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Finetuned(args.layers, args.size, ds.classes).cuda()
    tr_items, _ = ds.items("train", (0.5,))
    val_items, _ = ds.items("val", (0.5,))
    val_labels = np.array([it[3] for it in val_items])
    loader = torch.utils.data.DataLoader(Clips(tr_items, args.crop, True, args.size, args.context, ds.cache, ds.hands, args.prefix),
                                         batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True, worker_init_fn=_seed_worker,
                                         persistent_workers=True)
    groups = [{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": args.lr},
              {"params": model.probe.parameters(), "lr": args.lr * 20}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = args.epochs * (len(loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    log, best = [], {"val_top1": -1.0}
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        for i, (clips, y) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(clips.cuda()), y.cuda(), label_smoothing=0.1) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        top1 = float((predict(model, val_items, args, args.batch * 2, ds.cache, ds.hands).argmax(1) + 1
                      == val_labels).mean() * 100)
        log.append({"epoch": epoch + 1, "loss": round(loss.item() * args.accum, 3), "val_top1": round(top1, 2),
                    "minutes": round((time.time() - start) / 60, 1),
                    "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)})
        print(json.dumps(log[-1]), flush=True)
        if top1 > best["val_top1"]:
            best = {"epoch": epoch + 1, "val_top1": round(top1, 2), "state": trained_state(model)}
    model.load_state_dict(best.pop("state"), strict=False)
    torch.save(trained_state(model), ds.out / f"{name}.pt")
    arrays, report = {}, {"run": name, "chosen_epoch": best["epoch"], "args": vars(args), "log": log}
    for split in ("val", "test"):
        items, keys = ds.items(split, TTA_FRACS)
        p = predict(model, items, args, args.batch * 2, ds.cache, ds.hands)
        p = p.reshape(-1, len(TTA_FRACS), p.shape[1]).mean(1)
        labels = np.array([it[3] for it in items[::len(TTA_FRACS)]])
        arrays.update({f"keys/{split}": np.array(keys), f"labels/{split}": labels, f"{name}/{split}": p})
        report[f"{split}_top1"] = round(float((p.argmax(1) + 1 == labels).mean() * 100), 2)
    np.savez(ds.out / f"clip_probs_{name}.npz", **arrays)
    (ds.out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Build the 960 x 540 frame cache (8 processes).")
    parser.add_argument("--crop", choices=("area", "hand"), default="area")
    parser.add_argument("--layers", type=int, default=8, help="Fine-tuned last transformer layers (of 24).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5, help="Encoder layers; the new probe gets 20x.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=SIZE, help="Crop side in pixels (multiple of 16).")
    parser.add_argument("--context", type=float, default=0.0,
                        help="Also sample this fraction of the segment's length before and after it.")
    parser.add_argument("--prefix", type=float, default=0.0,
                        help="Train on action beginnings: clip end drawn from this fraction to 100 %% of the action.")
    args = parser.parse_args()
    if args.cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(8) as pool:
            list(pool.map(build_cache, [r for rids in SPLITS.values() for r in rids]))
        return
    train(args)


if __name__ == "__main__":
    main()
