"""Fine-tune DINOv2-small on hands crops (object + verb heads), 2-fold cross-fitted.

``--fold k`` trains on half k of the training recordings (the test recording is
never used) and then extracts features for every recording with that backbone.
The temporal model later uses, for each training recording, features from the
backbone that did NOT train on it.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monitor.settings import DINO_V2_DIR, ROOT
from scripts.segmentation.evaluate_ar_object_segments import gt_object_frame_labels
from scripts.data.ar_labels import load_ar_intervals

FPS = 10.0
INPUT_HW = (252, 378)
VERBS = ["", "align", "browse", "check", "fit", "loosen", "plug", "pull", "put", "take", "tighten"]
# Fine-tuning may only draw from these participants (official IndustReal test split, assy videos);
# other labelled recordings on disk are held-out validation/test videos.
POOL_SUBJECTS = frozenset({"03", "08", "09", "10", "12", "13", "17", "18", "19", "23"})
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def recordings() -> list[str]:
    return sorted(
        p.name for p in (ROOT / "data/evaluation/official").iterdir()
        if (p / "AR_labels.csv").is_file()
    )


FULL_NOUNS = False  # set by --full-nouns: official 18-noun vocabulary


def fold_split(tests: list[str], by_subject: bool = False, pool=POOL_SUBJECTS) -> list[list[str]]:
    train = [r for r in recordings() if r not in tests and r.split("_", 1)[0] in pool]
    if not by_subject:
        return [train[0::2], train[1::2]]
    subjects = sorted({r.split("_", 1)[0] for r in train})
    halves = [set(subjects[0::2]), set(subjects[1::2])]
    return [[r for r in train if r.split("_", 1)[0] in h] for h in halves]


def frame_labels(rid: str, n: int) -> tuple[list[str], list[str]]:
    intervals = load_ar_intervals(ROOT / "data/evaluation/official" / rid / "AR_labels.csv",
                                  source_fps=FPS)
    objects, _ = gt_object_frame_labels(intervals, n_frames=n, fps=FPS, full_nouns=FULL_NOUNS)
    verbs = [""] * n
    for row in intervals:
        verb = str(row["action"]).split("_", 1)[0]
        if verb not in VERBS:
            continue
        for i in range(max(0, int(round(row["start_s"] * FPS))), min(n, int(round(row["end_s"] * FPS)) + 1)):
            if not verbs[i]:
                verbs[i] = verb
    return objects, verbs


def _to_tensor(img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


class CropDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int, int]], train: bool):
        self.items = items
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, obj, verb = self.items[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        if self.train:
            scale = np.random.uniform(0.75, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            y = np.random.randint(0, h - ch + 1)
            x = np.random.randint(0, w - cw + 1)
            img = img[y:y + ch, x:x + cw]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2)
                          + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        img = cv2.resize(np.ascontiguousarray(img), (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(_to_tensor(img)), obj, verb


class Net(nn.Module):
    def __init__(self, n_obj: int, n_verb: int, freeze_blocks: int = 4):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True)
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for block in self.backbone.encoder.layer[:freeze_blocks]:
            for p in block.parameters():
                p.requires_grad = False
        self.drop = nn.Dropout(0.2)
        self.obj = nn.Linear(768, n_obj)
        self.verb = nn.Linear(768, n_verb)

    def features(self, x):
        hidden = self.backbone(pixel_values=x).last_hidden_state
        return hidden, torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)

    def forward(self, x):
        _, feat = self.features(x)
        feat = self.drop(feat)
        return self.obj(feat), self.verb(feat)


def build_items(rids, crop_dir: Path, objects: list[str], bg_ratio: float, rng) -> list:
    items = []
    for rid in rids:
        n = int((crop_dir / rid / "done.txt").read_text())
        obj, verb = frame_labels(rid, n)
        for i in range(n):
            if not obj[i] and rng.random() > bg_ratio:
                continue
            items.append((crop_dir / rid / f"{i:05d}.jpg", objects.index(obj[i]), VERBS.index(verb[i])))
    return items


def train(args, objects: list[str], train_ids: list[str], val_ids: list[str], out: Path, device: str) -> Net:
    rng = np.random.default_rng(args.fold)
    torch.manual_seed(args.fold)
    items = build_items(train_ids, args.crops, objects, args.bg_ratio, rng)
    val_items = build_items(val_ids, args.crops, objects, 0.0, rng)
    val_items = [val_items[i] for i in rng.choice(len(val_items), min(3000, len(val_items)), replace=False)]
    counts = np.bincount([o for _, o, _ in items], minlength=len(objects)).astype(np.float64) + 1
    sample_w = (1.0 / np.sqrt(counts))[[o for _, o, _ in items]]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=args.samples_per_epoch, replacement=True)
    loader = DataLoader(CropDataset(items, True), batch_size=args.batch, sampler=sampler,
                        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(CropDataset(val_items, False), batch_size=args.batch * 2, num_workers=0)

    model = Net(len(objects), len(VERBS)).to(device)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.obj.parameters()) + list(model.verb.parameters())
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 20}], weight_decay=0.05)
    steps = args.epochs * (args.samples_per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 20], total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    print(json.dumps({"fold": args.fold, "train": train_ids, "val": val_ids,
                      "train_items": len(items), "val_items": len(val_items)}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        total = 0.0
        for step, (x, yo, yv) in enumerate(loader):
            x, yo, yv = x.to(device), yo.to(device), yv.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                lo, lv = model(x)
                loss = F.cross_entropy(lo, yo, label_smoothing=0.1) + 0.5 * F.cross_entropy(lv, yv, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        model.eval()
        ok = tot = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for x, yo, _ in val_loader:
                lo, _ = model(x.to(device))
                pred = lo[:, 1:].argmax(dim=1).cpu() + 1
                ok += int((pred == yo).sum())
                tot += len(yo)
        torch.save(model.state_dict(), out / f"fold{args.fold}.pt")
        print(json.dumps({"fold": args.fold, "epoch": epoch + 1, "loss": round(total / (step + 1), 4),
                          "val_frame_acc": round(ok / max(1, tot), 4),
                          "seconds": round(time.time() - start, 1),
                          "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)
    return model


@torch.inference_mode()
def extract(model: Net, args, out_dir: Path, device: str, only: list[str] | None = None) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for rid in recordings():
        if only is not None and rid not in only:
            continue
        target = out_dir / f"{rid}.npz"
        if target.is_file():
            continue
        n = int((args.crops / rid / "done.txt").read_text())
        items = [(args.crops / rid / f"{i:05d}.jpg", 0, 0) for i in range(n)]
        loader = DataLoader(CropDataset(items, False), batch_size=32, num_workers=min(1, args.workers))
        parts = [embed_batch(model, x.to(device)) for x, _, _ in loader]
        np.savez(target, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        print(json.dumps({"extracted": rid, "fold": args.fold, "frames": n}), flush=True)


@torch.inference_mode()
def embed_batch(model: Net, x: torch.Tensor) -> dict[str, np.ndarray]:
    """TCN input from the fine-tuned backbone: CLS + mean + 2x2 patch grid, and both head logits."""
    with torch.autocast("cuda", dtype=torch.float16):
        hidden, feat = model.features(x)
        gh, gw = INPUT_HW[0] // 14, INPUT_HW[1] // 14
        patches = hidden[:, 1:].reshape(hidden.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
        grid = F.adaptive_avg_pool2d(patches.float(), 2).flatten(2).permute(0, 2, 1).flatten(1)
        return {"feats": torch.cat([feat.float(), grid], dim=1).cpu().numpy().astype(np.float16)[:, None, :],
                "obj_logits": model.obj(feat).float().cpu().numpy(),
                "verb_logits": model.verb(feat).float().cpu().numpy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--test-recording", default="13_assy_0_1")
    parser.add_argument("--crops", type=Path, default=ROOT / "output/ar_hand_crops")
    parser.add_argument("--output", type=Path, default=ROOT / "output/ar_finetune")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--bg-ratio", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--test-recordings", nargs="*", default=None,
                        help="Recordings excluded from fine-tuning (overrides --test-recording).")
    parser.add_argument("--subject-split", action="store_true",
                        help="Split the two cross-fit folds by subject, not by recording.")
    parser.add_argument("--extract-needed", action="store_true",
                        help="Only extract features for the other fold + test recordings.")
    parser.add_argument("--pool-subjects", nargs="*", default=None,
                        help="Participants fine-tuning may use (default: official test-split participants).")
    parser.add_argument("--full-nouns", action="store_true", help="Official 18-noun vocabulary.")
    args = parser.parse_args()
    global FULL_NOUNS
    FULL_NOUNS = args.full_nouns

    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(4)
    device = "cuda"
    tests = list(args.test_recordings or [args.test_recording])
    folds = fold_split(tests, by_subject=args.subject_split,
                       pool=frozenset(args.pool_subjects) if args.pool_subjects else POOL_SUBJECTS)
    train_ids, val_ids = folds[args.fold], folds[1 - args.fold]
    assert not set(tests) & set(train_ids + val_ids)
    objects = [""] + sorted({o for r in train_ids + val_ids
                             for o in frame_labels(r, int((args.crops / r / "done.txt").read_text()))[0] if o})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "meta.json").write_text(json.dumps({"objects": objects, "verbs": VERBS, "folds": folds,
                                                       "tests": tests, "full_nouns": FULL_NOUNS}),
                                           encoding="utf-8")
    if args.extract_only:
        model = Net(len(objects), len(VERBS)).to(device)
        model.load_state_dict(torch.load(args.output / f"fold{args.fold}.pt", map_location=device))
    else:
        model = train(args, objects, train_ids, val_ids, args.output, device)
        (args.output / f"fold{args.fold}.trained").write_text("ok", encoding="utf-8")
    torch.cuda.empty_cache()
    if not args.no_extract:
        only = (val_ids + tests) if args.extract_needed else None
        extract(model, args, args.output / f"feats_fold{args.fold}", device, only=only)
        (args.output / f"fold{args.fold}.extracted").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
