"""Fine-tune VideoMAE-SSv2 on IndustReal action clips (74 verb_noun actions + background), cross-fitted.

Backbone k trains on the training recordings of crossfit fold k (``finetune/meta.json``, the same subject
halves as the fine-tuned DINOv2) and picks its epoch on the other fold's clips. It then writes per-frame
features (``feats`` (T, 1, 768), ``video_logits`` (T, 75)) for the other fold and for every val/test
recording, so no recording ever gets features from a backbone that trained on it.

Frames are the hands/work-area crop of ``extract_videomae_features`` at 224 x 224, cached once as JPEG
(``--cache``); training, feature extraction and clip scoring all read the same cached frames.

``--score-clips`` reports the IndustReal action-recognition protocol on the test split: each annotated
segment is classified from 16 frames sampled inside its ground-truth start/end (background excluded),
averaging both backbones.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from monitor.settings import ROOT
from scripts.features.extract_videomae_features import _MEAN, _STD, _prep, download, MODEL_DIR, CLIP
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL
from scripts.data.tas_labels import load_mapping, read_segments

VIDEOS = ROOT / "data/generated/industreal/videos"
OUT = ROOT / "output/official_split_eval/vmae_finetune"
META = ROOT / "output/official_split_eval/finetune/meta.json"
CACHE = ROOT / "output/vmae_frames"


def build_cache(rid: str, video: Path | None = None, cache: Path = CACHE) -> None:
    """All frames of one video as the 224 x 224 crop, JPEG q92: <rid>.bin + <rid>.idx.npy (byte offsets)."""
    target = cache / f"{rid}.idx.npy"
    if target.is_file():
        return
    cap = cv2.VideoCapture(str(video or VIDEOS / f"{rid}.mp4"))
    offsets = [0]
    with (cache / f"{rid}.bin").open("wb") as handle:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            data = cv2.imencode(".jpg", cv2.cvtColor(_prep(frame), cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            handle.write(data)
            offsets.append(offsets[-1] + len(data))
    cap.release()
    np.save(target, np.asarray(offsets, np.int64))


class Frames:
    """Random access to one video's cached frames (RGB uint8 224 x 224)."""

    def __init__(self, rid: str, cache: Path = CACHE):
        self.path = cache / f"{rid}.bin"
        self.offsets = np.load(cache / f"{rid}.idx.npy")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, indices) -> np.ndarray:
        out = []
        with self.path.open("rb") as handle:
            for i in indices:
                handle.seek(int(self.offsets[i]))
                raw = handle.read(int(self.offsets[i + 1] - self.offsets[i]))
                out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                                        cv2.COLOR_BGR2RGB))
        return np.stack(out)


def clips_for(rids: list[str], actions: list[str], background_ratio: float, seed: int):
    """(rid, start, end, label) per annotated segment, plus background windows from the gaps."""
    rng = random.Random(seed)
    index = {a: i for i, a in enumerate(actions)}
    clips = []
    for rid in rids:
        n = len(Frames(rid))
        segs = [(a, max(0, s), min(n - 1, e)) for a, s, e in read_segments(rid) if min(n - 1, e) >= max(0, s)]
        clips += [(rid, s, e, index[a]) for a, s, e in segs]
        busy = np.zeros(n, bool)
        for _, s, e in segs:
            busy[s:e + 1] = True
        free = [i for i in range(0, n - CLIP) if not busy[i:i + CLIP].any()]
        for start in rng.sample(free, min(len(free), int(background_ratio * len(segs)))):
            clips.append((rid, start, start + CLIP - 1, 0))
    return clips


def segment_clips(rids: list[str], actions: list[str], cache: Path = CACHE) -> tuple[list, list[str]]:
    """Annotated segments as clips (clamped to the video) and their keys ``rid|start|end`` (as annotated)."""
    index = {a: i for i, a in enumerate(actions)}
    clips, keys = [], []
    for rid in rids:
        n = len(Frames(rid, cache))
        for action, start, end in read_segments(rid):
            if min(n - 1, end) >= max(0, start):
                clips.append((rid, max(0, start), min(n - 1, end), index[action]))
                keys.append(f"{rid}|{start}|{end}")
    return clips, keys


def sample_indices(start: int, end: int, train: bool, frac: float = 0.5) -> list[int]:
    """CLIP frame indices spread uniformly over [start, end]: jittered inside each bin when training,
    else at ``frac`` of each bin."""
    length = end - start + 1
    edges = np.linspace(0, length, CLIP + 1)
    if train:
        pos = [np.random.uniform(edges[i], max(edges[i], edges[i + 1] - 1e-6)) for i in range(CLIP)]
    else:
        pos = edges[:-1] + frac * (edges[1:] - edges[:-1])
    return [start + min(length - 1, int(p)) for p in pos]


def load_clip(frames: dict, rid: str, start: int, end: int, train: bool, frac: float = 0.5,
              cache: Path = CACHE) -> np.ndarray:
    if rid not in frames:
        frames[rid] = Frames(rid, cache)
    clip = frames[rid].get(sample_indices(start, end, train, frac))
    if train:
        scale = np.random.uniform(0.8, 1.0)
        side = int(224 * scale)
        y, x = np.random.randint(0, 224 - side + 1, size=2)
        clip = np.stack([cv2.resize(f[y:y + side, x:x + side], (224, 224), interpolation=cv2.INTER_LINEAR)
                         for f in clip])
    return clip


def to_pixels(batch: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(batch).to("cuda").float() / 255.0  # (B, T, H, W, C)
    x = (x - torch.tensor(_MEAN, device="cuda")) / torch.tensor(_STD, device="cuda")
    return x.permute(0, 1, 4, 2, 3).contiguous()


def load_model(nclasses: int, checkpoint: Path | None = None):
    download()
    from transformers import VideoMAEForVideoClassification

    model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR, num_labels=nclasses,
                                                           ignore_mismatched_sizes=True)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to("cuda")


@torch.no_grad()
def clip_logits(model, clips, batch: int, frac: float = 0.5, cache: Path = CACHE) -> np.ndarray:
    model.eval()
    frames, out = {}, []
    for i in range(0, len(clips), batch):
        x = np.stack([load_clip(frames, r, s, e, False, frac, cache) for r, s, e, _ in clips[i:i + batch]])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(pixel_values=to_pixels(x)).logits.float().cpu().numpy())
    return np.concatenate(out)


def top1_actions(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 over the 74 actions only (background logit excluded), action clips only."""
    keep = labels > 0
    return float((logits[keep, 1:].argmax(1) + 1 == labels[keep]).mean() * 100)


def train_fold(fold: int, args, actions: list[str]) -> None:
    folds = json.loads(META.read_text(encoding="utf-8"))["folds"]
    train_model(OUT / f"fold{fold}", folds[fold], folds[1 - fold], fold, args, actions)


def train_model(out: Path, train_rids: list[str], held_rids: list[str], seed: int, args,
                actions: list[str]) -> None:
    """Fine-tune on ``train_rids``; keep the epoch with the best action top-1 on ``held_rids``' clips."""
    if (out / "best.pt").is_file() and (out / "done.json").is_file():
        return
    train_clips = clips_for(train_rids, actions, args.background_ratio, seed=seed)
    held_clips = clips_for(held_rids, actions, 0.0, seed=0) if held_rids else []
    train_on_clips(out, train_clips, held_clips, seed, args, len(actions))


def train_on_clips(out: Path, train_clips: list, held_clips: list, seed: int, args, nclasses: int,
                   cache: Path = CACHE) -> None:
    """Fine-tune on ``(rid, start, end, label)`` clips of ``cache``; class 0 is background. With ``held_clips`` the
    epoch with the best held-out action top-1 is kept, otherwise the last epoch."""
    fold = seed
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").is_file() and (out / "done.json").is_file():
        return
    held_labels = np.array([c[3] for c in held_clips])

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = load_model(nclasses)
    model.gradient_checkpointing_enable()
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "classifier" not in n]
    other = [p for n, p in model.named_parameters() if p.ndim <= 1 and "classifier" not in n]
    head = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05, "lr": args.lr},
                                   {"params": other, "weight_decay": 0.0, "lr": args.lr},
                                   {"params": head, "weight_decay": 0.0, "lr": args.lr * 10}])
    scaler = torch.cuda.amp.GradScaler()
    steps_per_epoch = math.ceil(len(train_clips) / (args.batch * args.accum))
    total = steps_per_epoch * args.epochs
    base = [g["lr"] for g in optimizer.param_groups]
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    state_path = out / "last.pt"
    epoch0, history, best = 0, [], -1.0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        epoch0, history, best = state["epoch"], state["history"], state["best"]
    print(json.dumps({"fold": fold, "train_clips": len(train_clips), "held_clips": len(held_clips),
                      "resume_epoch": epoch0}), flush=True)

    frames = {}
    for epoch in range(epoch0, args.epochs):
        model.train()
        order = list(range(len(train_clips)))
        random.Random(1000 * fold + epoch).shuffle(order)
        started, seen, running = time.time(), 0, 0.0
        for step in range(steps_per_epoch):
            g = epoch * steps_per_epoch + step
            warm = min(1.0, (g + 1) / max(1, steps_per_epoch // 2))
            for group, lr in zip(optimizer.param_groups, base):
                group["lr"] = lr * warm * 0.5 * (1 + math.cos(math.pi * g / total))
            for micro in range(args.accum):
                lo = (step * args.accum + micro) * args.batch
                idx = order[lo:lo + args.batch]
                if not idx:
                    break
                x = np.stack([load_clip(frames, *train_clips[i][:3], True, cache=cache) for i in idx])
                y = torch.tensor([train_clips[i][3] for i in idx], device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = criterion(model(pixel_values=to_pixels(x)).logits.float(), y) / args.accum
                scaler.scale(loss).backward()
                running += loss.item() * args.accum
                seen += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(json.dumps({"fold": fold, "epoch": epoch, "step": step, "of": steps_per_epoch,
                                  "loss": round(running / max(1, seen), 3),
                                  "min": round((time.time() - started) / 60, 1)}), flush=True)
        # Without held-out clips (refit on train + val) the last epoch is kept.
        acc = (top1_actions(clip_logits(model, held_clips, args.batch * 2, cache=cache), held_labels) if held_clips
               else float(epoch))
        history.append({"epoch": epoch, "train_loss": round(running / max(1, seen), 3),
                        "heldout_fold_top1": round(acc, 2)})
        print(json.dumps(history[-1]), flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch + 1, "history": history, "best": best}, out / "last.tmp")
        (out / "last.tmp").replace(state_path)
    (out / "done.json").write_text(json.dumps({"history": history, "best_heldout_fold_top1": best},
                                              indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)


@torch.no_grad()
def extract(model, rid: str, stride: int, batch: int, cache: Path = CACHE,
            causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Same clip layout as extract_videomae_features: 16 frames centred every ``stride`` frames, or with
    ``causal`` ending there (a frame then takes the latest clip ending at or before it)."""
    model.eval()
    frames = Frames(rid, cache)
    n = len(frames)
    centres = list(range(0, n, stride))
    back = CLIP - 1 if causal else CLIP // 2
    pooled, logits = [], []
    for i in range(0, len(centres), batch):
        clips = [frames.get([min(max(j, 0), n - 1) for j in range(c - back, c - back + CLIP)])
                 for c in centres[i:i + batch]]
        with torch.autocast("cuda", dtype=torch.float16):
            seq = model.videomae(pixel_values=to_pixels(np.stack(clips))).last_hidden_state
            p = model.fc_norm(seq.mean(dim=1))
            lg = model.classifier(p)
        pooled.append(p.float().cpu().numpy())
        logits.append(lg.float().cpu().numpy())
    pooled_c, logits_c = np.concatenate(pooled), np.concatenate(logits)
    nearest = np.arange(n) // stride if causal else (np.arange(n) + stride // 2) // stride
    nearest = np.minimum(nearest, len(pooled_c) - 1)
    return pooled_c[nearest].astype(np.float16), logits_c[nearest].astype(np.float16)


def extract_fold(fold: int, args, actions: list[str], causal: bool = False) -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    extract_to(OUT / f"feats_fold{fold}{'_causal' if causal else ''}", OUT / f"fold{fold}" / "best.pt",
               meta["folds"][1 - fold] + meta["tests"], fold, args, actions, causal)


META4 = OUT / "meta4.json"
FOLDS4 = 4


def meta4() -> dict:
    """4 groups of training subjects (every 4th subject in sorted order); backbone k never sees heldout[k]."""
    if not META4.is_file():
        subjects = sorted({r.split("_", 1)[0] for r in TRAIN})
        heldout = [[r for r in TRAIN if r.split("_", 1)[0] in subjects[k::FOLDS4]] for k in range(FOLDS4)]
        META4.write_text(json.dumps({"heldout": heldout, "tests": VAL + TEST}, indent=2), encoding="utf-8")
    return json.loads(META4.read_text(encoding="utf-8"))


def train_fold4(k: int, args, actions: list[str]) -> None:
    """Train on 3 of the 4 subject groups (epoch chosen on group k), then write features for group k + val/test."""
    held = meta4()["heldout"][k]
    train_model(OUT / f"fold4_{k}", [r for r in TRAIN if r not in held], held, 10 + k, args, actions)
    extract_to(OUT / f"feats4_fold{k}", OUT / f"fold4_{k}" / "best.pt", held + VAL + TEST, f"4_{k}", args, actions)


def extract_to(out: Path, checkpoint: Path, rids: list[str], fold, args, actions: list[str],
               causal: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rids if not (out / f"{r}.npz").is_file()]
    if not todo:
        return
    model = load_model(len(actions), checkpoint)
    for rid in todo:
        start = time.time()
        feats, logits = extract(model, rid, args.stride, args.batch * 2, causal=causal)
        tmp = out / f"{rid}.tmp.npz"
        np.savez(tmp, feats=feats[:, None, :], video_logits=logits)
        tmp.replace(out / f"{rid}.npz")
        print(json.dumps({"fold": fold, "recording": rid, "frames": int(len(feats)),
                          "seconds": round(time.time() - start, 1)}), flush=True)
    del model
    torch.cuda.empty_cache()


TTA_FRACS = (0.5, 0.25, 0.75)
WINDOW_FRAMES, MAX_WINDOWS = 48, 4


def _softmax_actions(logits: np.ndarray) -> np.ndarray:
    lg = logits[:, 1:]
    p = np.exp(lg - lg.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def tta_probs(model, clips, batch: int, fracs=TTA_FRACS, cache: Path = CACHE) -> np.ndarray:
    """Per clip: action probabilities (background excluded), mean over the temporal samplings ``fracs``."""
    return sum(_softmax_actions(clip_logits(model, clips, batch, f, cache)) for f in fracs) / len(fracs)


def window_probs(model, clips, batch: int, cache: Path = CACHE) -> np.ndarray:
    """Per clip: split into up to MAX_WINDOWS equal parts (one per WINDOW_FRAMES frames), 16 frames from each,
    action probabilities averaged over the parts."""
    parts, owner = [], []
    for i, (rid, s, e, label) in enumerate(clips):
        k = int(np.clip((e - s + 1) // WINDOW_FRAMES, 1, MAX_WINDOWS))
        edges = np.linspace(s, e + 1, k + 1).astype(int)
        parts += [(rid, int(a), int(b) - 1, label) for a, b in zip(edges[:-1], edges[1:])]
        owner += [i] * k
    p = _softmax_actions(clip_logits(model, parts, batch, 0.5, cache))
    total = np.zeros((len(clips), p.shape[1]))
    np.add.at(total, np.array(owner), p)
    return total / np.bincount(owner)[:, None]


def score_clips(args, actions: list[str]) -> None:
    """Per-segment action probabilities (background excluded, mean over TTA_FRACS samplings) for the val
    and test segments, from every trained backbone -> clip_probs.npz + clip_recognition.json."""
    splits = {"val": segment_clips(VAL, actions), "test": segment_clips(TEST, actions)}
    arrays = {}
    for split, (clips, keys) in splits.items():
        arrays[f"labels/{split}"] = np.array([c[3] for c in clips])
        arrays[f"keys/{split}"] = np.array(keys)
    names = sorted(p.parent.name for p in OUT.glob("*/done.json"))
    old = np.load(OUT / "clip_probs.npz") if (OUT / "clip_probs.npz").is_file() else None
    for name in names:
        if old is not None and all(f"{name}/{s}" in old.files for s in splits):
            for split in splits:
                arrays[f"{name}/{split}"] = old[f"{name}/{split}"]
            continue
        model = load_model(len(actions), OUT / name / "best.pt")
        for split, (clips, _) in splits.items():
            arrays[f"{name}/{split}"] = tta_probs(model, clips, args.batch * 2).astype(np.float32)
        del model
        torch.cuda.empty_cache()
    np.savez(OUT / "clip_probs.npz", **arrays)

    def top1(probs, split):
        return round(float((probs.argmax(1) + 1 == arrays[f"labels/{split}"]).mean() * 100), 2)

    report = {"protocol": "IndustReal action recognition: 16 frames inside each GT segment, 74 actions "
                          f"(background excluded), mean over {len(TTA_FRACS)} temporal samplings",
              "segments": {s: int(len(c)) for s, (c, _) in splits.items()}}
    report["note"] = "refit_trainval trained on the val videos: its val numbers are not held-out"
    fulls = [n for n in names if n.startswith("full")]
    for split in splits:
        report[split] = {name: top1(arrays[f"{name}/{split}"], split) for name in names}
        if {"fold0", "fold1"} <= set(names):
            report[split]["fold0+fold1"] = top1(arrays[f"fold0/{split}"] + arrays[f"fold1/{split}"], split)
        if len(fulls) > 1:
            report[split]["+".join(fulls)] = top1(sum(arrays[f"{n}/{split}"] for n in fulls), split)
    (OUT / "clip_recognition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def score_windows(args, actions: list[str]) -> None:
    """Long segments split into up to MAX_WINDOWS equal parts (one per WINDOW_FRAMES frames), 16 frames from
    each part, action probabilities averaged over parts -> clip_probs_windows.npz (``full*`` / refit models)."""
    path = OUT / "clip_probs_windows.npz"
    arrays = dict(np.load(path)) if path.is_file() else {}
    names = sorted(p.parent.name for p in OUT.glob("*/done.json") if not p.parent.name.startswith("fold"))
    splits = {"val": segment_clips(VAL, actions), "test": segment_clips(TEST, actions)}
    for name in names:
        if all(f"{name}/{s}" in arrays for s in splits):
            continue
        model = load_model(len(actions), OUT / name / "best.pt")
        for split, (clips, keys) in splits.items():
            arrays[f"{name}/{split}"] = window_probs(model, clips, args.batch * 2).astype(np.float32)
            arrays[f"keys/{split}"] = np.array(keys)
            print(json.dumps({"windows": name, "split": split, "segments": len(clips)}), flush=True)
        del model
        torch.cuda.empty_cache()
        np.savez(path, **arrays)


FRAME_MODELS = ("full", "full_s3")


def score_frames(args, actions: list[str]) -> None:
    """Dense per-frame logits (T, 75) of the ``full*`` backbones on every val/test recording (16-frame clips
    centred every ``--stride`` frames) -> frame_logits/<model>/<rid>.npy, for segmentation fusion."""
    for name in FRAME_MODELS:
        out = OUT / "frame_logits" / name
        out.mkdir(parents=True, exist_ok=True)
        todo = [r for r in VAL + TEST if not (out / f"{r}.npy").is_file()]
        if not todo:
            continue
        model = load_model(len(actions), OUT / name / "best.pt")
        for rid in todo:
            start = time.time()
            _, logits = extract(model, rid, args.stride, args.batch * 2)
            np.save(out / f"{rid}.tmp.npy", logits)
            (out / f"{rid}.tmp.npy").replace(out / f"{rid}.npy")
            print(json.dumps({"frames": name, "recording": rid, "n": int(len(logits)),
                              "seconds": round(time.time() - start, 1)}), flush=True)
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true", help="Build the JPEG frame cache for all recordings.")
    parser.add_argument("--fold", type=int, choices=(0, 1), help="Train (resumable) and extract one backbone.")
    parser.add_argument("--fold4", type=int, choices=range(FOLDS4),
                        help="4-fold cross-fit backbone k (meta4.json): train on 3 subject groups, extract.")
    parser.add_argument("--full", action="store_true",
                        help="Train one backbone on all training recordings (epoch chosen on val clips); "
                             "used for the clip-recognition protocol only, never for TCN features.")
    parser.add_argument("--seed", type=int, default=2, help="--full seed (2 -> full/, else full_s<seed>/).")
    parser.add_argument("--refit", action="store_true",
                        help="Train one backbone on train + val recordings for --epochs (no epoch selection).")
    parser.add_argument("--score-clips", action="store_true")
    parser.add_argument("--score-windows", action="store_true",
                        help="Multi-window clip probabilities for long segments (full* / refit models).")
    parser.add_argument("--causal-fold", type=int, choices=(0, 1),
                        help="Extract backbone k's features with clips ending at each frame -> feats_fold{k}_causal.")
    parser.add_argument("--score-frames", action="store_true",
                        help="Dense per-frame logits of the full* models on val/test (segmentation fusion).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--background-ratio", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    actions = load_mapping()

    if args.cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        for rid in TRAIN + VAL + TEST:
            build_cache(rid)
        print(json.dumps({"cache": str(CACHE), "recordings": len(TRAIN + VAL + TEST)}), flush=True)
    if args.fold is not None:
        train_fold(args.fold, args, actions)
        extract_fold(args.fold, args, actions)
    if args.causal_fold is not None:
        extract_fold(args.causal_fold, args, actions, causal=True)
    if args.fold4 is not None:
        train_fold4(args.fold4, args, actions)
    if args.full:
        train_model(OUT / ("full" if args.seed == 2 else f"full_s{args.seed}"), TRAIN, VAL, args.seed, args, actions)
    if args.refit:
        train_model(OUT / "refit_trainval", TRAIN + VAL, [], 7, args, actions)
    if args.score_clips:
        score_clips(args, actions)
    if args.score_windows:
        score_windows(args, actions)
    if args.score_frames:
        score_frames(args, actions)


if __name__ == "__main__":
    main()
