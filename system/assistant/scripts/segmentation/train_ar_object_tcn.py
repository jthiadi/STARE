"""Held-out AR-object recognition from DINOv2 frame features + temporal conv net.

Trains a small multi-stage TCN on every AR-labeled recording except the test one,
as a k-fold ensemble. Decoding (smoothing / Viterbi switch penalty) is tuned on
out-of-fold predictions of the training recordings only. The test recording's
GT is used solely for the final score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.kit_catalog import KIT_TO_AR_OBJECT
from monitor.settings import ROOT
from scripts.segmentation.evaluate_ar_object_segments import (
    evaluate_object_stream,
    frames_to_segments,
    gt_object_frame_labels,
)
from scripts.data.ar_labels import load_ar_intervals

FPS = 10.0
_AR_TO_KIT = {v: k for k, v in KIT_TO_AR_OBJECT.items()}
FULL_NOUNS = False  # set by --full-nouns: official 18-noun vocabulary instead of kit parts
TARGET = "noun"  # set by --target: which half of the verb_noun action label to predict


def _gt_labels(rid: str, n: int) -> tuple[list[str], list[tuple[str, float, float]]]:
    if TARGET == "verb":
        from scripts.features.finetune_dino_ar import frame_labels

        verbs = frame_labels(rid, n)[1]
        return verbs, frames_to_segments(verbs, FPS)
    return gt_object_frame_labels(
        load_ar_intervals(ROOT / "data/evaluation/official" / rid / "AR_labels.csv", source_fps=FPS),
        n_frames=n, fps=FPS, full_nouns=FULL_NOUNS,
    )


def _set_low_priority() -> None:
    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def progress_features(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    k = np.arange(1, 5, dtype=np.float32)[None]
    return np.concatenate([t, np.sin(np.pi * k * t), np.cos(np.pi * k * t)], axis=1)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _load_views(path: Path) -> np.ndarray:
    archive = np.load(path)
    feats = archive["feats"].astype(np.float32)
    n, v, d = feats.shape
    feats = feats.reshape(n, v, d // 384, 384)
    feats /= np.linalg.norm(feats, axis=3, keepdims=True) + 1e-6
    out = feats.reshape(n, -1) * np.sqrt(1.0 / (d // 384))
    if "obj_logits" in archive.files:
        out = np.concatenate([out, _softmax(archive["obj_logits"]).astype(np.float32),
                              _softmax(archive["verb_logits"]).astype(np.float32)], axis=1)
    if "video_logits" in archive.files:
        out = np.concatenate([out, _softmax(archive["video_logits"].astype(np.float32))], axis=1)
    return out


def load_recording(rid: str, feature_dir: Path, progress: bool = False,
                   extra_dirs: list[Path] | None = None) -> tuple[np.ndarray, list[str]]:
    feats = load_features(rid, feature_dir, progress, extra_dirs)
    gt, _ = _gt_labels(rid, feats.shape[0])
    return feats, gt


def load_features(rid: str, feature_dir: Path, progress: bool = False,
                  extra_dirs: list[Path] | None = None) -> np.ndarray:
    """(T, D) float16 TCN input: base views, then each extra feature set, then video progress."""
    feats = _load_views(feature_dir / f"{rid}.npz")
    for extra in extra_dirs or []:
        more = _load_views(extra / f"{rid}.npz")
        n = min(len(feats), len(more))
        feats = np.concatenate([feats[:n], more[:n]], axis=1)
    if progress:
        feats = np.concatenate([feats, progress_features(feats.shape[0])], axis=1)
    return feats.astype(np.float16)


class Stage(nn.Module):
    """Dilated residual stage; ``causal``: each output frame sees only that frame and earlier ones."""

    def __init__(self, dim_in: int, hidden: int, classes: int, layers: int, dropout: float, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.inp = nn.Conv1d(dim_in, hidden, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(hidden, hidden, 3, padding=0 if causal else 2 ** i, dilation=2 ** i) for i in range(layers)
        )
        self.mix = nn.ModuleList(nn.Conv1d(hidden, hidden, 1) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden, classes, 1)

    def forward(self, x):
        h = self.inp(x)
        for i, (conv, mix) in enumerate(zip(self.blocks, self.mix)):
            z = F.pad(h, (2 * 2 ** i, 0)) if self.causal else h
            h = h + self.drop(mix(F.relu(conv(z))))
        return self.out(h)


class TCN(nn.Module):
    """Multi-stage TCN; the last ``aux`` output channels are an auxiliary verb head."""

    def __init__(self, dim_in: int, classes: int, hidden: int = 128, layers: int = 9, stages: int = 3,
                 aux: int = 0, causal: bool = False):
        super().__init__()
        self.classes = classes
        width = classes + aux
        self.in_drop = nn.Dropout(0.5)
        self.first = Stage(dim_in, hidden, width, layers, 0.3, causal)
        self.rest = nn.ModuleList(Stage(width, hidden, width, layers, 0.3, causal) for _ in range(stages - 1))

    def _probs(self, logits):
        c = self.classes
        if logits.shape[1] == c:
            return F.softmax(logits, dim=1)
        return torch.cat([F.softmax(logits[:, :c], dim=1), F.softmax(logits[:, c:], dim=1)], dim=1)

    def forward(self, x):  # x: (B, D, T)
        outs = [self.first(self.in_drop(x))]
        for stage in self.rest:
            outs.append(stage(self._probs(outs[-1])))
        return outs


class BiGRU(nn.Module):
    """Bidirectional GRU over time, refined by one TCN stage."""

    def __init__(self, dim_in: int, classes: int, hidden: int = 128, layers: int = 9):
        super().__init__()
        self.in_drop = nn.Dropout(0.5)
        self.proj = nn.Linear(dim_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.head = nn.Linear(2 * hidden, classes)
        self.refine = Stage(classes, hidden, classes, layers, 0.3)

    def forward(self, x):  # x: (B, D, T)
        h = F.relu(self.proj(self.in_drop(x).transpose(1, 2)))
        first = self.head(self.gru(h)[0]).transpose(1, 2)
        return [first, self.refine(F.softmax(first, dim=1))]


def _loss(outs, y, weights, yv=None):
    total = 0.0
    c = weights.shape[0]
    for logits in outs:
        if yv is not None:
            total = total + 0.5 * F.cross_entropy(logits[:, c:], yv)
        logits = logits[:, :c]
        total = total + F.cross_entropy(logits, y, weight=weights, ignore_index=-100)
        logp = F.log_softmax(logits, dim=1)
        total = total + 0.15 * torch.clamp(
            F.mse_loss(logp[:, :, 1:], logp.detach()[:, :, :-1], reduction="none"), max=16
        ).mean()
    return total


def _time_warp(x, y, yv, rng):
    rate = float(rng.uniform(0.8, 1.2))
    length = x.shape[1]
    idx = torch.clamp((torch.arange(int(length / rate), device=x.device) * rate).long(), max=length - 1)
    return x[:, idx], y[idx], (yv[idx] if yv is not None else None)


def _fresh(x):
    """Read through a short-lived mapping: a long-lived one keeps every touched page in the
    working set, while unmapped pages go to the standby list (available RAM on Windows)."""
    return np.load(x.filename, mmap_mode="r") if isinstance(x, np.memmap) else x


def _memmap_window(x: np.memmap, y, yv, rng, warp: bool, device):
    """Same sampling as warp + crop below, but reads only the window rows of a (T, D) memmap."""
    length = x.shape[0]
    idx = np.arange(length)
    if warp:
        rate = float(rng.uniform(0.8, 1.2))
        idx = np.minimum((np.arange(int(length / rate)) * rate).astype(np.int64), length - 1)
    crop = int(rng.integers(600, 1500))
    s = int(rng.integers(0, max(1, len(idx) - crop)))
    sel = idx[s:s + crop]
    xb = torch.from_numpy(np.ascontiguousarray(_fresh(x)[sel])).to(device).float().T
    t = torch.from_numpy(sel).to(device)
    return xb, y[t], (yv[t][None] if yv is not None else None)


def train_model(train, classes, *, epochs, seed, device, val=None, cosine=False, warp=False,
                arch=(128, 9, 3), model_type="tcn", verbs=None):
    """``verbs``: optional per-recording verb index lists (same order as ``train``)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    index = {c: i for i, c in enumerate(classes)}
    first = train[0][0]
    dim = first.shape[0] if isinstance(first, torch.Tensor) else first.shape[1]
    n_aux = (max(max(v) for v in verbs) + 1) if verbs else 0
    if model_type == "gru":
        model = BiGRU(dim, len(classes), hidden=arch[0], layers=arch[1]).to(device)
    else:
        model = TCN(dim, len(classes), hidden=arch[0], layers=arch[1], stages=arch[2], aux=n_aux,
                    causal=model_type == "causal_tcn").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    counts = np.ones(len(classes))
    for _, gt in train:
        for g in gt:
            counts[index[g]] += 1
    weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device=device)
    weights /= weights.mean()
    data = [
        (f if isinstance(f, (torch.Tensor, np.memmap)) else _resident(f, device),
         torch.tensor([index[g] for g in gt], device=device),
         torch.tensor(verbs[k][:len(gt)], device=device) if verbs else None)
        for k, (f, gt) in enumerate(train)
    ]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    history = []
    for epoch in range(epochs):
        model.train()
        for i in rng.permutation(len(data)):
            x, y, yv = data[i]
            if isinstance(x, np.memmap):
                xb, yb, yvb = _memmap_window(x, y, yv, rng, warp, device)
            else:
                if warp:
                    x, y, yv = _time_warp(x, y, yv, rng)
                length = x.shape[1]
                crop = int(rng.integers(600, 1500))
                s = int(rng.integers(0, max(1, length - crop)))
                xb, yb = x[:, s:s + crop].to(device).float(), y[s:s + crop]
                yvb = yv[s:s + crop][None] if yv is not None else None
            xb = xb + 0.02 * torch.randn_like(xb)
            loss = _loss(model(xb[None]), yb[None], weights, yvb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        if val is not None and (epoch + 1) % 5 == 0:
            history.append((epoch + 1, _val_acc(model, val, classes, device)))
    return model, history


def _to_device(feats, device) -> torch.Tensor:
    """(T, D) host array or resident (D, T) fp16 tensor -> (D, T) float32 tensor on ``device``."""
    if isinstance(feats, torch.Tensor):
        return feats.to(device).float()
    return torch.from_numpy(np.ascontiguousarray(_fresh(feats))).to(device).float().T.contiguous()


def _resident(feats: np.ndarray, device) -> torch.Tensor:
    """Keep features on the GPU as (D, T) fp16 so host RAM stays small."""
    return torch.from_numpy(feats).to(device).T.contiguous()


@torch.inference_mode()
def predict_proba(model, feats: np.ndarray, device) -> np.ndarray:
    model.eval()
    x = _to_device(feats, device)[None]
    logits = model(x)[-1][0]
    c = getattr(model, "classes", logits.shape[0])
    return F.softmax(logits[:c], dim=0).T.cpu().numpy()


def _val_acc(model, val, classes, device) -> float:
    ok = tot = 0
    for f, gt in val:
        pred = decode(predict_proba(model, f, device), classes, switch=0.0)
        for p, g in zip(pred, gt):
            if g:
                tot += 1
                ok += int(p == g)
    return ok / max(1, tot)


def learn_transitions(gts: list[list[str]], classes: list[str]) -> np.ndarray:
    """Log P(next object | current object) from training label order (add-one smoothed)."""
    obj = [c for c in classes if c]
    index = {c: i for i, c in enumerate(obj)}
    counts = np.ones((len(obj), len(obj)))
    for gt in gts:
        seq = [g for g, _ in _runs(gt) if g]
        for a, b in zip(seq, seq[1:]):
            if a != b:
                counts[index[a], index[b]] += 1
    np.fill_diagonal(counts, 0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    trans = np.log(probs + 1e-9)
    np.fill_diagonal(trans, 0.0)
    return trans


def _runs(labels: list[str]):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            yield labels[start], i - start
            start = i


def decode(prob: np.ndarray, classes: list[str], *, switch: float,
           trans: np.ndarray | None = None, trans_weight: float = 0.0) -> list[str]:
    """Viterbi over object classes (background ignored): switch cost + learned order prior."""
    obj = [i for i, c in enumerate(classes) if c]
    logp = np.log(prob[:, obj] + 1e-6)
    if switch <= 0 and trans_weight <= 0:
        return [classes[obj[j]] for j in logp.argmax(axis=1)]
    t, k = logp.shape
    cost = np.full((k, k), -switch)
    if trans is not None and trans_weight > 0:
        cost = cost + trans_weight * trans
    np.fill_diagonal(cost, 0.0)
    score = logp[0].copy()
    back = np.zeros((t, k), dtype=np.int32)
    for i in range(1, t):
        cand = score[:, None] + cost
        back[i] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logp[i]
    path = np.zeros(t, dtype=np.int32)
    path[-1] = int(score.argmax())
    for i in range(t - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return [classes[obj[j]] for j in path]


def labeled_acc(pred: list[str], gt: list[str]) -> tuple[int, int]:
    ok = sum(1 for p, g in zip(pred, gt) if g and p == g)
    return ok, sum(1 for g in gt if g)


def subject_of(rid: str) -> str:
    return rid.split("_", 1)[0]


def make_folds(train_ids: list[str], n_folds: int, by_subject: bool) -> list[list[str]]:
    if not by_subject:
        return [train_ids[i::n_folds] for i in range(n_folds)]
    subjects = sorted({subject_of(r) for r in train_ids})
    groups = [subjects[i::n_folds] for i in range(n_folds)]
    return [[r for r in train_ids if subject_of(r) in g] for g in groups if g]


def _write_outputs(out: Path, rid: str, prob: np.ndarray, pred: list[str], classes: list[str],
                   bg_threshold: float) -> dict:
    gt, gt_segs = _gt_labels(rid, len(pred))
    active = prob[:, 0] < bg_threshold
    gated = [p if a else "" for p, a in zip(pred, active)]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_probs.npy", prob)
    with (out / "ar_object_timeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "frame_index", "timestamp_s", "detect_index", "kit_label", "ar_object",
            "score", "contact", "held",
        ])
        writer.writeheader()
        for i, ar in enumerate(gated):
            writer.writerow({
                "frame_index": i, "timestamp_s": round(i / FPS, 3), "detect_index": i,
                "kit_label": _AR_TO_KIT.get(ar, ar.replace("_", " ")) if ar else "",
                "ar_object": ar,
                "score": round(float(prob[i, classes.index(ar)]), 3) if ar else "",
                "contact": "", "held": False,
            })
    full = evaluate_object_stream(gt, pred, gt_segs, frames_to_segments(pred, FPS))
    gat = evaluate_object_stream(gt, gated, gt_segs, frames_to_segments(gated, FPS))
    ok, tot = labeled_acc(pred, gt)
    return {"always_predict": full, "background_gated": gat, "labeled_ok": ok, "labeled_total": tot}


def run(args: argparse.Namespace) -> None:
    global FULL_NOUNS, TARGET
    FULL_NOUNS = args.full_nouns
    TARGET = args.target
    assert not (args.verb_aux and TARGET == "verb"), "--verb-aux is only for the noun target"
    _set_low_priority()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dir = Path(args.features)
    extra = [Path(p) for p in args.extra_features]
    tests = list(args.test_recordings or [args.test_recording])
    rids = sorted(p.stem for p in feature_dir.glob("*.npz"))
    rids = [r for r in rids if all((e / f"{r}.npz").is_file() for e in extra)]
    train_ids = [r for r in rids if r not in tests]
    if args.train_recordings:
        missing = set(args.train_recordings) - set(rids)
        assert not missing, f"no features for training recordings {sorted(missing)}"
        train_ids = [r for r in rids if r in args.train_recordings]
    assert not set(train_ids) & set(tests), "a test recording is also in training"
    test_variants: dict[str, list[np.ndarray]] = {}
    data = {}
    # Many held-out recordings: load each from disk only while it is scored, so GPU memory
    # (which counts against the Windows commit limit) holds just the training set.
    lazy_tests = len(tests) > 4
    cache = Path(args.output) / "_feat_cache"

    def cached_train(r: str):
        path = cache / f"{r}.npy"
        if not (args.host_features and path.is_file()):
            return None
        feats = np.load(path, mmap_mode="r")
        return feats, _gt_labels(r, feats.shape[0])[0]

    def store_train(r: str, feats: np.ndarray):
        if not args.host_features:
            return _resident(feats, device)
        # File-backed pages do not count against the Windows commit limit.
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{r}.npy"
        if not path.is_file():
            np.save(path, feats)
        return np.load(path, mmap_mode="r")

    def load_test(r: str) -> list[np.ndarray]:
        dirs = [variant_dirs(i) for i in range(n_variants)]
        if not args.host_features:
            return [load_recording(r, feature_dir, progress=args.progress, extra_dirs=x)[0] for x in dirs]
        cache.mkdir(parents=True, exist_ok=True)
        out = []
        for i, x in enumerate(dirs):
            path = cache / f"{r}__v{i}.npy"
            if not path.is_file():
                np.save(path, load_recording(r, feature_dir, progress=args.progress, extra_dirs=x)[0])
            out.append(np.load(path, mmap_mode="r"))
        return out

    cf_pairs = [[Path(a), Path(b)] for a, b in args.crossfit] if args.crossfit else []
    # --crossfit-set META DIR_0 .. DIR_n-1: backbone k never trained on META's heldout[k] recordings.
    cf_sets = [(json.loads(Path(meta).read_text(encoding="utf-8"))["heldout"], [Path(d) for d in dirs])
               for meta, *dirs in (args.crossfit_set or [])]
    for heldout, dirs in cf_sets:
        assert len(heldout) == len(dirs), "one feature dir per held-out group"
        assert not set(tests) & {r for h in heldout for r in h}, "test recording used in fine-tuning"
    n_variants = max([2] * bool(cf_pairs) + [len(dirs) for _, dirs in cf_sets] + [1])

    def variant_dirs(i: int) -> list[Path]:
        """Test-time feature variant i: backbone i (mod the set's size) of every fine-tuned feature set."""
        return extra + [pair[i % 2] for pair in cf_pairs] + [dirs[i % len(dirs)] for _, dirs in cf_sets]

    if cf_pairs or cf_sets:
        # Backbone k trained on crossfit folds[k]; use the other backbone for those recordings.
        cf_folds = (json.loads(Path(args.crossfit_meta).read_text(encoding="utf-8"))["folds"] if cf_pairs
                    else [[], []])
        assert not set(tests) & set(cf_folds[0] + cf_folds[1]), "test recording used in fine-tuning"
        for r in [] if lazy_tests else tests:
            variants = []
            for i in range(n_variants):
                feats, gt = load_recording(r, feature_dir, progress=args.progress, extra_dirs=variant_dirs(i))
                variants.append(_resident(feats, device))
            data[r] = (variants[0], gt)
            test_variants[r] = variants
        for r in train_ids:
            data[r] = cached_train(r)
            if data[r] is None:
                other = 1 if r in cf_folds[0] else 0
                owned = [dirs[next(k for k, h in enumerate(heldout) if r in h)] for heldout, dirs in cf_sets]
                feats, gt = load_recording(r, feature_dir, progress=args.progress,
                                           extra_dirs=extra + [pair[other] for pair in cf_pairs] + owned)
                data[r] = (store_train(r, feats), gt)
                del feats
    else:
        for r in train_ids + ([] if lazy_tests else tests):
            feats, gt = load_recording(r, feature_dir, progress=args.progress, extra_dirs=extra)
            data[r] = (store_train(r, feats) if r in train_ids else _resident(feats, device), gt)
            if r in tests:
                test_variants[r] = [data[r][0]]
    verbs: dict[str, list[int]] = {}
    if args.verb_aux:
        from scripts.features.finetune_dino_ar import VERBS, frame_labels

        verbs = {r: [VERBS.index(v) for v in frame_labels(r, len(data[r][1]))[1]] for r in train_ids}
    classes = [""] + sorted({g for r in train_ids for g in data[r][1] if g})
    print(json.dumps({"train": train_ids, "tests": tests, "classes": classes, "device": device,
                      "progress": args.progress, "seeds": args.seeds}), flush=True)

    folds = make_folds(train_ids, args.folds, args.subject_folds)
    oof: dict[str, np.ndarray] = {}
    oof_trans: dict[str, np.ndarray] = {}
    test_probs: dict[str, list[np.ndarray]] = {r: [] for r in tests}
    fold_report = []
    ckpt_dir = Path(args.output) / "_fold_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for k, held in enumerate(folds):
        start = time.time()
        tr_ids = [r for r in train_ids if r not in held]
        tr = [data[r] for r in tr_ids]
        va = [data[r] for r in held]
        fold_trans = learn_transitions([gt for _, gt in tr], classes)
        ckpt = ckpt_dir / f"fold{k}.npz"
        if ckpt.is_file():
            saved = np.load(ckpt)
            for r in held:
                oof[r] = saved[f"oof__{r}"]
                oof_trans[r] = fold_trans
            for r in tests:
                test_probs[r].extend(saved[f"test__{r}"])
            fold_report.append({"fold": k, "held": held, "resumed": True})
            print(json.dumps(fold_report[-1]), flush=True)
            continue
        fold_test_start = {r: len(test_probs[r]) for r in tests}
        held_probs = {r: [] for r in held}
        histories = []
        fold_models = []
        for s in range(args.seeds):
            model, history = train_model(tr, classes, epochs=args.epochs,
                                         seed=100 * k + s + args.seed_offset,
                                         device=device, val=va, cosine=args.cosine, warp=args.warp,
                                         arch=(args.hidden, args.layers, args.stages),
                                         model_type=args.model,
                                         verbs=[verbs[r] for r in tr_ids] if verbs else None)
            histories.append(history)
            for r in held:
                held_probs[r].append(predict_proba(model, data[r][0], device))
            fold_models.append(model)
        torch.save([{k2: v.half().cpu() for k2, v in m.state_dict().items()} for m in fold_models],
                   ckpt_dir / f"fold{k}_models.pt")
        for r in tests:
            variants = load_test(r) if lazy_tests else test_variants[r]
            for model in fold_models:
                test_probs[r].append(np.mean([predict_proba(model, v, device) for v in variants], axis=0))
            del variants
        del fold_models, model
        torch.cuda.empty_cache()
        for r in held:
            oof[r] = np.mean(held_probs[r], axis=0)
            oof_trans[r] = fold_trans
        np.savez(ckpt, **{f"oof__{r}": oof[r] for r in held},
                 **{f"test__{r}": np.stack(test_probs[r][fold_test_start[r]:]) for r in tests})
        fold_report.append({"fold": k, "held": held, "val_acc_by_epoch": histories,
                            "seconds": round(time.time() - start, 1)})
        print(json.dumps(fold_report[-1]), flush=True)

    # Tune decoding on out-of-fold training predictions only.
    tuning = []
    for switch in (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
        for tw in (0.0, 0.5, 1.0, 2.0):
            ok = tot = 0
            for r in train_ids:
                pred_r = decode(oof[r], classes, switch=switch, trans=oof_trans[r], trans_weight=tw)
                o, t = labeled_acc(pred_r, data[r][1])
                ok, tot = ok + o, tot + t
            tuning.append({"switch": switch, "trans_weight": tw, "oof_acc": round(100 * ok / tot, 2)})
    best = max(tuning, key=lambda row: row["oof_acc"])
    print(json.dumps({"oof_tuning_best5": sorted(tuning, key=lambda r: -r["oof_acc"])[:5],
                      "chosen": best}), flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "oof_probs.npz", **oof)
    trans = learn_transitions([data[r][1] for r in train_ids], classes)
    per_test = {}
    for r in tests:
        prob = np.mean(test_probs[r], axis=0)
        pred = decode(prob, classes, switch=best["switch"], trans=trans, trans_weight=best["trans_weight"])
        target = out if len(tests) == 1 else out / r
        per_test[r] = _write_outputs(target, r, prob, pred, classes, args.bg_threshold)
        print(json.dumps({
            "test": r,
            "always_predict_all_labeled": round(per_test[r]["always_predict"]["all_labeled"], 2),
            "gated_covered": round(per_test[r]["background_gated"]["covered"], 2),
            "segment_top1": round(per_test[r]["always_predict"]["segment_top1"], 2),
        }), flush=True)

    report = {
        "method": "dinov2_mstcn_kfold_ensemble",
        "test_recordings": tests,
        "train_recordings": train_ids,
        "gt_used_at_inference": False,
        "decoding_tuned_on": "out-of-fold training predictions",
        "inner_folds_by_subject": args.subject_folds,
        "viterbi_switch": best["switch"],
        "transition_weight": best["trans_weight"],
        "progress_features": args.progress,
        "features": [str(feature_dir)] + [str(e) for e in extra],
        "crossfit": [[str(p) for p in pair] for pair in args.crossfit] if args.crossfit else None,
        "crossfit_set": args.crossfit_set,
        "verb_aux": args.verb_aux,
        "target": args.target,
        "classes": classes,
        "seeds_per_fold": args.seeds,
        "architecture": {"model": args.model, "hidden": args.hidden, "layers": args.layers, "stages": args.stages},
        "epochs": args.epochs,
        "cosine_lr": args.cosine,
        "time_warp": args.warp,
        "oof_acc": best["oof_acc"],
        "oof_tuning": tuning,
        "folds": fold_report,
        "per_test": per_test,
    }
    if len(tests) == 1:
        report.update(per_test[tests[0]])
        report["test_recording"] = tests[0]
    (out / "ar_object_paper_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fold_files = [ckpt_dir / f"fold{k}_models.pt" for k in range(len(folds))]
    if all(f.is_file() for f in fold_files):
        first = data[train_ids[0]][0]
        torch.save({
            "target": args.target, "classes": classes, "model_type": args.model,
            "dim": int(first.shape[0] if isinstance(first, torch.Tensor) else first.shape[1]),
            "aux": (max(max(v) for v in verbs.values()) + 1) if verbs else 0,
            "hidden": args.hidden, "layers": args.layers, "stages": args.stages,
            "features": {"base": feature_dir.name, "extra": [e.name for e in extra],
                         "finetuned": bool(args.crossfit or args.crossfit_set),
                         "crossfit_sets": len(args.crossfit or []) + len(args.crossfit_set or []),
                         "crossfit_set_sizes": [2] * len(args.crossfit or []) + [len(d) for _, d in cf_sets],
                         "progress": args.progress},
            "models": [sd for f in fold_files for sd in torch.load(f, weights_only=True)],
        }, out / "model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-recording", default="13_assy_0_1")
    parser.add_argument("--test-recordings", nargs="*", default=None,
                        help="Several held-out recordings (overrides --test-recording).")
    parser.add_argument("--train-recordings", nargs="*", default=None,
                        help="Explicit training recordings (default: every other recording with features).")
    parser.add_argument("--full-nouns", action="store_true",
                        help="Official IndustReal noun vocabulary (adds instruction, objects, ...).")
    parser.add_argument("--host-features", action="store_true",
                        help="Keep training features in host RAM; only the training window goes to the GPU.")
    parser.add_argument("--subject-folds", action="store_true",
                        help="Inner CV folds split by subject instead of recording.")
    parser.add_argument("--features", type=Path, default=ROOT / "output/ar_frame_features")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/kit_card_sam31_13_assy_0_1/overlay_tcn")
    parser.add_argument("--extra-features", type=Path, nargs="*", default=[])
    parser.add_argument("--crossfit", type=Path, nargs=2, action="append", default=None,
                        help="Fine-tuned feature dirs from backbone fold 0 and fold 1; repeat for more "
                             "fine-tuned feature sets (all must use --crossfit-meta's folds).")
    parser.add_argument("--crossfit-meta", type=Path, default=ROOT / "output/ar_finetune/meta.json")
    parser.add_argument("--crossfit-set", nargs="+", action="append", default=None, metavar="META DIR",
                        help="META (json with 'heldout': one recording list per backbone) then one fine-tuned "
                             "feature dir per backbone; repeat for more sets.")
    parser.add_argument("--verb-aux", action="store_true", help="Auxiliary verb head.")
    parser.add_argument("--target", choices=("noun", "verb"), default="noun")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--model", choices=("tcn", "causal_tcn", "gru"), default="tcn")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--progress", action="store_true",
                        help="Append normalised video-progress features.")
    parser.add_argument("--cosine", action="store_true", help="Cosine learning-rate decay.")
    parser.add_argument("--warp", action="store_true", help="Random 0.8-1.2x time-warp augmentation.")
    parser.add_argument("--bg-threshold", type=float, default=0.6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
