"""The IndustReal recognition recipe on MECCANO (GT segment boundaries, 61 actions, top-1 on the 7 test videos).

Per segment: (1 - w) log p_TCN + w log p_VideoMAE, then each video's segments decoded jointly with an action-order
prior (first action + consecutive pairs) - exactly as for IndustReal; only the inputs differ:
  * VideoMAE-SSv2 fine-tuned on MECCANO clips (16 frames over the segment, 3 samplings + multi-window).
  * One 62-class TCN (the IndustReal multi-stage TCN, 3 seeds averaged) on pretrained per-frame features:
    DINOv2-small (3 views) + VideoMAE-SSv2 (pooled + SSv2 logits) + video progress.
Two protocols, test scored once in each:
  * official: train on the 11 train videos, VideoMAE epoch / fusion weight / order prior chosen on the 2 val videos.
  * trainval: the 13 train + val videos pooled; settings chosen on 4-fold cross-validation (by video = person),
    final models trained on all 13.

Steps (resumable): --prepare, --train-vmae NAME, --train-tcn, --score, --select.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from meccano_action_end import data
from meccano_action_end.data import CACHE, OUT

VMAE_DIR = OUT / "vmae"
FEATS = {"dino": OUT / "feats/dino", "vmae_ssv2": OUT / "feats/vmae_ssv2"}
TCN_DIR = OUT / "tcn"
PROBS = OUT / "vmae_probs.npz"
GROUPS = 4
OFFICIAL = ("off_s2", "off_s3")
FINAL = ("tv_s7", "tv_s8")


def pool() -> list[str]:
    return sorted(data.split("train") + data.split("val"))


def groups() -> list[list[str]]:
    return [pool()[k::GROUPS] for k in range(GROUPS)]


def prepare() -> None:
    """Frame cache for VideoMAE, then pretrained DINOv2 and VideoMAE-SSv2 per-frame features for the TCN."""
    from scripts.features.finetune_videomae_ar import build_cache

    vids = data.split("train") + data.split("val") + data.split("test")
    CACHE.mkdir(parents=True, exist_ok=True)
    for vid in vids:
        start = time.time()
        build_cache(vid, data.video_path(vid), CACHE)
        print(json.dumps({"cache": vid, "frames": data.n_frames(vid), "seconds": round(time.time() - start, 1)}),
              flush=True)

    from transformers import AutoModel

    from monitor.settings import DINO_V2_DIR
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as vmae_x

    for name, out in FEATS.items():
        out.mkdir(parents=True, exist_ok=True)
        todo = [v for v in vids if not (out / f"{v}.npz").is_file()]
        if not todo:
            continue
        model = (AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda")
                 .half().eval() if name == "dino" else vmae_x.load_model())
        for vid in todo:
            start = time.time()
            if name == "dino":
                feats = dino_x.extract(model, data.video_path(vid), 16)
                np.savez(out / f"{vid}.npz", feats=feats, views=np.array([v[0] for v in dino_x.VIEWS]))
            else:
                feats, logits = vmae_x.extract(model, data.video_path(vid), 4, 4)
                np.savez(out / f"{vid}.npz", feats=feats[:, None, :], video_logits=logits)
            print(json.dumps({name: vid, "frames": int(len(feats)), "seconds": round(time.time() - start, 1)}),
                  flush=True)
        del model
        torch.cuda.empty_cache()


def train_vmae(name: str, args) -> None:
    """off_s<seed>: 11 train videos, epoch chosen on val clips; cv<k>: pool minus group k; tv_s<seed>: pool."""
    from scripts.features.finetune_videomae_ar import train_on_clips

    if name.startswith("off_s"):
        train, held, seed = data.clips(data.split("train")), data.clips(data.split("val")), int(name[5:])
    elif name.startswith("cv"):
        k = int(name[2:])
        train, held, seed = data.clips([v for v in pool() if v not in groups()[k]]), [], 20 + k
    else:
        train, held, seed = data.clips(pool()), [], int(name[4:])
    train_on_clips(VMAE_DIR / name, train, held, seed, args, len(data.actions()), cache=CACHE)


def _tcn_input(vid: str) -> np.ndarray:
    from scripts.segmentation.train_ar_object_tcn import load_features

    return load_features(vid, FEATS["dino"], True, [FEATS["vmae_ssv2"]])


def train_tcn(args) -> None:
    """``--tcn-seeds`` networks per training set, mean probabilities saved per predicted video: official
    (train -> val + test), cv<k> (pool minus group k -> group k), trainval (pool -> test)."""
    from scripts.segmentation.train_ar_object_tcn import _resident, predict_proba, train_model

    classes = [str(i) for i in range(len(data.actions()))]
    jobs = {"official": (data.split("train"), data.split("val") + data.split("test")),
            **{f"cv{k}": ([v for v in pool() if v not in g], g) for k, g in enumerate(groups())},
            "trainval": (pool(), data.split("test"))}
    for name, (train_vids, pred_vids) in jobs.items():
        out = TCN_DIR / name
        if all((out / f"{v}.npy").is_file() for v in pred_vids):
            continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        # Training features live on the GPU as fp16 so host RAM only ever holds one video.
        train = []
        for v in train_vids:
            f = _tcn_input(v)
            train.append((_resident(f, "cuda"), [str(x) for x in data.frame_labels(v, len(f))]))
            del f
        models = [train_model(train, classes, epochs=80, seed=100 + seed, device="cuda", cosine=True, warp=True,
                              arch=(192, 10, 4))[0] for seed in range(args.tcn_seeds)]
        del train
        torch.cuda.empty_cache()
        for v in pred_vids:
            f = _tcn_input(v)
            np.save(out / f"{v}.npy", np.mean([predict_proba(m, f, "cuda") for m in models], axis=0).astype(np.float16))
            del f
        del models
        torch.cuda.empty_cache()
        print(json.dumps({"tcn": name, "train": len(train_vids), "predicted": len(pred_vids),
                          "seconds": round(time.time() - start, 1)}), flush=True)


def score(args) -> None:
    """Whole-segment (3 samplings) + multi-window VideoMAE probabilities -> vmae_probs.npz."""
    from scripts.features.finetune_videomae_ar import load_model, tta_probs, window_probs

    arrays = dict(np.load(PROBS)) if PROBS.is_file() else {}
    jobs = [(n, s, data.split(s)) for n in OFFICIAL for s in ("val", "test")]
    jobs += [(f"cv{k}", "oof", g) for k, g in enumerate(groups())] + [(n, "test", data.split("test")) for n in FINAL]
    for name, split_name, vids in jobs:
        if f"{name}/{split_name}/win" in arrays:
            continue
        clips = data.clips(vids)
        model = load_model(len(data.actions()), VMAE_DIR / name / "best.pt")
        arrays[f"{name}/{split_name}/whole"] = tta_probs(model, clips, args.batch * 2, cache=CACHE).astype(np.float32)
        arrays[f"{name}/{split_name}/win"] = window_probs(model, clips, args.batch * 2, cache=CACHE).astype(np.float32)
        del model
        torch.cuda.empty_cache()
        np.savez(PROBS, **arrays)
        print(json.dumps({"scored": name, "split": split_name, "segments": len(clips)}), flush=True)


def _videos(clips: list) -> list[np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, c in enumerate(clips):
        out.setdefault(c[0], []).append(i)
    return [np.array(v) for v in out.values()]


def _tcn_segments(clips: list, run: str) -> np.ndarray:
    """log mean TCN action probability (background excluded, renormalised) over each segment."""
    frames, out = {}, []
    for vid, s, e, _ in clips:
        if vid not in frames:
            p = np.load(TCN_DIR / run / f"{vid}.npy").astype(np.float32)[:, 1:]
            frames[vid] = p / p.sum(axis=1, keepdims=True)
        p = frames[vid]
        out.append(p[min(s, len(p) - 1):min(e, len(p) - 1) + 1].mean(axis=0))
    return np.log(np.stack(out) + 1e-8)


def _vjepa_probs() -> dict[str, dict[str, np.ndarray]]:
    """Per fine-tuned V-JEPA 2 run (``meccano_action_end.vjepa2``): its val / test probabilities, in ``data.clips``
    order."""
    from meccano_action_end.vjepa2 import VJ_DIR

    out: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(VJ_DIR.glob("clip_probs_*.npz")):
        r = np.load(f)
        name = f.stem[len("clip_probs_"):]
        for split_name in ("val", "test"):
            keys = [f"{v}|{s}|{e}" for v, s, e, _ in data.clips(data.split(split_name))]
            assert [str(k) for k in r[f"keys/{split_name}"]] == keys, f"{name} {split_name}: segments differ"
            out.setdefault(name, {})[split_name] = r[f"{name}/{split_name}"]
    return out


def select() -> None:
    from scripts.recognition.sequence_recognition import SMOOTHING, STRENGTHS, WEIGHTS, decode_segments

    arrays = np.load(PROBS)
    tr, va, te = data.split("train"), data.split("val"), data.split("test")

    def vm(models, split_name):
        return {m: sum(arrays[f"{n}/{split_name}/{m}"] for n in models) / len(models) for m in ("whole", "win")}

    folds = groups()
    oof_clips = [data.clips(g) for g in folds]
    sets = {
        "official": {
            "select": {"clips": data.clips(va), "tcn": _tcn_segments(data.clips(va), "official"), **vm(OFFICIAL, "val"),
                       "priors": [(np.arange(len(data.clips(va))), data.transition_counts(tr))]},
            "test": {"clips": data.clips(te), "tcn": _tcn_segments(data.clips(te), "official"), **vm(OFFICIAL, "test"),
                     "priors": [(np.arange(len(data.clips(te))), data.transition_counts(tr))]}},
        "trainval": {
            "select": {"clips": sum(oof_clips, []),
                       "tcn": np.concatenate([_tcn_segments(c, f"cv{k}") for k, c in enumerate(oof_clips)]),
                       **{m: np.concatenate([arrays[f"cv{k}/oof/{m}"] for k in range(GROUPS)]) for m in ("whole", "win")},
                       "priors": []},
            "test": {"clips": data.clips(te), "tcn": _tcn_segments(data.clips(te), "trainval"), **vm(FINAL, "test"),
                     "priors": [(np.arange(len(data.clips(te))), data.transition_counts(pool()))]}},
    }
    vj = _vjepa_probs()                                # official protocol only (trained on the 11 train videos)
    for name, probs in vj.items():
        for split_name, part in (("val", "select"), ("test", "test")):
            sets["official"][part][name] = probs[split_name]
    start = 0
    for k, c in enumerate(oof_clips):
        sets["trainval"]["select"]["priors"].append(
            (np.arange(start, start + len(c)), data.transition_counts([v for v in pool() if v not in folds[k]])))
        start += len(c)
    runs = tuple(vj)                                   # each V-JEPA 2 run is one member, as on IndustReal
    clip_sets = {"whole": ("whole",), "whole+win": ("whole", "win")}
    if runs:
        clip_sets.update({"vj": runs, "whole+vj": ("whole",) + runs, "whole+win+vj": ("whole", "win") + runs})

    def top1(d: dict, clip_set: str, w: float, lam: float, alpha: float, viterbi: bool) -> float:
        vmp = sum(d[m] for m in clip_sets[clip_set]) / len(clip_sets[clip_set])
        e = (1 - w) * d["tcn"] + w * np.log(vmp + 1e-8)
        pred = np.empty(len(e), np.int64)
        for idx, (first, pairs) in d["priors"]:
            pred[idx] = decode_segments(e[idx], _videos([d["clips"][i] for i in idx]), first, pairs, lam, alpha,
                                        viterbi)
        labels = np.array([c[3] for c in d["clips"]])
        return float((pred == labels).mean() * 100)

    report = {"protocol": "MECCANO action recognition: GT segment boundaries, 61 actions, top-1 over every test "
                          "segment of the 7 test videos; RGB only",
              "published_rgb_test_top1": {"SlowFast (CVIU 2023)": 45.16,
                                          "Swin3D-B SSv2 + focal loss, train (challenge winner)": 50.80,
                                          "Swin3D-B SSv2 + focal loss, train + val": 53.03}}
    for protocol, d in sets.items():
        grid = []
        usable = [c for c, members in clip_sets.items() if all(m in d["select"] for m in members)]
        for clip_set, w, lam in itertools.product(usable, WEIGHTS, STRENGTHS):
            for alpha, viterbi in (itertools.product(SMOOTHING, (True, False)) if lam else [(SMOOTHING[0], True)]):
                grid.append({"clips": clip_set, "weight": w, "strength": lam, "smoothing": alpha, "viterbi": viterbi,
                             "select_top1": round(top1(d["select"], clip_set, w, lam, alpha, viterbi), 2)})
        best = max(grid, key=lambda r: (r["select_top1"], -r["strength"]))
        plain = max((r for r in grid if r["strength"] == 0), key=lambda r: r["select_top1"])
        cfg = lambda r: (r["clips"], r["weight"], r["strength"], r["smoothing"], r["viterbi"])  # noqa: E731
        report[protocol] = {
            "selection": "2 val videos" if protocol == "official" else "4-fold cross-validation over 13 videos",
            "segments": {"select": len(d["select"]["clips"]), "test": len(d["test"]["clips"])},
            "chosen": best,
            "test_top1": round(top1(d["test"], *cfg(best)), 2),
            "without_order": {**plain, "test_top1": round(top1(d["test"], *cfg(plain)), 2)},
            "test_top1_tcn_only": round(top1(d["test"], "whole", 0.0, 0.0, 0.01, True), 2),
            "test_top1_videomae_only": round(top1(d["test"], best["clips"], 1.0, 0.0, 0.01, True), 2),
            "select_top1_tcn_only": round(top1(d["select"], "whole", 0.0, 0.0, 0.01, True), 2),
            "select_top1_videomae_only": round(top1(d["select"], best["clips"], 1.0, 0.0, 0.01, True), 2),
            "grid": grid,
        }
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "grid"} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-vmae", help="off_s2, off_s3, cv0..cv3, tv_s7, tv_s8")
    parser.add_argument("--train-tcn", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tcn-seeds", type=int, default=3)
    args = parser.parse_args()
    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(4)
    if args.prepare:
        prepare()
    if args.train_vmae:
        train_vmae(args.train_vmae, args)
    if args.train_tcn:
        train_tcn(args)
    if args.score:
        score(args)
    if args.select:
        select()


if __name__ == "__main__":
    main()
