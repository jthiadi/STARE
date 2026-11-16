"""IndustReal action-recognition protocol (GT segment boundaries given, background excluded): fuse the TCN's
mean action probability inside each segment with the fine-tuned VideoMAE clip probabilities.

Which VideoMAE backbones to use and the fusion weight are chosen on the val segments; test is scored once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from monitor.settings import ROOT
from scripts.segmentation.combine_verb_noun_actions import joint_scores, run_classes
from scripts.segmentation.run_official_split_eval import TEST, VAL
from scripts.data.tas_labels import load_mapping, read_segments

CLIP_PROBS = ROOT / "output/official_split_eval/vmae_finetune/clip_probs.npz"
CLIP_WINDOWS = CLIP_PROBS.with_name("clip_probs_windows.npz")
FRAME_LOGITS = CLIP_PROBS.with_name("frame_logits")
DENSE_MODELS = ("full", "full_s3")
VJEPA_DIR = ROOT / "output/official_split_eval/vjepa2"   # clip_probs_*.npz of vjepa2_recognition / vjepa2_finetune


def tcn_segment_probs(decoder: dict, eval_dir: Path, rids: list[str], actions: list[str]) -> dict[str, np.ndarray]:
    """``rid|start|end`` -> mean over the segment's frames of softmax(joint verb x noun scores) over 74 actions."""
    verbs, nouns = decoder["verbs"], decoder["nouns"]
    out = {}
    for rid in rids:
        pv = [np.load(eval_dir / r / rid / "test_probs.npy") for r in decoder["verb_runs"]]
        pn = [np.load(eval_dir / r / rid / "test_probs.npy") for r in decoder["noun_runs"]]
        n = min(len(p) for p in pv + pn)
        mv = sum(w * p[:n] for w, p in zip(decoder["verb_weights"], pv))
        mn = sum(w * p[:n] for w, p in zip(decoder["noun_weights"], pn))
        s = joint_scores(mv, mn, verbs, nouns, actions)[:, 1:]
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        for _, start, end in read_segments(rid):
            lo, hi = max(0, start), min(n - 1, end)
            if hi >= lo:
                out[f"{rid}|{start}|{end}"] = p[lo:hi + 1].mean(axis=0)
    return out


def dense_segment_probs(keys: list[str], models=DENSE_MODELS) -> np.ndarray:
    """Per ``rid|start|end``: mean over the segment's frames of the dense per-frame action probabilities
    (``finetune_videomae_ar --score-frames``, 74 actions), averaged over ``models``."""
    frames, out = {}, []
    for key in keys:
        rid, start, end = key.split("|")
        if rid not in frames:
            probs = []
            for m in models:
                lg = np.load(FRAME_LOGITS / m / f"{rid}.npy").astype(np.float32)[:, 1:]
                p = np.exp(lg - lg.max(axis=1, keepdims=True))
                probs.append(p / p.sum(axis=1, keepdims=True))
            frames[rid] = sum(probs) / len(probs)
        p = frames[rid]
        lo, hi = max(0, int(start)), min(len(p) - 1, int(end))
        out.append(p[lo:hi + 1].mean(axis=0))
    return np.stack(out)


def load_segment_data(actions_dir: Path, eval_dir: Path, extra: bool = False,
                      vjepa: bool = False) -> tuple[dict, list[str], dict, dict]:
    """(decoder, clip model names, val-selectable clip sets, per split: labels / keys / tcn / clip probs).

    ``extra`` adds the 4-fold backbones (``fold4_*``) and the dense whole-segment probabilities (``dense``) as
    further val-selectable members; ``vjepa`` adds every V-JEPA 2 stream in ``VJEPA_DIR`` (the frozen probe
    ``vjepa`` and the fine-tuned ``vjft_*``): each alone and with every other set, all of them together, and the
    fine-tuned ones together (alone and with every other set)."""
    actions = load_mapping()
    decoder = json.loads((actions_dir / "decoder.json").read_text(encoding="utf-8"))
    assert run_classes(eval_dir / decoder["verb_runs"][0]) == decoder["verbs"]
    clip = np.load(CLIP_PROBS)
    names = sorted({f.split("/")[0] for f in clip.files} - {"labels", "keys"})
    fold4 = tuple(n for n in names if n.startswith("fold4_"))
    if not extra:
        names = [n for n in names if n not in fold4]
    fulls = tuple(n for n in names if n.startswith("full"))
    folds = tuple(n for n in names if n in ("fold0", "fold1"))
    # refit_trainval trained on the val videos, so it never takes part in the val selection.
    candidates = [folds] + [(n,) for n in fulls] + [fulls, fulls + folds]
    if extra and fold4:
        candidates += [fold4, fulls + fold4, fulls + folds + fold4]
    win = np.load(CLIP_WINDOWS) if CLIP_WINDOWS.is_file() else None
    windowed = [n for n in names if win is not None and all(f"{n}/{s}" in win.files for s in ("val", "test"))]
    candidates += [c + tuple(f"{n}~win" for n in c if n in windowed) for c in candidates
                   if any(n in windowed for n in c)]
    if extra:
        candidates += [("dense",)] + [c + ("dense",) for c in candidates if c]
    vj = {}
    if vjepa:
        for path in sorted(VJEPA_DIR.glob("clip_probs_*.npz")):
            arrays = np.load(path)
            vj.update({n: arrays for n in {f.split("/")[0] for f in arrays.files} - {"keys", "labels"}})
        base = list(candidates)
        tuned = tuple(n for n in vj if n.startswith("vjft_"))    # without the weaker frozen probe
        candidates += [(n,) for n in vj] + [c + (n,) for n in vj for c in base if c] + [tuple(vj)]
        candidates += [c + tuple(vj) for c in base if c]
        if tuned:
            candidates += [tuned] + [c + tuned for c in base if c]
    sets = {"+".join(c): c for c in candidates if c}

    data = {}
    for split, rids in (("val", VAL), ("test", TEST)):
        tcn = tcn_segment_probs(decoder, eval_dir, rids, actions)
        keys = [str(k) for k in clip[f"keys/{split}"]]
        assert set(keys) == set(tcn), f"{split}: segment keys differ"
        data[split] = {"labels": clip[f"labels/{split}"], "keys": keys, "tcn": np.stack([tcn[k] for k in keys]),
                       **{n: clip[f"{n}/{split}"] for n in names}}
        if windowed:
            assert [str(k) for k in win[f"keys/{split}"]] == keys, f"{split}: window keys differ"
            data[split].update({f"{n}~win": win[f"{n}/{split}"] for n in windowed})
        if extra:
            data[split]["dense"] = dense_segment_probs(keys)
        for n, arrays in vj.items():                 # same segments in the same order (some share their keys)
            assert [str(k) for k in arrays[f"keys/{split}"]] == keys, f"{split}: {n} segments differ"
            assert (arrays[f"labels/{split}"] == data[split]["labels"]).all(), f"{split}: {n} labels differ"
            data[split][n] = arrays[f"{n}/{split}"]
    names = names + [f"{n}~win" for n in windowed] + (["dense"] if extra else []) + list(vj)
    return decoder, names, sets, data


def with_refit(members: tuple, names: list[str]) -> tuple:
    """``members`` plus the train+val refit backbone (and its window probabilities when ``members`` use windows)."""
    extra = ("refit_trainval",)
    if any(m.endswith("~win") for m in members) and "refit_trainval~win" in names:
        extra += ("refit_trainval~win",)
    return members + extra


def fused_scores(d: dict, members: tuple, w: float) -> np.ndarray:
    """Per segment: (1 - w) log p_tcn + w log mean(p_videomae over ``members``), over the 74 actions."""
    if not members:
        return np.log(d["tcn"] + 1e-8)
    vm = sum(d[m] for m in members) / len(members)
    return (1 - w) * np.log(d["tcn"] + 1e-8) + w * np.log(vm + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-dir", type=Path, default=ROOT / "output/actions/vmaeft",
                        help="combine_verb_noun_actions output (decoder.json: runs + mix weights).")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "output/official_split_eval")
    parser.add_argument("--extra", action="store_true",
                        help="Also select among the 4-fold backbones and the dense whole-segment VideoMAE scores.")
    args = parser.parse_args()
    decoder, names, sets, data = load_segment_data(args.actions_dir, args.eval_dir, args.extra)

    def top1(split: str, members: tuple, w: float) -> float:
        d = data[split]
        return float((fused_scores(d, members, w).argmax(1) + 1 == d["labels"]).mean() * 100)

    grid = [{"clips": name, "weight": round(w, 2), "val_top1": round(top1("val", members, w), 2)}
            for name, members in sets.items() for w in np.arange(0, 1.01, 0.1)]
    best = max(grid, key=lambda r: r["val_top1"])
    members = sets[best["clips"]]
    report = {
        "protocol": "IndustReal action recognition: each GT segment of the 32 test videos, 74 actions, "
                    "background excluded; fusion chosen on the 16 val videos",
        "segments": {s: int(len(d["labels"])) for s, d in data.items()},
        "tcn_runs": {"verb": decoder["verb_runs"], "noun": decoder["noun_runs"]},
        "chosen_on_val": best,
        "test_top1_fused": round(top1("test", members, best["weight"]), 2),
        "test_top1_tcn_only": round(top1("test", (), 0.0), 2),
        "test_top1_videomae_only": {name: round(top1("test", m, 1.0), 2) for name, m in sets.items()},
        "val_grid": grid,
    }
    if "refit_trainval" in names:
        refit = with_refit(members, names)
        report["refit_trainval"] = {
            "note": "chosen clip set plus a VideoMAE refit on train + val (epochs fixed beforehand), same weight; "
                    "not validated, reported separately",
            "clips": "+".join(refit),
            "test_top1_fused": round(top1("test", refit, best["weight"]), 2),
            "test_top1_videomae_only": round(top1("test", refit, 1.0), 2)}
    (args.actions_dir / "recognition_fusion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from scripts.features.finetune_videomae_ar import TTA_FRACS

    (args.actions_dir / "recognition.json").write_text(json.dumps({
        "clip_models": list(members), "videomae_weight": best["weight"], "tta_fracs": list(TTA_FRACS),
        "test_top1_fused": report["test_top1_fused"], "test_top1_tcn_only": report["test_top1_tcn_only"],
        "test_top1_videomae_only": report["test_top1_videomae_only"][best["clips"]]}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "val_grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
