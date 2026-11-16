"""IndustReal action recognition (GT segment boundaries given, background excluded) with the procedure order:
a first-order prior over consecutive actions, counted on the TRAIN videos, decodes each video's segments jointly
on top of the fused TCN + VideoMAE scores (``fuse_recognition``).

Everything (clip set, VideoMAE weight, prior strength, smoothing, Viterbi vs posterior argmax) is chosen on the
val segments; test is scored once.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from monitor.settings import ROOT
from scripts.recognition.fuse_recognition import fused_scores, load_segment_data, with_refit
from scripts.segmentation.run_official_split_eval import TRAIN
from scripts.data.tas_labels import load_mapping, read_segments

STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
SMOOTHING = (0.01, 0.1, 1.0)
WEIGHTS = tuple(round(w, 1) for w in np.arange(0, 1.01, 0.1))


def transition_counts(rids: list[str], actions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Counts of each video's first action and of consecutive action pairs, over the 74 actions (0-based)."""
    index = {a: i - 1 for i, a in enumerate(actions)}
    k = len(actions) - 1
    first, pairs = np.zeros(k), np.zeros((k, k))
    for rid in rids:
        seq = [index[a] for a, _, _ in read_segments(rid) if index[a] >= 0]
        if seq:
            first[seq[0]] += 1
        for a, b in zip(seq, seq[1:]):
            pairs[a, b] += 1
    return first, pairs


def log_prior(first: np.ndarray, pairs: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    start = np.log((first + alpha) / (first.sum() + alpha * len(first)))
    trans = np.log((pairs + alpha) / (pairs.sum(1, keepdims=True) + alpha * pairs.shape[1]))
    return start, trans


def _lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def decode(e: np.ndarray, start: np.ndarray, trans: np.ndarray, viterbi: bool) -> np.ndarray:
    """0-based action per segment for one video's emissions ``e`` (segments x actions, in time order)."""
    n = len(e)
    if viterbi:
        delta, back = e[0] + start, np.zeros((n, e.shape[1]), np.int64)
        for i in range(1, n):
            cand = delta[:, None] + trans
            back[i] = cand.argmax(0)
            delta = cand.max(0) + e[i]
        path = [int(delta.argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return np.array(path[::-1])
    fwd = np.zeros_like(e)
    fwd[0] = e[0] + start
    for i in range(1, n):
        fwd[i] = e[i] + _lse(fwd[i - 1][:, None] + trans, 0)
    bwd = np.zeros_like(e)
    for i in range(n - 2, -1, -1):
        bwd[i] = _lse(trans + (e[i + 1] + bwd[i + 1])[None, :], 1)
    return (fwd + bwd).argmax(1)


def decode_online(e: np.ndarray, start: np.ndarray, trans: np.ndarray, mode: str) -> np.ndarray:
    """0-based action per segment of one video, each using only that segment and the ones before it."""
    pred = np.empty(len(e), np.int64)
    if mode == "filter":
        fwd = e[0] + start
        pred[0] = fwd.argmax()
        for i in range(1, len(e)):
            fwd = e[i] + _lse(fwd[:, None] + trans, 0)
            fwd -= fwd.max()
            pred[i] = fwd.argmax()
        return pred
    pred[0] = (e[0] + start).argmax()
    for i in range(1, len(e)):
        pred[i] = (e[i] + trans[pred[i - 1]]).argmax()
    return pred


def videos(keys: list[str]) -> list[np.ndarray]:
    """Segment indices of each video, in time order."""
    by_rid: dict[str, list[tuple[int, int]]] = {}
    for i, key in enumerate(keys):
        rid, start, _ = key.split("|")
        by_rid.setdefault(rid, []).append((int(start), i))
    return [np.array([i for _, i in sorted(v)]) for v in by_rid.values()]


def decode_segments(e: np.ndarray, groups: list[np.ndarray], first: np.ndarray, pairs: np.ndarray,
                    strength: float, smoothing: float, viterbi: bool) -> np.ndarray:
    """1-based action per segment: each video's segments decoded jointly under the action-order prior."""
    if strength == 0:
        return e.argmax(1) + 1
    start, trans = log_prior(first, pairs, smoothing)
    pred = np.empty(len(e), np.int64)
    for idx in groups:
        pred[idx] = decode(e[idx], strength * start, strength * trans, viterbi)
    return pred + 1


def release_config(members: tuple, choice: dict, names: list[str]) -> dict:
    """What evaluate_release needs to redo the chosen recognition: backbones, fusion weight, order prior."""
    return {"clip_models": [m for m in members if not m.endswith("~win")],
            "window_models": [m[:-4] for m in members if m.endswith("~win")],
            "videomae_weight": choice["weight"],
            "order": {k: choice[k] for k in ("strength", "smoothing", "viterbi")},
            "order_counts": "action_order_counts.npz",
            "refit_model": "refit_trainval" if "refit_trainval" in names else None}


_VAL: dict = {}                                     # per worker process: val data, segment groups, prior counts


def _init_val(d: dict, groups: list[np.ndarray], first: np.ndarray, pairs: np.ndarray) -> None:
    _VAL.update(d=d, groups=groups, first=first, pairs=pairs)


def _val_rows(name: str, members: tuple) -> list[dict]:
    """Val top-1 of one clip set over every fusion weight and prior setting."""
    d, rows = _VAL["d"], []
    for w, lam in itertools.product(WEIGHTS, STRENGTHS):
        e = fused_scores(d, members, w)
        for alpha, viterbi in (itertools.product(SMOOTHING, (True, False)) if lam else [(SMOOTHING[0], True)]):
            pred = decode_segments(e, _VAL["groups"], _VAL["first"], _VAL["pairs"], lam, alpha, viterbi)
            rows.append({"clips": name, "weight": w, "strength": lam, "smoothing": alpha, "viterbi": viterbi,
                         "val_top1": round(float((pred == d["labels"]).mean() * 100), 2)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-dir", type=Path, default=ROOT / "output/actions/vmaeft")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "output/official_split_eval")
    parser.add_argument("--extra", action="store_true",
                        help="Also select among the 4-fold backbones and the dense whole-segment VideoMAE scores.")
    parser.add_argument("--vjepa", action="store_true",
                        help="Also select among sets with the frozen V-JEPA 2 probe; writes sequence_recognition_vjepa.json "
                             "only (the release files are left alone).")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Processes for the val search.")
    args = parser.parse_args()
    _, names, sets, data = load_segment_data(args.actions_dir, args.eval_dir, args.extra, args.vjepa)
    first, pairs = transition_counts(TRAIN, load_mapping())
    groups = {split: videos(d["keys"]) for split, d in data.items()}

    def top1(split: str, members: tuple, w: float, lam: float, alpha: float, viterbi: bool) -> float:
        pred = decode_segments(fused_scores(data[split], members, w), groups[split], first, pairs, lam, alpha,
                               viterbi)
        return float((pred == data[split]["labels"]).mean() * 100)

    with ProcessPoolExecutor(args.workers, initializer=_init_val,
                             initargs=(data["val"], groups["val"], first, pairs)) as pool:
        grid = [r for rows in pool.map(_val_rows, sets, sets.values()) for r in rows]   # in set order
    best = max(grid, key=lambda r: (r["val_top1"], -r["strength"]))
    plain = max((r for r in grid if r["strength"] == 0), key=lambda r: r["val_top1"])
    cfg = lambda r: (sets[r["clips"]], r["weight"], r["strength"], r["smoothing"], r["viterbi"])  # noqa: E731
    report = {
        "protocol": "IndustReal action recognition: each GT segment of the 32 test videos, 74 actions, background "
                    "excluded; action-order prior counted on the 36 train videos; all settings chosen on the 16 val "
                    "videos",
        "chosen_on_val": best,
        "test_top1_with_order": round(top1("test", *cfg(best)), 2),
        "without_order": {**plain, "test_top1": round(top1("test", *cfg(plain)), 2)},
    }
    if "refit_trainval" in names:
        refit = (with_refit(sets[best["clips"]], names),) + cfg(best)[1:]
        report["refit_trainval"] = {"note": "chosen settings, refit VideoMAE added; not validated",
                                    "test_top1_with_order": round(top1("test", *refit), 2)}
    report["val_grid"] = grid
    if args.vjepa:
        (args.actions_dir / "sequence_recognition_vjepa.json").write_text(json.dumps(report, indent=2),
                                                                          encoding="utf-8")
        # Test predictions of the chosen system, in render_online_overlay's layout (labels index load_mapping()).
        members, w, lam, alpha, viterbi = cfg(best)
        e = fused_scores(data["test"], members, w)
        pred = decode_segments(e, groups["test"], first, pairs, lam, alpha, viterbi)
        p = np.exp(e - e.max(1, keepdims=True))
        keys = [k.split("|") for k in data["test"]["keys"]]
        np.savez(args.actions_dir / "sequence_recognition_vjepa_test_predictions.npz",
                 actions=np.array(load_mapping()), rid=np.array([k[0] for k in keys]),
                 start=np.array([int(k[1]) for k in keys]), end=np.array([int(k[2]) for k in keys]),
                 label=data["test"]["labels"], pred=pred,
                 conf=(p / p.sum(1, keepdims=True))[np.arange(len(pred)), pred - 1])
        print(json.dumps({k: v for k, v in report.items() if k != "val_grid"}, indent=2))
        return
    (args.actions_dir / "sequence_recognition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.actions_dir / "action_order_counts.npz", first=first, pairs=pairs)
    config = release_config(sets[best["clips"]], best, names)
    config.update(test_top1=report["test_top1_with_order"],
                  test_top1_with_refit=report.get("refit_trainval", {}).get("test_top1_with_order"))
    (args.actions_dir / "recognition_order.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "val_grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
