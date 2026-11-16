"""Train / val / test evaluation on recordings never used for training.

Each run directory comes from ``train_ar_object_tcn`` with the val and test
recordings passed as held-out recordings (one subfolder of probabilities each).
Ensemble weights and decoding are chosen on the val recordings only; the test
recordings are then decoded once with those settings and scored.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import scripts.segmentation.train_ar_object_tcn as tcn
from monitor.settings import ROOT
from scripts.segmentation.evaluate_ar_object_segments import gt_noun_from_action, gt_object_frame_labels
from scripts.data.ar_labels import gt_object_from_action, load_ar_intervals
from scripts.segmentation.train_ar_object_tcn import FPS, _write_outputs, decode, labeled_acc, learn_transitions


def _intervals(rid: str) -> list[dict]:
    return load_ar_intervals(ROOT / "data/evaluation/official" / rid / "AR_labels.csv", source_fps=FPS)


def _gt(rid: str, n: int) -> list[str]:
    return gt_object_frame_labels(_intervals(rid), n_frames=n, fps=FPS, full_nouns=tcn.FULL_NOUNS)[0]


def clip_accuracy(rid: str, prob: np.ndarray, classes: list[str]) -> tuple[int, int]:
    """Paper protocol: each GT action clip (start/end given) gets the noun with the highest mean probability."""
    noun_of = gt_noun_from_action if tcn.FULL_NOUNS else gt_object_from_action
    obj = [i for i, c in enumerate(classes) if c]
    ok = tot = 0
    for row in _intervals(rid):
        noun = noun_of(row["action"])
        if not noun:
            continue
        s = max(0, int(round(row["start_s"] * FPS)))
        e = min(len(prob), int(round(row["end_s"] * FPS)) + 1)
        if e <= s:
            continue
        pred = classes[obj[int(prob[s:e, obj].mean(axis=0).argmax())]]
        tot += 1
        ok += int(pred == noun)
    return ok, tot


def _pooled(rows: list[dict]) -> dict:
    if not rows:
        return {}

    def pct(key_ok: str, key_tot: str) -> float:
        return round(100 * sum(r[key_ok] for r in rows) / max(1, sum(r[key_tot] for r in rows)), 2)

    return {
        "videos": len(rows),
        "clip_acc_paper_protocol_pct": pct("clip_ok", "clip_total"),
        "pooled_all_labeled_pct": pct("labeled_ok", "labeled_total"),
        "pooled_frame_incl_idle_pct": pct("frame_ok", "frame_total"),
        "mean_video_all_labeled_pct": round(sum(r["all_labeled"] for r in rows) / len(rows), 2),
        "mean_video_covered_gated_pct": round(sum(r["covered_gated"] for r in rows) / len(rows), 2),
        "mean_video_segment_top1_pct": round(sum(r["segment_top1"] for r in rows) / len(rows), 2),
        "clips": sum(r["clip_total"] for r in rows),
        "labeled_frames": sum(r["labeled_total"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[],
                        help="Extra held-out recordings reported separately (not pooled).")
    parser.add_argument("--bg-threshold", type=float, default=0.6)
    parser.add_argument("--full-nouns", action="store_true",
                        help="Score with the official 18-noun vocabulary (must match training).")
    parser.add_argument("--output", type=Path, default=ROOT / "output/unseen_split_eval/result")
    args = parser.parse_args()
    tcn.FULL_NOUNS = args.full_nouns

    train_ids = sorted(np.load(args.runs[0] / "oof_probs.npz").files)
    oof0 = np.load(args.runs[0] / "oof_probs.npz")
    train_gts = {r: _gt(r, len(oof0[r])) for r in train_ids}
    classes = [""] + sorted({g for gt in train_gts.values() for g in gt if g})
    held = args.val + args.test + args.reference
    probs = {r: [np.load(run / r / "test_probs.npy") for run in args.runs] for r in held}
    assert probs[held[0]][0].shape[1] == len(classes), (probs[held[0]][0].shape, classes)
    gts = {r: _gt(r, len(probs[r][0])) for r in held}
    trans = learn_transitions(list(train_gts.values()), classes)

    steps = np.linspace(0, 1, 6)
    grid = [w for w in itertools.product(steps, repeat=len(args.runs)) if abs(sum(w) - 1) < 1e-6]
    tuning = []
    for w in grid:
        mixed = {r: sum(wi * p for wi, p in zip(w, probs[r])) for r in args.val}
        for switch in (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0):
            for tw in (0.0, 0.5, 1.0, 2.0):
                ok = tot = 0
                for r in args.val:
                    o, t = labeled_acc(decode(mixed[r], classes, switch=switch, trans=trans, trans_weight=tw),
                                       gts[r])
                    ok, tot = ok + o, tot + t
                tuning.append({"weights": [round(float(x), 2) for x in w], "switch": switch,
                               "trans_weight": tw, "val_acc": round(100 * ok / tot, 2)})
    best = max(tuning, key=lambda row: row["val_acc"])
    print(json.dumps({"chosen_on_val": best}), flush=True)

    rows: dict[str, dict] = {}
    for r in held:
        prob = sum(wi * p for wi, p in zip(best["weights"], probs[r]))
        pred = decode(prob, classes, switch=best["switch"], trans=trans, trans_weight=best["trans_weight"])
        res = _write_outputs(args.output / r, r, prob, pred, classes, args.bg_threshold)
        clip_ok, clip_total = clip_accuracy(r, prob, classes)
        frame_ok, frame_total = res["background_gated"]["full_frame_n"]
        rows[r] = {
            "clip_acc": round(100 * clip_ok / max(1, clip_total), 2),
            "clip_ok": clip_ok,
            "clip_total": clip_total,
            "frame_incl_idle": round(100 * frame_ok / max(1, frame_total), 2),
            "frame_ok": frame_ok,
            "frame_total": frame_total,
            "recording": r,
            "task": "assy" if "_assy_" in r else "main",
            "all_labeled": round(res["always_predict"]["all_labeled"], 2),
            "covered_gated": round(res["background_gated"]["covered"], 2),
            "segment_top1": round(res["always_predict"]["segment_top1"], 2),
            "labeled_ok": res["labeled_ok"],
            "labeled_total": res["labeled_total"],
        }
        print(json.dumps(rows[r]), flush=True)

    test_rows = [rows[r] for r in args.test]
    report = {
        "train_recordings": train_ids,
        "val_recordings": args.val,
        "test_recordings": args.test,
        "selection": "ensemble weights + Viterbi decoding chosen on val recordings only",
        "vocabulary": "official 18 nouns" if args.full_nouns else "kit parts",
        "classes": classes,
        "gt_used_at_inference": False,
        "chosen": best,
        "val": _pooled([rows[r] for r in args.val]),
        "test": _pooled(test_rows),
        "test_assy": _pooled([r for r in test_rows if r["task"] == "assy"]),
        "test_main": _pooled([r for r in test_rows if r["task"] == "main"]),
        "reference": {r: rows[r] for r in args.reference},
        "per_video": rows,
        "val_tuning_top5": sorted(tuning, key=lambda row: -row["val_acc"])[:5],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("chosen", "val", "test", "test_assy", "test_main")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
