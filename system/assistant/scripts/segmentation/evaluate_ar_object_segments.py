"""Paper-style AR *object* metrics (IndustReal AR eval definitions, noun-only).

Mirrors ``assembly-task-checking/AR/scripts/eval_accuracy.py`` FairAcc, but
compares object nouns instead of full verb+object actions. GT verbs are stripped
(``take_nut`` → ``nut``). No GT is used at inference — this is eval only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from monitor.settings import ROOT
from scripts.data.ar_labels import gt_object_from_action, load_ar_intervals


BG = ""  # background / no object


def load_pred_timeline(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "kit_label": str(row.get("kit_label") or "").strip(),
                "ar_object": str(row.get("ar_object") or "").strip(),
                "score": float(row["score"]) if row.get("score") not in {None, ""} else None,
                "held": str(row.get("held") or "").lower() in {"1", "true", "yes"},
            })
    return rows


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def frames_to_segments(labels: list[str], fps: float) -> list[tuple[str, float, float]]:
    """Collapse consecutive equal non-empty labels into [start_s, end_s) segments."""
    segments: list[tuple[str, float, float]] = []
    i = 0
    n = len(labels)
    while i < n:
        label = labels[i]
        if not label:
            i += 1
            continue
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((label, i / fps, j / fps))
        i = j
    return segments


_NOUN_ALIASES = {"short_pin": "pin_short"}


def gt_noun_from_action(action: str) -> str | None:
    """Every noun of the official IndustReal AR vocabulary (instruction, objects, ... included)."""
    _, _, noun = str(action or "").strip().partition("_")
    return _NOUN_ALIASES.get(noun, noun) or None


def gt_object_frame_labels(
    intervals: list[dict],
    *,
    n_frames: int,
    fps: float,
    full_nouns: bool = False,
) -> tuple[list[str], list[tuple[str, float, float]]]:
    """Per-frame GT object noun + GT object segments (verb stripped).

    ``full_nouns``: use the official 18-noun vocabulary instead of the kit-part subset.
    """
    labels = [BG] * n_frames
    for row in intervals:
        obj = gt_noun_from_action(row["action"]) if full_nouns else gt_object_from_action(row["action"])
        if not obj:
            continue
        start = max(0, int(round(row["start_s"] * fps)))
        end = min(n_frames, int(round(row["end_s"] * fps)) + 1)
        for index in range(start, end):
            # Prefer keeping an existing object if two overlap; first write wins
            # unless current is empty.
            if not labels[index]:
                labels[index] = obj
    segments = frames_to_segments(labels, fps)
    return labels, segments


def pred_object_frame_labels(
    timeline: list[dict],
    *,
    n_frames: int,
) -> list[str]:
    labels = [BG] * n_frames
    for row in timeline:
        index = int(row["frame_index"])
        if 0 <= index < n_frames:
            labels[index] = str(row.get("ar_object") or "").strip()
    return labels


def majority_smooth(labels: list[str], *, window: int = 7) -> list[str]:
    """Causal-ish centered majority vote over non-empty labels in a window."""
    if window <= 1:
        return list(labels)
    half = window // 2
    out = list(labels)
    n = len(labels)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        counts: dict[str, int] = {}
        for j in range(lo, hi):
            lab = labels[j]
            if not lab:
                continue
            counts[lab] = counts.get(lab, 0) + 1
        if counts:
            out[i] = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return out


def evaluate_object_stream(
    gt: list[str],
    pred: list[str],
    gt_segments: list[tuple[str, float, float]],
    pred_segments: list[tuple[str, float, float]],
    *,
    tiou_th: float = 0.5,
) -> dict:
    """FairAcc-style report for object nouns."""
    n = min(len(gt), len(pred))
    gt = gt[:n]
    pred = pred[:n]
    gt_arr = np.array(gt, dtype=object)
    pred_arr = np.array(pred, dtype=object)
    gt_act = gt_arr != BG
    pred_act = pred_arr != BG

    full_ok = int((pred_arr == gt_arr).sum())
    full_tot = n
    gt_ok = int((pred_arr[gt_act] == gt_arr[gt_act]).sum()) if gt_act.any() else 0
    gt_tot = int(gt_act.sum())
    covered = gt_act & pred_act
    cov_ok = int((pred_arr[covered] == gt_arr[covered]).sum()) if covered.any() else 0
    cov_tot = int(covered.sum())

    tp = int((gt_act & pred_act).sum())
    fp = int((~gt_act & pred_act).sum())
    fn = int((gt_act & ~pred_act).sum())
    tn = int((~gt_act & ~pred_act).sum())

    ious = []
    seg_ok = 0
    for label, gs, ge in gt_segments:
        best_i, best_iou = -1, 0.0
        for j, (plabel, ps, pe) in enumerate(pred_segments):
            value = tiou((gs, ge), (ps, pe))
            if value > best_iou:
                best_iou, best_i = value, j
        ious.append(best_iou)
        if best_i >= 0 and best_iou >= tiou_th and pred_segments[best_i][0] == label:
            seg_ok += 1

    def pct(a: int, b: int) -> float:
        return 100.0 * a / max(1, b)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "covered": pct(cov_ok, cov_tot),
        "covered_n": [cov_ok, cov_tot],
        "all_labeled": pct(gt_ok, gt_tot),
        "all_labeled_n": [gt_ok, gt_tot],
        "full_frame_incl_bg": pct(full_ok, full_tot),
        "full_frame_n": [full_ok, full_tot],
        "actionness_p": 100.0 * prec,
        "actionness_r": 100.0 * rec,
        "actionness_f1": 100.0 * f1,
        "segment_top1": pct(seg_ok, len(gt_segments)),
        "segment_n": [seg_ok, len(gt_segments)],
        "mean_tiou": float(np.mean(ious)) if ious else 0.0,
        "tiou_th": tiou_th,
        "tp_fp_fn_tn": [tp, fp, fn, tn],
        "note": (
            "Paper-style FairAcc, object-noun only (verb stripped). "
            "'covered' matches the legacy IndustReal AR covered definition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeline",
        type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1/overlay/ar_object_timeline.csv",
    )
    parser.add_argument(
        "--ar-labels",
        type=Path,
        default=ROOT / "data/evaluation/official/13_assy_0_1/AR_labels.csv",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--tiou-th", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=0,
                        help="Optional majority smooth on pred before scoring.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    timeline = load_pred_timeline(args.timeline)
    intervals = load_ar_intervals(args.ar_labels, source_fps=float(args.fps))
    n_frames = max((row["frame_index"] for row in timeline), default=-1) + 1
    if n_frames <= 0:
        raise SystemExit("Empty timeline")

    gt, gt_segments = gt_object_frame_labels(
        intervals, n_frames=n_frames, fps=float(args.fps),
    )
    pred = pred_object_frame_labels(timeline, n_frames=n_frames)
    if args.smooth_window > 1:
        pred = majority_smooth(pred, window=int(args.smooth_window))
    pred_segments = frames_to_segments(pred, float(args.fps))
    report = evaluate_object_stream(
        gt, pred, gt_segments, pred_segments, tiou_th=float(args.tiou_th),
    )
    report["timeline"] = str(args.timeline)
    report["ar_labels"] = str(args.ar_labels)
    report["n_frames"] = n_frames
    report["smooth_window"] = int(args.smooth_window)
    report["gt_object_segments"] = len(gt_segments)
    report["pred_object_segments"] = len(pred_segments)

    out = args.output
    if out is None:
        out = args.timeline.parent / "ar_object_paper_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
