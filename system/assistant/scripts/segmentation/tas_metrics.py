"""Temporal action segmentation metrics: frame accuracy, Edit, F1@{10,25,50}.

Same definitions as MS-TCN / FACT: Edit is the normalised Levenshtein score of the segment
sequences averaged over videos; F1 pools true/false positives over all videos.
"""
from __future__ import annotations

import numpy as np

OVERLAPS = (0.10, 0.25, 0.50)


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end_exclusive) runs."""
    labels = np.asarray(labels)
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(labels)]
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _levenshtein(p: list[int], y: list[int]) -> float:
    d = np.zeros((len(p) + 1, len(y) + 1))
    d[:, 0] = np.arange(len(p) + 1)
    d[0, :] = np.arange(len(y) + 1)
    for j in range(1, len(y) + 1):
        for i in range(1, len(p) + 1):
            d[i, j] = d[i - 1, j - 1] if y[j - 1] == p[i - 1] else 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[-1, -1]


def edit_score(pred: np.ndarray, gt: np.ndarray, bg: set[int]) -> float:
    p = [s[0] for s in segments(pred) if s[0] not in bg]
    y = [s[0] for s in segments(gt) if s[0] not in bg]
    if not p and not y:
        return 100.0
    return (1 - _levenshtein(p, y) / max(len(p), len(y))) * 100


def f_counts(pred: np.ndarray, gt: np.ndarray, overlap: float, bg: set[int]) -> tuple[int, int, int]:
    p = [s for s in segments(pred) if s[0] not in bg]
    y = [s for s in segments(gt) if s[0] not in bg]
    if not y:
        return 0, len(p), 0
    y_label = np.array([s[0] for s in y])
    y_start = np.array([s[1] for s in y])
    y_end = np.array([s[2] for s in y])
    hits = np.zeros(len(y), bool)
    tp = fp = 0
    for label, start, end in p:
        inter = np.minimum(end, y_end) - np.maximum(start, y_start)
        union = np.maximum(end, y_end) - np.minimum(start, y_start)
        iou = inter / union * (y_label == label)
        idx = int(iou.argmax())
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = True
        else:
            fp += 1
    return tp, fp, int(len(y) - hits.sum())


def score(preds: dict, include_bg: bool, bg_label: int = 0) -> dict:
    """``preds``: {video: {"pred": int array, "gt": int array}}.

    include_bg=True: background is an ordinary class (frame accuracy over all frames).
    include_bg=False: background frames/segments are ignored.
    """
    bg = set() if include_bg else {bg_label}
    gts = [np.asarray(p["gt"]) for p in preds.values()]
    prs = [np.asarray(p["pred"]) for p in preds.values()]
    gt_all, pr_all = np.concatenate(gts), np.concatenate(prs)
    keep = np.ones(len(gt_all), bool) if include_bg else gt_all != bg_label
    out = {"Acc": float((gt_all == pr_all)[keep].mean() * 100),
           "Edit": float(np.mean([edit_score(p, g, bg) for p, g in zip(prs, gts)]))}
    for overlap in OVERLAPS:
        tp = fp = fn = 0
        for p, g in zip(prs, gts):
            a, b, c = f_counts(p, g, overlap, bg)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        out[f"F1@{int(overlap * 100)}"] = float(2 * precision * recall / (precision + recall + 1e-5) * 100)
    return out
