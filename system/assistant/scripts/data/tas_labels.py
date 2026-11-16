"""IndustReal AR start/end annotations -> one action label per frame (10 fps).

Frames outside every action are background (index 0). Where two annotated actions overlap,
the one that starts later wins. Labels are used for training and evaluation only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from monitor.settings import ROOT

LABELS = ROOT / "data/evaluation/official"
MAPPING = ROOT / "data/action_mapping.txt"
BG = "background"


def load_mapping() -> list[str]:
    """Index -> action name; index 0 is background."""
    rows = MAPPING.read_text(encoding="utf-8").splitlines()
    return [r.split(" ", 1)[1] for r in rows if r.strip()]


def read_segments(rid: str) -> list[tuple[str, int, int]]:
    """(action, start_frame, end_frame_inclusive), sorted by start."""
    rows = []
    with (LABELS / rid / "AR_labels.csv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 5 or not row[1].strip().isdigit():
                continue
            rows.append((row[2].strip(), int(Path(row[3]).stem), int(Path(row[4]).stem)))
    return sorted(rows, key=lambda r: r[1])


def frame_labels(rid: str, n: int, mapping: list[str] | None = None) -> np.ndarray:
    index = {a: i for i, a in enumerate(mapping or load_mapping())}
    labels = np.zeros(n, np.int64)
    for action, start, end in read_segments(rid):
        labels[max(0, start):min(n, end + 1)] = index[action]
    return labels
