"""MECCANO (Ragusa et al., WACV 2021 / CVIU 2023): 20 egocentric videos of 20 people assembling a toy motorbike,
12 fps, 61 verb_object actions; official split 11 train / 2 val / 7 test videos (one person per video).

Class 0 is an unused background slot so MECCANO action ids map to 1..61 (the IndustReal recognition tools treat
class 0 as background and exclude it). Data root: ``$MECCANO_ROOT``, else STARE's ``data/MECCANO``, holding
``annotations/MECCANO_{train,val,test}_actions.csv`` and the extracted ``MECCANO_RGB_Videos.zip``.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from monitor.settings import ROOT

DATA = Path(os.environ.get("MECCANO_ROOT", ROOT.parents[1] / "data/MECCANO"))          # STARE/data/MECCANO
OUT = ROOT / "output/meccano_action_end"
CACHE = OUT / "frames"
FPS = 12


@lru_cache(maxsize=None)
def _rows() -> dict[str, list[dict]]:
    out = {}
    for split in ("train", "val", "test"):
        with (DATA / "annotations" / f"MECCANO_{split}_actions.csv").open(encoding="utf-8") as handle:
            out[split] = list(csv.DictReader(handle))
    return out


def split(name: str) -> list[str]:
    """Video ids of ``train`` / ``val`` / ``test``."""
    return sorted({r["video_id"] for r in _rows()[name]})


@lru_cache(maxsize=None)
def actions() -> list[str]:
    """Index -> name; index 0 is the unused background slot, index i is MECCANO action id i - 1."""
    names = {int(r["action_id"]): r["action_name"] for rows in _rows().values() for r in rows}
    return ["background"] + [names[i] for i in range(len(names))]


@lru_cache(maxsize=None)
def segments(vid: str) -> tuple[tuple[int, int, int], ...]:
    """(label 1..61, start, end inclusive) with 0-based frame indices, sorted by start."""
    rows = [r for rows in _rows().values() for r in rows if r["video_id"] == vid]
    segs = [(int(r["action_id"]) + 1, int(Path(r["start_frame"]).stem) - 1, int(Path(r["end_frame"]).stem) - 1)
            for r in rows]
    return tuple(sorted(segs, key=lambda s: (s[1], s[2])))


def video_path(vid: str) -> Path:
    found = sorted(p for p in DATA.rglob(f"{vid}.*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    assert found, f"no video for {vid} under {DATA}"
    return found[0]


def n_frames(vid: str) -> int:
    return len(np.load(CACHE / f"{vid}.idx.npy")) - 1


def clips(vids: list[str]) -> list[tuple[str, int, int, int]]:
    """(vid, start, end, label) per annotated segment, clamped to the cached frames, per video in start order."""
    out = []
    for vid in vids:
        n = n_frames(vid)
        out += [(vid, max(0, s), min(n - 1, e), label) for label, s, e in segments(vid) if min(n - 1, e) >= max(0, s)]
    return out


def frame_labels(vid: str, n: int) -> np.ndarray:
    """One label per frame (0 between actions); where segments overlap the later-starting one wins."""
    labels = np.zeros(n, np.int64)
    for label, s, e in segments(vid):
        labels[max(0, s):min(n, e + 1)] = label
    return labels


def transition_counts(vids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Counts of each video's first action and of consecutive action pairs (0-based over the 61 actions)."""
    k = len(actions()) - 1
    first, pairs = np.zeros(k), np.zeros((k, k))
    for vid in vids:
        seq = [label - 1 for label, _, _ in segments(vid)]
        if seq:
            first[seq[0]] += 1
        for a, b in zip(seq, seq[1:]):
            pairs[a, b] += 1
    return first, pairs
