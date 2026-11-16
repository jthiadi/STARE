"""Procedure definitions, dataset access and step/state helpers.

Semantics follow the official IndustReal code (PSR/psr_utils.py, procedure_info.json):
  * 11 components, state bit k = component k installed (1) / not (0) / incorrectly (-1).
  * step id = 3k (install), 3k+1 (incorrect install), 3k+2 (remove).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # repository root (selection/psr/core.py)
DATA = Path(os.environ.get("INDUSTREAL_DATA", REPO / "data"))
OFFICIAL_PSR = REPO / "third_party" / "industreal_psr"   # official psr_utils.py + procedure_info.json (unmodified)
CAD_DIR = DATA / "part_geometries"

SPLIT_DIRS = {
    "train": DATA / "train" / "train_annotations",
    "val": DATA / "validation" / "val_annotations",
    "test": DATA / "test" / "test_annotations",
}

COMPONENTS = [
    "base", "front chassis", "front chassis pin", "rear chassis", "short rear chassis",
    "front rear chassis pin", "rear rear chassis pin", "front bracket", "front bracket screw",
    "front wheel assy", "rear wheel assy",
]

# Official ASD category codes (psr_utils.categories), state1..state22 -> CAD snapshot stateN.fbx
CAD_STATE_CODES = [
    "10000000000", "10010010000", "10010100000", "10010110000", "11100000000", "11110010000",
    "11110100000", "11110110000", "11110111100", "11110111110", "11110110001", "11110111101",
    "11110111111", "11110101111", "11110011111", "11110011110", "11110101110", "11100001110",
    "11101101110", "11101011110", "11101111110", "11101111111",
]

INITIAL_STATE = {"assy": "10000000000", "main": "11110111111"}


def load_procedure_info() -> list[dict]:
    with open(OFFICIAL_PSR / "procedure_info.json") as f:
        return json.load(f)


PROC_INFO = load_procedure_info()
STEP_DESC = {p["id"]: p["description"] for p in PROC_INFO}


def procedure_type(video: str) -> str:
    return "assy" if "assy" in video else "main"


def list_videos(split: str) -> list[str]:
    return sorted(os.listdir(SPLIT_DIRS[split]))


def video_dir(split: str, video: str) -> Path:
    return SPLIT_DIRS[split] / video


def frame_paths(split: str, video: str) -> list[Path]:
    d = video_dir(split, video) / "rgb"
    return [d / f for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


@dataclass
class ARSegment:
    cls: int
    name: str
    start: int
    end: int


def load_ar_labels(split: str, video: str) -> list[ARSegment]:
    out = []
    with open(video_dir(split, video) / "AR_labels.csv") as f:
        for r in csv.reader(f):
            out.append(ARSegment(int(r[1]), r[2], int(r[3][:-4]), int(r[4][:-4])))
    return out


def load_psr_gt(split: str, video: str, with_errors: bool = False) -> list[dict]:
    """Official GT loader format: list of {frame, id, description}. Evaluation-only."""
    name = "PSR_labels_with_errors.csv" if with_errors else "PSR_labels.csv"
    out = []
    with open(video_dir(split, video) / name) as f:
        for r in csv.reader(f):
            out.append({"frame": int(r[0][:-4]), "id": int(r[1]), "description": r[2]})
    return out


# ---------------------------------------------------------------- step/state algebra
def bits(code: str) -> tuple[int, ...]:
    return tuple(int(c) for c in code)


def code(state: tuple[int, ...]) -> str:
    return "".join(str(max(s, 0)) for s in state)


def step_is_legal(state: tuple[int, ...], step: int) -> bool:
    k, kind = divmod(step, 3)
    if kind == 0:
        return state[k] != 1          # install
    if kind == 1:
        return state[k] == 0          # incorrect install
    return state[k] != 0              # remove


def apply_steps(state: tuple[int, ...], steps) -> tuple[int, ...]:
    s = list(state)
    for st in steps:
        k, kind = divmod(st, 3)
        s[k] = {0: 1, 1: -1, 2: 0}[kind]
    return tuple(s)


def group_is_legal(state, group) -> bool:
    s = state
    for st in sorted(group):
        if not step_is_legal(s, st):
            return False
        s = apply_steps(s, [st])
    return True


def gt_groups(gt: list[dict]) -> list[tuple[int, tuple[int, ...]]]:
    """Group GT step completions that share a frame (e.g. chassis + pin fastened by one nut)."""
    by_f: dict[int, list[int]] = {}
    for e in gt:
        by_f.setdefault(e["frame"], []).append(e["id"])
    return [(f, tuple(sorted(v))) for f, v in sorted(by_f.items())]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))
