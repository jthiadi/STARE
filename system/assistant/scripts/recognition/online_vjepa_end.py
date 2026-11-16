"""The 77.2 % V-JEPA 2 ensemble labelled online: each action's label is fixed the moment it ends.

Same segments, members and fusion as the 77.2 % result (``sequence_recognition --vjepa``, 4-run report); the clip
scores only ever saw the action's own frames, so the one change is the action-order prior: forward filtering
(an action's label depends on it and the actions before it), not Viterbi over the whole video. Strength and
smoothing are chosen on the 16 val videos; the 32 test videos are scored once.

    python -m scripts.recognition.online_vjepa_end    # -> output/actions/vmaeft/online_vjepa_end.json
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping
from scripts.recognition.fuse_recognition import fused_scores, load_segment_data
from scripts.recognition.sequence_recognition import (SMOOTHING, STRENGTHS, decode_online, decode_segments,
                                                      log_prior, transition_counts, videos)
from scripts.segmentation.run_official_split_eval import TRAIN

ACTIONS_DIR = ROOT / "output/actions/vmaeft"
OFFLINE = ACTIONS_DIR / "sequence_recognition_vjepa_4runs.json"


def decode_filter(e: np.ndarray, groups: list[np.ndarray], first: np.ndarray, pairs: np.ndarray,
                  strength: float, smoothing: float) -> np.ndarray:
    """1-based action per segment, each video's segments forward-filtered under the order prior."""
    if strength == 0:
        return e.argmax(1) + 1
    start, trans = log_prior(first, pairs, smoothing)
    pred = np.empty(len(e), np.int64)
    for idx in groups:
        pred[idx] = decode_online(e[idx], strength * start, strength * trans, "filter")
    return pred + 1


def main() -> None:
    offline = json.loads(OFFLINE.read_text(encoding="utf-8"))["chosen_on_val"]
    members, w = tuple(offline["clips"].split("+")), offline["weight"]
    _, _, _, data = load_segment_data(ACTIONS_DIR, ROOT / "output/official_split_eval", False, True)
    first, pairs = transition_counts(TRAIN, load_mapping())
    scores = {split: fused_scores(data[split], members, w) for split in ("val", "test")}
    groups = {split: videos(data[split]["keys"]) for split in ("val", "test")}

    def top1(split: str, pred: np.ndarray) -> float:
        return round(float((pred == data[split]["labels"]).mean() * 100), 2)

    grid = [{"strength": lam, "smoothing": alpha,
             "val_top1": top1("val", decode_filter(scores["val"], groups["val"], first, pairs, lam, alpha))}
            for lam, alpha in itertools.product(STRENGTHS, SMOOTHING)]
    best = max(grid, key=lambda r: (r["val_top1"], -r["strength"]))
    offline_check = top1("test", decode_segments(scores["test"], groups["test"], first, pairs, offline["strength"],
                                                 offline["smoothing"], offline["viterbi"]))
    report = {
        "protocol": "IndustReal action recognition (GT segments, 74 actions); label fixed when the action ends, "
                    "order prior by forward filtering (past actions only); chosen on the 16 val videos, test once",
        "members": list(members), "weight": w,
        "chosen_on_val": best,
        "test_top1": top1("test", decode_filter(scores["test"], groups["test"], first, pairs, best["strength"],
                                                best["smoothing"])),
        "offline": {"val_top1": offline["val_top1"], "test_top1": offline_check,
                    "note": "the 77.2 result recomputed here (Viterbi over the whole video)"},
        "val_grid": grid,
    }
    (ACTIONS_DIR / "online_vjepa_end.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "val_grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
