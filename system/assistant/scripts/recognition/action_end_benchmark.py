"""The action-end label on the IndustReal paper's action-recognition test: true segments, 74 actions, background left out.

It calls the live assistant's own function (``assist.extract.action_end.end_labels``) with the annotated segments as its spans.
The recipe was locked on val before test was loaded for it. Test is scored once and reported whatever the result; it is refused
unless val first reproduces the locked 73.81. Test has been used before for related models (77.16, 77.6, 78.1), so this is not a
first look at that test set, though nothing in this recipe was chosen on it.

    python -m scripts.recognition.action_end_benchmark   # CPU -> output/assist/action_end/paper_test.json
"""
from __future__ import annotations

import json

import numpy as np

from assist.common import load_config
from assist.extract.action_end import MEMBERS, SAMPLINGS, SMOOTHING, STRENGTH, WEIGHT, end_labels, mean_clip
from assist.extract.run_ar import from_cache
from assist.industreal import SPLITS
from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping
from scripts.recognition.pipeline_on_segments import guard_single_look
from scripts.recognition.sequence_recognition import transition_counts
from scripts.segmentation.run_official_split_eval import TRAIN

OUT = ROOT / "output/assist/action_end"
VAL_LOCKED = 73.81
PAPER_BEST = 65.25
VMAE = ROOT / "output/official_split_eval/vmae_finetune/clip_probs.npz"
VJEPA = ROOT / "output/official_split_eval/vjepa2"


def load_split(split: str) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Segment keys, 1-based labels, and each member's (segments, 74) probabilities (saved with these 3 samplings)."""
    vmae = np.load(VMAE)
    keys, labels = [str(k) for k in vmae[f"keys/{split}"]], vmae[f"labels/{split}"]
    probs = {"vmae_full": vmae[f"full/{split}"]}
    for name in MEMBERS[1:]:
        saved = np.load(VJEPA / f"clip_probs_{name}.npz")
        if [str(k) for k in saved[f"keys/{split}"]] != keys or (saved[f"labels/{split}"] != labels).any():
            raise ValueError(f"{name}: {split} segments differ")
        probs[name] = saved[f"{name}/{split}"]
    return keys, labels, probs


def top1(split: str, cfg: dict, actions: list[str], first: np.ndarray, pairs: np.ndarray) -> dict:
    keys, labels, probs = load_split(split)
    clip = mean_clip(probs)
    right, total = 0, 0
    for rid in SPLITS[split]:
        rows = sorted(((int(k.split("|")[1]), i) for i, k in enumerate(keys) if k.split("|")[0] == rid))
        index = np.array([i for _, i in rows])
        spans = [tuple(int(x) for x in keys[i].split("|")[1:]) for i in index]
        pred = end_labels(from_cache(rid, cfg, actions), spans, clip[index], WEIGHT, STRENGTH, SMOOTHING, first, pairs)
        right += int((pred == labels[index]).sum())
        total += len(index)
    if total != len(labels):
        raise ValueError(f"{split}: scored {total} of {len(labels)} segments")
    return {"segments": total, "top1": round(100 * right / total, 2)}


def main() -> None:
    result = OUT / "paper_test.json"
    guard_single_look(result)
    cfg, actions = load_config(), load_mapping()
    first, pairs = transition_counts(TRAIN, actions)
    val = top1("val", cfg, actions, first, pairs)
    print(json.dumps({"val": val}), flush=True)
    if abs(val["top1"] - VAL_LOCKED) > 0.01:
        raise SystemExit(f"val gives {val['top1']}, not the locked {VAL_LOCKED}: test is not scored")
    report = {"protocol": "IndustReal action recognition: annotated segments, 74 actions; the live assistant's end_labels; "
                          "only each action's frames and earlier actions; recipe locked on val, test once",
              "members": list(MEMBERS), "samplings": list(SAMPLINGS), "weight": WEIGHT, "strength": STRENGTH,
              "smoothing": SMOOTHING, "val": val, "test": top1("test", cfg, actions, first, pairs), "paper_best": PAPER_BEST}
    OUT.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
