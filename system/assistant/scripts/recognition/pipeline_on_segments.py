"""The models of the live pipeline on the IndustReal paper's action-recognition test, with no live limits.

Each annotated segment is cut at its true start and end and classified among the 74 actions (background left out), as in
the paper; the label is fixed when the segment ends. The models are the pipeline's:

  TCN       the causal verb x noun TCN, its frame scores averaged over the segment's own frames
  V-JEPA 2  the hand-crop, 12-layer, action-beginning run, on the segment's own frames (3 temporal samplings)

They are blended as (1 - w) log p_tcn + w log p_vjepa, and the action-order prior adds the earlier actions only.

The rule, fixed before any result was seen: w (0 to 1 in steps of 0.1) and the prior's strength and smoothing are chosen on the
16 val videos; the 32 test videos are scored once, and the result is reported whether or not it beats the paper's 65.25. For
comparison the same run scores V-JEPA 2 alone (it should give the 64.55 saved when that run was trained), the TCN alone, and the
weight the live system uses (0.1). The blend weight is chosen for this test, so this does not describe the labels the assistant
shows live, which get no true boundaries.

    python -m scripts.recognition.pipeline_on_segments --score    # GPU, about 1-2 h on 6 GB; resumes -> probs/<video>.npz
    python -m scripts.recognition.pipeline_on_segments --select   # CPU, one look at test -> output/assist/pipeline_segments/result.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from monitor.settings import ROOT

OUT = ROOT / "output/assist/pipeline_segments"
WEIGHTS_FILE = ROOT / "output/official_split_eval/vjepa2/vjft_hand_k12_p10.pt"
PAPER_BEST = 65.25
VJEPA_ALONE_SAVED = {"val": 62.45, "test": 64.55}
LIVE_WEIGHT, LIVE_STRENGTH, LIVE_SMOOTHING = 0.1, 1.5, 0.01    # online_vjepa_prefix/online_vjepa_test.json
BLEND_WEIGHTS = tuple(round(i / 10, 1) for i in range(11))


def tcn_on_segment(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    """(actions) mean over the segment's own frames; ``frames`` is (T, 1 + actions) with background in column 0."""
    part = frames[max(0, start):min(len(frames) - 1, end) + 1, 1:]
    return (part / part.sum(1, keepdims=True)).mean(0)


def pick_on_val(grid: list[dict]) -> dict:
    """The best row on val; a tie goes to the smaller prior, then the smaller weight. Rows may not hold test scores."""
    if any("test_top1" in row for row in grid):
        raise ValueError("settings are chosen on val only")
    return max(grid, key=lambda r: (r["val_top1"], -r["strength"], -r["w"]))


def guard_single_look(result: Path) -> None:
    """Test is scored once: a saved result stops a second run."""
    if result.is_file():
        raise FileExistsError(f"{result} exists: test has already been scored")


def score(batch: int, workers: int) -> None:
    """V-JEPA 2 on every val and test segment, one file per video, so an interrupted run resumes."""
    import torch

    from scripts.data.tas_labels import load_mapping
    from scripts.features import vjepa2_finetune as ft
    from scripts.features.finetune_videomae_ar import TTA_FRACS
    from scripts.features.vjepa2_recognition import SPLITS, segments

    folder = OUT / "probs"
    folder.mkdir(parents=True, exist_ok=True)
    actions = load_mapping()
    model = ft.Finetuned(12, ft.SIZE).cuda()
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cuda", weights_only=True), strict=False)
    args = SimpleNamespace(crop="hand", size=ft.SIZE, context=0.0, workers=workers)
    rids = list(SPLITS["val"]) + list(SPLITS["test"])
    for i, rid in enumerate(rids):
        path = folder / f"{rid}.npz"
        if path.is_file():
            continue
        rows = segments(rid, len(ft.Frames(rid, ft.CACHE)), actions)
        items = [(rid, s, e, label, f) for _, s, e, label in rows for f in TTA_FRACS]
        p = ft.predict(model, items, args, batch)
        p = p.reshape(-1, len(TTA_FRACS), p.shape[1]).mean(1)
        np.savez(path, keys=np.array([r[0] for r in rows]), labels=np.array([r[3] for r in rows]), probs=p)
        print(json.dumps({"video": rid, "done": i + 1, "of": len(rids), "segments": len(rows)}), flush=True)


def _split(rids: list[str], cfg: dict, actions: list[str]) -> dict:
    """Labels, keys, and per segment the two models' 74 action probabilities."""
    from assist.extract.run_ar import from_cache

    keys, labels, clip, tcn = [], [], [], []
    for rid in rids:
        saved = np.load(OUT / "probs" / f"{rid}.npz")
        frames = from_cache(rid, cfg, actions)
        for key in saved["keys"]:
            _, start, end = str(key).split("|")
            tcn.append(tcn_on_segment(frames, int(start), int(end)))
        keys += [str(k) for k in saved["keys"]]
        labels.append(saved["labels"])
        clip.append(saved["probs"])
    return {"keys": keys, "labels": np.concatenate(labels), "vjepa": np.concatenate(clip), "tcn": np.stack(tcn)}


def select() -> None:
    from assist.common import load_config
    from assist.industreal import SPLITS
    from scripts.data.tas_labels import load_mapping
    from scripts.recognition.fuse_recognition import fused_scores
    from scripts.recognition.online_vjepa_end import decode_filter
    from scripts.recognition.sequence_recognition import SMOOTHING, STRENGTHS, transition_counts, videos
    from scripts.segmentation.run_official_split_eval import TRAIN

    result = OUT / "result.json"
    guard_single_look(result)
    cfg, actions = load_config(), load_mapping()
    data = {split: _split(SPLITS[split], cfg, actions) for split in ("val", "test")}
    first, pairs = transition_counts(TRAIN, actions)
    groups = {split: videos(data[split]["keys"]) for split in data}

    def top1(split: str, w: float, strength: float, smoothing: float) -> float:
        scores = fused_scores(data[split], ("vjepa",), w)
        pred = decode_filter(scores, groups[split], first, pairs, strength, smoothing)
        return round(float((pred == data[split]["labels"]).mean() * 100), 2)

    grid = [{"w": w, "strength": lam, "smoothing": alpha, "val_top1": top1("val", w, lam, alpha)}
            for w, lam, alpha in itertools.product(BLEND_WEIGHTS, STRENGTHS, SMOOTHING)]
    best = pick_on_val(grid)
    alone = {split: top1(split, 1.0, 0.0, SMOOTHING[0]) for split in data}
    if abs(alone["val"] - VJEPA_ALONE_SAVED["val"]) > 0.5 or abs(alone["test"] - VJEPA_ALONE_SAVED["test"]) > 0.5:
        print(json.dumps({"warning": "V-JEPA 2 alone does not match the saved run", "now": alone,
                          "saved": VJEPA_ALONE_SAVED}), flush=True)
    live_strength = round(LIVE_WEIGHT * LIVE_STRENGTH, 3)         # online_vjepa scales its prior by w
    report = {
        "protocol": "IndustReal action recognition (true segments, 74 actions); causal TCN + action-beginning V-JEPA 2, "
                    "label fixed at the segment's end, order prior over earlier actions; chosen on val, test once",
        "segments": {split: len(data[split]["labels"]) for split in data},
        "chosen_on_val": best,
        "test_top1": top1("test", best["w"], best["strength"], best["smoothing"]),
        "paper_best": PAPER_BEST,
        "for_comparison": {
            "tcn_alone": {split: top1(split, 0.0, 0.0, SMOOTHING[0]) for split in data},
            "vjepa_alone": alone, "vjepa_alone_saved": VJEPA_ALONE_SAVED,
            "live_weight": {"w": LIVE_WEIGHT, "strength": live_strength, "smoothing": LIVE_SMOOTHING,
                            **{split: top1(split, LIVE_WEIGHT, live_strength, LIVE_SMOOTHING) for split in data}}},
        "val_grid": grid,
    }
    result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "val_grid"}, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.score:
        score(args.batch, args.workers)
    if args.select:
        select()


if __name__ == "__main__":
    main()
