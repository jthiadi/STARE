"""Causal action recognition -> per-frame top-k labels + confidences (``output/assist/<rid>/ar.jsonl``).

The model is the causal verb x noun TCN ensemble (each frame's output only uses frames up to it; the
VideoMAE clips end at the frame). ``P(action) ~ P(verb) * P(noun)`` over the 74 official actions plus
background, renormalised per frame. No decoding over the future (no Viterbi), so the labels are online.

    python -m assist.extract.run_ar --video all              # from the cached causal TCN outputs (CPU)
    python -m assist.extract.run_ar --video 18_assy_2_5 --force

Each video is skipped if its JSONL exists (resumable). ``--compute`` (GPU, run on the server) recomputes
the features and the TCN from the video instead of reading the cache, and logs fps and ms/frame.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from assist.common import load_config, write_jsonl
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping
from scripts.segmentation.combine_verb_noun_actions import joint_scores, run_classes


def action_probs(pv: np.ndarray, pn: np.ndarray, verbs: list[str], nouns: list[str],
                 actions: list[str]) -> np.ndarray:
    """(T, len(actions)) probabilities, rows sum to 1; column 0 is background."""
    s = joint_scores(pv, pn, verbs, nouns, actions)
    s = np.exp(s - s.max(1, keepdims=True))
    return s / s.sum(1, keepdims=True)


def topk_rows(probs: np.ndarray, actions: list[str], fps: float, k: int) -> list[dict]:
    idx = np.argsort(-probs, 1)[:, :k]
    return [{"f": i, "t": round(i / fps, 2), "top": [[actions[j], round(float(probs[i, j]), 3)] for j in row]}
            for i, row in enumerate(idx)]


def from_cache(rid: str, cfg: dict, actions: list[str]) -> np.ndarray:
    """Test videos: the ensemble's test outputs; train / val videos: out-of-fold outputs (each video predicted
    by the fold models that did not train on it)."""
    runs = {t: Path(cfg["paths"]["causal_tcn"].format(target=t)) for t in ("verb", "noun")}
    if (runs["verb"] / rid / "test_probs.npy").is_file():
        pv, pn = (np.load(runs[t] / rid / "test_probs.npy") for t in ("verb", "noun"))
    else:
        pv, pn = (np.load(runs[t] / "oof_probs.npz")[rid] for t in ("verb", "noun"))
    return action_probs(pv, pn, run_classes(runs["verb"]), run_classes(runs["noun"]), actions)


def compute(rid: str, cfg: dict, actions: list[str]) -> tuple[np.ndarray, dict]:
    """Full causal pipeline from the video on the GPU; returns probabilities and timing."""
    from assist.extract.causal_pipeline import run_pipeline   # imports torch, only needed on the GPU server
    t0 = time.perf_counter()
    pv, pn, verbs, nouns, timing = run_pipeline(rid, cfg)
    probs = action_probs(pv, pn, verbs, nouns, actions)
    seconds = time.perf_counter() - t0
    timing.update({"frames": len(probs), "seconds": round(seconds, 1), "fps": round(len(probs) / seconds, 1),
                   "ms_per_frame": round(1000 * seconds / len(probs), 1)})
    return probs, timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="all", help='Recording id, or "all" (the test split).')
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--compute", action="store_true", help="Recompute from the video on the GPU.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    actions = load_mapping()
    rids = SPLITS[args.split] if args.video == "all" else [args.video]
    for rid in rids:
        dest = cfg["paths"]["out"] / rid / "ar.jsonl"
        if dest.is_file() and not args.force:
            continue
        if args.compute:
            probs, timing = compute(rid, cfg, actions)
            (dest.parent / "ar_timing.json").parent.mkdir(parents=True, exist_ok=True)
            (dest.parent / "ar_timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
            print(json.dumps({"video": rid, **timing}), flush=True)
        else:
            probs = from_cache(rid, cfg, actions)
        write_jsonl(dest, topk_rows(probs, actions, cfg["fps"], args.topk))
        print(json.dumps({"video": rid, "frames": len(probs), "written": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
