"""IndustReal assembly state detector (ASD) -> boxes + state labels (``output/assist/<rid>/asd.jsonl``).

``--source ours`` reads our own detector's predictions (``assist/extract/train_asd.py``, val + test) and writes
``asd_ours.jsonl`` instead.

``--source release`` uses the officially published ASD predictions (YOLOv8 trained on IndustReal + synthetic data,
one CSV per test video). ``--source weights --split val`` uses our run of IndustReal's published weights on the val
videos (``train_asd --predict --weights``), so val can compare against the same detector. That run reads the compressed
videos and only approximates the published predictions, so a confidence cutoff and a shift were set on 5 test videos,
with no labels, to match them: 99 % of the detections, 93 % the same frames, 98 % the same class, 88 % of the engine's
step completions within 10 s.

Keyframes are taken at ``--fps`` (default 2) as an online detector would run. Rows: ``{f, t, state, conf, box}`` with
the box as ``[x, y, w, h]`` in pixels and ``state`` an 11-character string or ``"error_state"``.

    python -m assist.extract.run_asd --video all
    python -m assist.extract.run_asd --source weights --split val
"""
from __future__ import annotations

import argparse
import json

from assist.common import ROOT, load_config, write_jsonl
from assist.industreal import SPLITS, asd_release

FOLDERS = {"ours": ROOT / "output/assist/asd_ours/predictions", "weights": ROOT / "output/assist/asd_released/predictions"}
WEIGHTS_CONF_SHIFT = 0.08                # our run of the published weights reads this much lower than their predictions


def jsonl_rows(rows: list[dict], video_fps: float, source: str) -> list[dict]:
    shift = WEIGHTS_CONF_SHIFT if source == "weights" else 0.0
    return [{"f": r["frame"], "t": round(r["frame"] / video_fps, 2), "state": r["state"],
             "conf": min(1.0, r["conf"] + shift), "box": r["box"]} for r in rows]


def keyframes(rows: list[dict], fps: float, video_fps: float) -> list[dict]:
    """The detection at (or the first one after) every 1/fps seconds."""
    step = max(1, round(video_fps / fps))
    out, next_frame = [], 0
    for r in rows:
        if r["frame"] >= next_frame:
            out.append(r)
            next_frame = (r["frame"] // step + 1) * step
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="all", help='Recording id, or "all" (the test split).')
    parser.add_argument("--fps", type=float, default=2.0, help="Keyframe rate the detector runs at.")
    parser.add_argument("--source", choices=("release", "ours", "weights"), default="release")
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    folder = FOLDERS.get(args.source, cfg["paths"]["asd_release"])
    name = "asd_ours.jsonl" if args.source == "ours" else "asd.jsonl"
    rids = SPLITS[args.split] if args.video == "all" else [args.video]
    for rid in rids:
        dest = cfg["paths"]["out"] / rid / name
        if dest.is_file() and not args.force:
            continue
        rows = keyframes(asd_release(rid, folder, cfg["asd_categories"], cfg["asd_min_conf"][args.source]),
                         args.fps, cfg["fps"])
        write_jsonl(dest, jsonl_rows(rows, cfg["fps"], args.source))
        print(json.dumps({"video": rid, "keyframes": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
