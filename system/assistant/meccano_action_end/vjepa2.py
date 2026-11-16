"""MECCANO action recognition with the IndustReal 77.2 % recipe unchanged: V-JEPA 2 ViT-L with its last 8 or 12
layers fine-tuned, work-area or hand crop, 10 epochs (``scripts.features.vjepa2_finetune``). Official protocol:
trained on the 11 train videos, epoch chosen on the 2 val videos, val and test scored with 3 samplings;
``recognition --select`` then offers the runs next to the MECCANO VideoMAE, each run a member as on IndustReal.

    python -m meccano_action_end.vjepa2 --cache                    # 960 x 540 JPEG frames of the 20 videos
    python -m meccano_action_end.vjepa2 --hands                    # MediaPipe hand landmarks (hand crop), CPU
    python -m meccano_action_end.vjepa2 --crop area --layers 8
    python -m meccano_action_end.vjepa2 --crop hand --layers 8
    python -m meccano_action_end.vjepa2 --crop area --layers 12 --batch 4 --accum 4
    python -m meccano_action_end.vjepa2 --crop hand --layers 12 --batch 4 --accum 4

Needs the VideoMAE frame index (``output/meccano/frames/*.idx.npy``), which fixes each video's frame count.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from scripts.features import vjepa2_finetune as ft
from meccano_action_end import data

CACHE = data.OUT / "vjepa_frames_960"
HANDS = data.OUT / "hands"                          # MediaPipe landmarks per video (mistsense.extract_video)
VJ_DIR = data.OUT / "vjepa"


def items(split: str, fracs: tuple) -> tuple[list[tuple], list[str]]:
    """Segments of ``data.clips`` (same order) as fine-tuning items, ``fracs`` samplings each."""
    clips = data.clips(data.split(split))
    return ([(v, s, e, label, f) for v, s, e, label in clips for f in fracs],
            [f"{v}|{s}|{e}" for v, s, e, _ in clips])


def _cache(vid: str) -> str:
    return ft.build_cache(vid, data.video_path(vid), CACHE)


def _hands(vid: str) -> str:
    from scripts.mistakes.mistsense import extract_video

    return str(extract_video(vid, data.video_path(vid), HANDS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Build the 960 x 540 frame cache (8 processes).")
    parser.add_argument("--hands", action="store_true", help="MediaPipe hand landmarks of the 20 videos (7 processes).")
    parser.add_argument("--crop", choices=("area", "hand"), default="area")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=float, default=0.0)
    parser.add_argument("--prefix", type=float, default=0.0)   # added: vjepa2_finetune gained --prefix on Nov 4; 0 = whole segment, as on IndustReal
    args = parser.parse_args()
    videos = data.split("train") + data.split("val") + data.split("test")
    if args.cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(8) as pool:
            list(pool.map(_cache, videos))
        return
    if args.hands:
        with ProcessPoolExecutor(7) as pool:
            list(pool.map(_hands, videos))
        return
    VJ_DIR.mkdir(parents=True, exist_ok=True)
    args.size = ft.SIZE
    ft.train(args, SimpleNamespace(items=items, cache=CACHE, hands=HANDS, classes=len(data.actions()) - 1,
                                   out=VJ_DIR))


if __name__ == "__main__":
    main()
