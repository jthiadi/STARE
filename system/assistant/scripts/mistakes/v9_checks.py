"""Teammate's v9 visual checks (branch ``psr``, folder ``PSR``) as a check on installs in our engine.

v9 finds the assembly in every 2nd frame and two DINOv2-base models give P(attached) for 10 parts, plus a view quality.
Here: after an install, if the averaged check says the part is absent for ``looks`` good views in a row, the engine raises
"installed wrong". Hands on the assembly pause it, as for every other state check. Steps are still completed by the
released state detector, so step scores cannot change. The engine itself is not changed.

v9 read the original frames; this reads our compressed videos, so --reproduce compares its step scores with its saved ones.

Rule, fixed before any result, with the live actions and the released detector: at least 2 more wrong installs found, no
more new false alarms than new finds, step F1 and POS not lower. v9's checks were trained on train + val, so val is only
used to choose among SETTINGS and is optimistic: if no setting passes even there, test is not scored. Test is scored once
and only test decides. This is one more look at a test set that has already been used for other experiments.

    python -m scripts.mistakes.v9_checks --link                      # GPU server: their folder layout, linked to our labels
    python -m scripts.mistakes.v9_checks --crops --shard 0 --shards 4  # GPU server -> PSR/cache/crops_hr/<split>/
    VBACK=base VRES=336 python -m scripts.mistakes.v9_checks --verify val weights/verif_base_r336.pt   # their script
    VBACK=base VRES=336 VEPOCHS=6 VUNFREEZE=6 python -m scripts.mistakes.v9_checks --verify val weights/verif_base_r336_e6u6.pt
    python -m scripts.mistakes.v9_checks --reproduce val             # their step scores, from our videos
    python -m scripts.mistakes.v9_checks --select                    # CPU: val grid + rule -> output/assist/v9_checks/select_val.json
    python -m scripts.mistakes.v9_checks --test                      # CPU: only if val passed -> test.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import runpy
import shutil
import sys
import types
from pathlib import Path

import numpy as np

from assist.common import load_config, write_json
from assist.industreal import SPLITS
from monitor.settings import ROOT

V9 = Path(os.environ.get("V9_DIR", "../industreal_extra/v9/PSR"))
DATASET = Path(os.environ.get("INDUSTREAL_DATA", V9.parent / "dataset"))
VIDEOS = ROOT / "data/generated/industreal/videos"
OUT = ROOT / "output/assist/v9_checks"
TAGS = ("verif_base_r336", "verif_base_r336_e6u6")
FOLDERS = {"train": "train/train_annotations", "val": "validation/val_annotations", "test": "test/test_annotations"}
RES = 336
MIN_VIEW = 0.1                  # v9's own q_min
SETTINGS = [{"thr": thr, "window_s": window, "looks": looks}          # looks come at 5 per second
            for thr, window, looks in itertools.product((0.5, 0.7, 0.9), (10, 30), (5, 10, 25))]


def wrong_rows(frames: np.ndarray, first: np.ndarray, second: np.ndarray, quality: np.ndarray,
               fps: float = 10) -> list[tuple[float, list[float], float]]:
    """(time, P(part reads absent) per component, 1 for a usable view else 0). The base is never judged."""
    if not len(frames) == len(first) == len(second) == len(quality):
        raise ValueError("the two checks must cover the same frames")
    absent = 1 - (first.astype(np.float32) + second.astype(np.float32)) / 2
    return [(round(float(f) / fps, 2), [0.0] + a.tolist(), 1.0 if q >= MIN_VIEW else 0.0)
            for f, a, q in zip(frames, absent, quality)]


def same_steps(plain: list[dict], checked: list[dict]) -> bool:
    """The check may close a pending install check, which adds a "verified" note; it must not move any step."""
    steps = lambda log: [e for e in log if e["status"] != "verified"]          # noqa: E731
    return steps(plain) == steps(checked)


# ---------------------------------------------------------------------------------------------- GPU (server)
def link(dataset: Path = DATASET, source: Path = ROOT / "data/evaluation/official", splits: dict = SPLITS) -> None:
    """Their code lists videos and reads labels from folders; copy our small label files there. Frames come from videos."""
    for split, rids in splits.items():
        for rid in rids:
            target = dataset / FOLDERS[split] / rid
            target.mkdir(parents=True, exist_ok=True)
            for label in (source / rid).glob("*.csv"):
                shutil.copy(label, target / label.name)


def crops(shard: int, shards: int) -> None:
    """Their extract_crops.py + extract_crops_hr.py, reading video frames: same localizer, boxes, sizes and JPEG quality."""
    import time

    import cv2

    sys.path.insert(0, str(V9))
    from psr import vision

    vision.MODEL_ID = str(ROOT / "models/dinov2-small")
    finder = vision.VisionVerifier(bank="exemplar")
    for split in ("val", "test"):
        out = vision.CACHE / "crops_hr" / split
        out.mkdir(parents=True, exist_ok=True)
        for rid in SPLITS[split][shard::shards]:
            target = out / f"{rid}.npz"
            if target.exists():
                continue
            start, finder.prev_box = time.time(), None
            video = cv2.VideoCapture(str(VIDEOS / f"{rid}.mp4"))
            frames, jpgs, views, boxes, index = [], [], [], [], -1
            while video.grab():
                index += 1
                if index % 2:
                    continue
                bgr = video.retrieve()[1]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                _, patches = finder.enc.encode(rgb, finder.FRAME_SIZE)
                box, _ = finder.localize(patches)
                if box is None:
                    continue
                view, _ = finder.view_quality(vision.crop(rgb, box, pad=0.1))
                roi = cv2.resize(vision.crop(bgr, box, pad=0.1), (RES, RES), interpolation=cv2.INTER_AREA)
                frames.append(index)
                jpgs.append(cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                views.append(view)
                boxes.append(box)
            video.release()
            images = np.empty(len(jpgs), dtype=object)
            images[:] = jpgs
            np.savez(target.with_suffix(".tmp.npz"), t=np.array(frames), jpg=images, q=np.array(views, np.float32),
                     box=np.array(boxes, np.float32))
            target.with_suffix(".tmp.npz").rename(target)
            print(json.dumps({"split": split, "video": rid, "views": len(frames), "of": index + 1,
                              "seconds": round(time.time() - start)}), flush=True)


def verify(split: str, weights: str) -> None:
    """Their predict_verifier.py, in smaller batches so it fits a 6 GB GPU. Batch size does not change the outputs."""
    sys.path.insert(0, str(V9))
    import train_verifier

    train_verifier.predict.__defaults__ = (32,)
    sys.argv = ["predict_verifier.py", split, weights]
    runpy.run_path(str(V9 / "predict_verifier.py"), run_name="__main__")


def reproduce(split: str) -> None:
    """Their eval_psr.py with actual actions, frame count from the video. Saved scores: PSR/results/v9_gtAR_<split>.json."""
    import cv2

    sys.path.insert(0, str(V9))
    import psr.core as core

    def frame_paths(_split: str, video: str) -> list:
        capture = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4"))
        return [None] * int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    core.frame_paths = frame_paths
    vision = types.ModuleType("psr.vision")          # eval_psr.py only wants the cache folder: no GPU packages needed
    vision.CACHE = Path(os.environ.get("PSR_CACHE", V9 / "cache"))
    sys.modules["psr.vision"] = vision
    sys.argv = ["eval_psr.py", "--split", split, "--ar", "gt", "--out", str(V9 / "results_from_videos")]
    runpy.run_path(str(V9 / "eval_psr.py"), run_name="__main__")


# ---------------------------------------------------------------------------------------------- the decision (CPU)
def load_rows(split: str, rid: str) -> list[tuple[float, list[float], float]]:
    first, second = (np.load(OUT / f"{tag}_{split}" / f"{rid}.npz") for tag in TAGS)
    if not np.array_equal(first["t"], second["t"]):
        raise ValueError(f"{rid}: the two checks cover different frames")
    return wrong_rows(first["t"], first["p"], second["p"], first["q"])


def _run_all(split: str, setting: dict, rows: dict, cfg: dict, tasks: dict) -> dict:
    from assist.run_engine import run
    from scripts.mistakes.part_check import ACTIONS

    return {rid: run(rid, ACTIONS, "asd", cfg, tasks, part=(rows[rid], setting)) for rid in SPLITS[split]}


def agreement(rows: dict, cfg: dict, tasks: dict) -> dict:
    """Does each output belong to the part we think it does? Mean P(absent) with the part truly on / off, val labels."""
    from assist.industreal import psr_states, task_of
    from scripts.mistakes.part_state import LABELS, state_at

    on, off = [[] for _ in cfg["components"]], [[] for _ in cfg["components"]]
    for rid in SPLITS["val"]:
        changes, initial = psr_states(rid, LABELS), list(tasks[task_of(rid)].initial_state)
        for t, wrong, _ in rows[rid]:
            for k, value in enumerate(state_at(changes, initial, round(t * cfg["fps"]))):
                (on if value == 1 else off)[k].append(wrong[k])
    return {name: {"absent_when_on": round(float(np.mean(on[k])), 3) if on[k] else None,
                   "absent_when_off": round(float(np.mean(off[k])), 3) if off[k] else None}
            for k, name in enumerate(cfg["components"])}


def select() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data
    from assist.run_engine import run
    from scripts.mistakes.part_check import ACTIONS, choose, scored

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "weights")
    rows = {rid: load_rows("val", rid) for rid in SPLITS["val"]}
    print(json.dumps(agreement(rows, cfg, tasks), indent=1), flush=True)
    plain = {rid: run(rid, ACTIONS, "asd", cfg, tasks) for rid in SPLITS["val"]}
    base = scored(f"{ACTIONS}_asd", plain, cfg, tasks, data)
    print(json.dumps({"baseline": base}), flush=True)
    grid = []
    for setting in SETTINGS:
        results = _run_all("val", setting, rows, cfg, tasks)
        for rid in SPLITS["val"]:
            if not same_steps(plain[rid]["log"], results[rid]["log"]):
                raise RuntimeError(f"{rid}: the check changed a step's status")
        grid.append({**setting, **scored(f"{ACTIONS}_v9check", results, cfg, tasks, data)})
        print(json.dumps(grid[-1]), flush=True)
    best = choose(base, grid)
    report = {"rule": "at least 2 more wrong installs, no more new false alarms than new finds, step F1 and POS not lower",
              "note": "v9's checks were trained on val, so this is optimistic and only chooses a setting",
              "baseline_val": base, "passed": best is not None, "chosen": best, "grid": grid}
    write_json(OUT / "select_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=1))


def test() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data
    from assist.run_engine import run
    from scripts.mistakes import part_check

    report = json.loads((OUT / "select_val.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise SystemExit("no setting passed on val: test is not scored")
    chosen = report["chosen"]
    setting = {k: chosen[k] for k in ("thr", "window_s", "looks")}
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "release")
    part_check.SPLIT = "test"
    actions = part_check.ACTIONS
    rows = {rid: load_rows("test", rid) for rid in SPLITS["test"]}
    plain = {rid: run(rid, actions, "asd", cfg, tasks) for rid in SPLITS["test"]}
    base = part_check.scored(f"{actions}_asd", plain, cfg, tasks, data)
    results = _run_all("test", setting, rows, cfg, tasks)
    for rid in SPLITS["test"]:
        if not same_steps(plain[rid]["log"], results[rid]["log"]):
            raise RuntimeError(f"{rid}: the check changed a step's status")
    check = part_check.scored(f"{actions}_v9check", results, cfg, tasks, data)
    result = {"chosen_on_val": setting, "baseline_test": base, "v9check_test": check,
              "passed": part_check.passes(base, check)}
    write_json(OUT / "test.json", result)
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("link", "crops", "select", "test"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reproduce", choices=("val", "test"))
    parser.add_argument("--verify", nargs=2, metavar=("SPLIT", "WEIGHTS"))
    args = parser.parse_args()
    if args.verify:
        verify(*args.verify)
    if args.link:
        link()
    if args.crops:
        crops(args.shard, args.shards)
    if args.reproduce:
        reproduce(args.reproduce)
    if args.select:
        select()
    if args.test:
        test()


if __name__ == "__main__":
    main()
