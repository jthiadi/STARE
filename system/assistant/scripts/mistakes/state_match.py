"""Does the view look like the state this install should have produced? Trained on correct states only.

Uses the method of Schoonbeek et al., "Supervised Representation Learning towards Generalizable Assembly State
Recognition" (RA-L 2024, https://github.com/TimSchoonbeek/AssemblyStateRecognition): ResNet34 with the supervised
contrastive loss and their intermediate-state change, trained on real and synthetic images of correct states. Each state
has one clean synthetic reference image. No wrongly installed part is used in training, so the shortage of wrong
installs matters less.

Here it only adds alerts: after an install, if ``looks`` views in a row are too unlike the state the procedure expects,
the engine raises "installed wrong". Steps are still completed by the released state detector.

Rule, fixed before any result, against the released detector on the 16 val videos with the live actions: at least 2 more
of the 5 wrong installs found, no more new false alarms than new finds, step F1 and POS not lower. Only then is test
scored, once. Thresholds and settings come from val only.

    python -m scripts.mistakes.state_match --embed    # GPU server -> output/assist/state_match/sims_{val,test}.npz
    python -m scripts.mistakes.state_match --errors   # GPU server: their test on the train + val error images, no test images
    python -m scripts.mistakes.state_match --select   # CPU: val grid + rule -> select_val.json
    python -m scripts.mistakes.state_match --test     # CPU: only if val passed -> test.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from assist.common import load_config, read_jsonl, write_json
from assist.industreal import SPLITS, psr_states, task_of
from monitor.settings import ROOT
from scripts.mistakes.part_check import ACTIONS, choose, scored
from scripts.mistakes.part_state import LABELS, state_at

OUT = ROOT / "output/assist/state_match"
BASE = Path(os.environ.get("STATE_REC", "../industreal_extra/state_rec"))
CODE = Path(os.environ.get("STATE_REC_CODE", "../industreal_extra/AssemblyStateRecognition"))
WEIGHTS = BASE / "runs/resnet34_supcon_isil/checkpoints/best.pth"
MODEL_NAME = "resnet34.a1_in1k"
CACHE = ROOT / "output/vjepa_frames_960"
SCALE = 960 / 1280
SIZE = 224
REAL_MEAN, REAL_STD = (0.608, 0.545, 0.520), (0.172, 0.197, 0.188)        # as in their code
SYNTH_MEAN, SYNTH_STD = (0.838, 0.805, 0.761), (0.134, 0.143, 0.136)
SETTINGS = [{"false_share": f, "window_s": w, "looks": k} for f, w, k in itertools.product((0.01, 0.02, 0.05), (10, 30), (3, 8))]
MIN_VIEWS = 30                    # a state needs this many correct val views to get a threshold


def similarities(rows: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Cosine of every row to every reference: (rows, references)."""
    a = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    b = references / np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-9)
    return a @ b.T


def thresholds(correct: dict[str, np.ndarray], false_share: float) -> dict[str, float]:
    """For each state, the similarity that at most ``false_share`` of its correct views fall below."""
    return {code: float(np.quantile(s, false_share, method="lower")) for code, s in correct.items() if len(s) >= MIN_VIEWS}


def crop_box(box: list[float]) -> tuple[int, int, int, int]:
    """Box (x, y, w, h in 1280 x 720) -> (x1, y1, x2, y2) in cache pixels. Their crops are the box itself, unpadded."""
    x, y, w, h = (v * SCALE for v in box)
    return round(max(x, 0)), round(max(y, 0)), round(min(x + w, 960)), round(min(y + h, 540))


# ---------------------------------------------------------------------------------------------- GPU (server)
def _model():
    import torch

    sys.path.insert(0, str(CODE))
    from models import ContrastiveModel

    net = ContrastiveModel(SimpleNamespace(model=MODEL_NAME, channels=3, hidden=128))
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.use_projection_head(False)
    return net.eval().cuda()


def _tensor(image, mean, std):
    import torch
    from PIL import Image

    img = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    x = torch.from_numpy(np.asarray(img.convert("RGB").resize((SIZE, SIZE)))).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def _embed_images(net, tensors, batch: int = 128) -> np.ndarray:
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch):
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(net(torch.stack(tensors[i:i + batch]).cuda()).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


def references(net) -> tuple[list[str], np.ndarray]:
    """The defined states (their 11-character codes) and each one's reference embedding."""
    from PIL import Image

    names = load_config()["asd_categories"]
    d = json.loads((BASE / "data/anchors/labels.json").read_text(encoding="utf-8"))
    tensors, codes = [], []
    for image, label in zip(d["images"], d["labels"]):
        codes.append(names[int(label)])
        tensors.append(_tensor(Image.open(BASE / "data/anchors/images" / image), SYNTH_MEAN, SYNTH_STD))
    return codes, _embed_images(net, tensors)


def embed() -> None:
    from scripts.features.finetune_videomae_ar import Frames

    net = _model()
    codes, refs = references(net)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        arrays = {"codes": np.array(codes)}
        for rid in SPLITS[split]:
            rows = read_jsonl(ROOT / "output/assist" / rid / "asd.jsonl")
            frames = Frames(rid, CACHE)
            tensors = []
            for r in rows:
                x1, y1, x2, y2 = crop_box(r["box"])
                frame = frames.get([min(r["f"], len(frames) - 1)])[0]
                tensors.append(_tensor(frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)], REAL_MEAN, REAL_STD))
            arrays[rid] = similarities(_embed_images(net, tensors), refs).astype(np.float32)
            print(json.dumps({"split": split, "video": rid, "views": len(rows)}), flush=True)
        np.savez(OUT / f"sims_{split}.npz", **arrays)


def _average_precision(error_sim: np.ndarray, correct_sim: np.ndarray) -> float:
    """Their measure: correct views should be more like the reference than error views. Balanced, so chance is 50."""
    scores = np.concatenate([correct_sim, error_sim])
    positive = np.concatenate([np.ones(len(correct_sim)), np.zeros(len(error_sim))])
    order = np.argsort(-scores)
    hits = positive[order].cumsum()
    return float(100 * (hits[positive[order] == 1] / (np.flatnonzero(positive[order] == 1) + 1)).mean())


def errors_check() -> None:
    """Is our trained model about as good as the paper's? Their test, on train and val recordings only; test is untouched."""
    from PIL import Image

    net = _model()
    codes, refs = references(net)
    names = load_config()["asd_categories"]
    data = BASE / "data"
    d = json.loads((data / "assembly_states_errors/labels.json").read_text(encoding="utf-8"))
    known = {r: s for s in ("train", "val") for r in SPLITS[s]}
    rng = np.random.default_rng(0)
    correct: dict[int, list[str]] = {}
    for split in ("train", "val"):
        c = json.loads((data / split / "labels.json").read_text(encoding="utf-8"))
        for image, label in zip(c["images"], c["labels"]):
            if 0 < label < 23:
                correct.setdefault(int(label), []).append(str(data / split / "images" / image))
    rows = []
    for image, intended, cat in zip(d["images"], d["intended"], d["error_cat"]):
        found = re.search(r"\d\d_(assy|main)_\d+_\d+", image)
        if found and found.group(0) in known and int(intended) in correct and names[int(intended)] in codes:
            rows.append((image, int(intended), int(cat), known[found.group(0)]))
    print(json.dumps({"error_images": len(rows)}), flush=True)
    error_tensors = [_tensor(Image.open(data / "assembly_states_errors/images" / r[0]), REAL_MEAN, REAL_STD) for r in rows]
    correct_paths = [correct[r[1]][rng.integers(len(correct[r[1]]))] for r in rows]
    correct_tensors = [_tensor(Image.open(p), REAL_MEAN, REAL_STD) for p in correct_paths]
    index = np.array([codes.index(names[r[1]]) for r in rows])
    take = np.arange(len(rows))
    error_sim = similarities(_embed_images(net, error_tensors), refs)[take, index]
    correct_sim = similarities(_embed_images(net, correct_tensors), refs)[take, index]
    cats = np.array([r[2] for r in rows])
    groups = np.array([r[3] for r in rows])
    names_of = {1: "missing", 2: "placement", 3: "orientation", 4: "part_level"}
    report = {}
    for group in ("train", "val"):
        sel = groups == group
        report[group] = {names_of[c]: {"images": int((sel & (cats == c)).sum()),
                                       "ap": round(_average_precision(error_sim[sel & (cats == c)], correct_sim[sel & (cats == c)]), 1)}
                         for c in names_of if (sel & (cats == c)).sum()}
    write_json(OUT / "errors_check.json", report)
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------------------------- the decision (CPU)
def _rows(split: str, rid: str, z, cfg: dict) -> list[tuple[float, dict, float]]:
    codes = [str(c) for c in z["codes"]]
    asd = read_jsonl(cfg["paths"]["out"] / rid / "asd.jsonl")
    if len(asd) != len(z[rid]):
        raise ValueError(f"{rid}: {len(asd)} rows but {len(z[rid])} similarities")
    return [(r["t"], dict(zip(codes, s.tolist())), r["conf"]) for r, s in zip(asd, z[rid])]


def correct_views(z, cfg: dict, tasks) -> dict[str, np.ndarray]:
    """For each state, the similarity to its reference of val views annotated as exactly that state."""
    codes = [str(c) for c in z["codes"]]
    out: dict[str, list[float]] = {c: [] for c in codes}
    for rid in SPLITS["val"]:
        changes, init = psr_states(rid, LABELS), list(tasks[task_of(rid)].initial_state)
        for r, s in zip(read_jsonl(cfg["paths"]["out"] / rid / "asd.jsonl"), z[rid]):
            code = "".join(map(str, state_at(changes, init, r["f"])))
            if code in out:
                out[code].append(float(s[codes.index(code)]))
    return {c: np.array(v) for c, v in out.items()}


def _run_all(split: str, z, setting: dict, thr: dict, cfg: dict, tasks) -> dict:
    from assist.run_engine import run

    check = {"thr": thr, "window_s": setting["window_s"], "looks": setting["looks"]}
    return {rid: run(rid, ACTIONS, "asd", cfg, tasks, part=(_rows(split, rid, z, cfg), check)) for rid in SPLITS[split]}


def select() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data
    from assist.run_engine import run

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "weights")
    z = np.load(OUT / "sims_val.npz")
    plain = {rid: run(rid, ACTIONS, "asd", cfg, tasks) for rid in SPLITS["val"]}
    base = scored(f"{ACTIONS}_asd", plain, cfg, tasks, data)
    print(json.dumps({"baseline": base}), flush=True)
    views = correct_views(z, cfg, tasks)
    grid = []
    for setting in SETTINGS:
        thr = thresholds(views, setting["false_share"])
        results = _run_all("val", z, setting, thr, cfg, tasks)
        for rid in SPLITS["val"]:
            if results[rid]["log"] != plain[rid]["log"]:
                raise RuntimeError(f"{rid}: the check changed a step's status")
        grid.append({**setting, "states_checked": len(thr), **scored(f"{ACTIONS}_match", results, cfg, tasks, data)})
        print(json.dumps(grid[-1]), flush=True)
    best = choose(base, grid)
    report = {"rule": "val, released detector: at least 2 more of the 5 wrong installs, no more new false alarms than new "
                      "finds, step F1 and POS not lower", "baseline_val": base, "passed": best is not None, "chosen": best,
              "grid": grid}
    write_json(OUT / "select_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=1))


def test() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data
    from assist.run_engine import run

    report = json.loads((OUT / "select_val.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise SystemExit("val did not pass the rule: test is not scored")
    c = report["chosen"]
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "release")
    thr = thresholds(correct_views(np.load(OUT / "sims_val.npz"), cfg, tasks), c["false_share"])
    z = np.load(OUT / "sims_test.npz")
    from scripts.mistakes import part_check
    part_check.SPLIT = "test"
    plain = {rid: run(rid, ACTIONS, "asd", cfg, tasks) for rid in SPLITS["test"]}
    result = {"chosen_on_val": c, "baseline_test": scored(f"{ACTIONS}_asd", plain, cfg, tasks, data),
              "match_test": scored(f"{ACTIONS}_match", _run_all("test", z, c, thr, cfg, tasks), cfg, tasks, data)}
    write_json(OUT / "test.json", result)
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("embed", "errors", "select", "test"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.embed:
        embed()
    if args.errors:
        errors_check()
    if args.select:
        select()
    if args.test:
        test()


if __name__ == "__main__":
    main()
