"""Spike: flag wrongly mounted parts as "looks unlike any correct example" (anomaly detection, no error examples).

Every assembly-state detection (box + state) is cropped and embedded with frozen DINOv2-small. Reference bank:
the annotated assembly boxes of the 36 train videos (``OD_labels.json``, correct states only). A detection's
score is its distance to the nearest reference of the state the detector reported, divided by that state's
typical distance between references of different train videos. ``k`` scores in a row above ``thr``, with the
hands not covering the assembly, turn those detections into ``error_state``, which the engine already reports
as an incorrectly installed part (``asd_anom.jsonl``, ``--state asdanom``).

Stop rule, fixed before any result: on the 16 val videos (our ASD, the only detector with val predictions) the
chosen setting must find at least 2 of the 5 wrongly mounted parts with no more false alarms per hour than the
same engine without it (``pred_asdours``). Otherwise the spike stops and test is not scored.

    python -m scripts.mistakes.state_anomaly --embed            # GPU server: crops -> output/assist/anomaly/*.npz
    python -m scripts.mistakes.state_anomaly --select           # CPU: grid on val + stop rule
    python -m scripts.mistakes.state_anomaly --test             # CPU: only if val passed; test scored once
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.industreal import SPLITS
from monitor.settings import ROOT

OUT = ROOT / "output/assist/anomaly"
LABELS = ROOT / "data/evaluation/official"
VIDEOS = ROOT / "data/generated/industreal/videos"
ASD = {"val": "asd_ours.jsonl", "test": "asd.jsonl"}       # state detector per split (our ASD on val)
STATE = {"val": "asdours", "test": "asd"}
REFS_PER_STATE = 300
MARGIN = 0.1                                               # extra context around each box
THRESHOLDS = (1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
RUNS = (2, 3, 4)                                           # k: detections in a row (engine needs state_stable = 2)


# ---------------------------------------------------------------------------------------------- GPU (server)
def _crop_boxes(frame: np.ndarray, box) -> tuple[float, float, float, float]:
    """Pixel box (x, y, w, h) with margin -> fractions (x1, y1, x2, y2) for ``_view_tensor``."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x1, y1 = max(0.0, x - MARGIN * bw), max(0.0, y - MARGIN * bh)
    x2, y2 = min(w, x + (1 + MARGIN) * bw), min(h, y + (1 + MARGIN) * bh)
    return x1 / w, y1 / h, x2 / w, y2 / h


def _embed_video(model, rid: str, items: list[tuple[int, list[float]]]) -> np.ndarray:
    """(frame, pixel box) per item, in any order -> (len(items), 768) DINOv2 CLS + mean patch."""
    import cv2

    from scripts.features.extract_ar_frame_features import _embed, _view_tensor

    want = defaultdict(list)
    for i, (f, box) in enumerate(items):
        want[f].append((i, box))
    out = np.zeros((len(items), 768), np.float32)
    cap, f, batch, idx = cv2.VideoCapture(str(VIDEOS / f"{rid}.mp4")), 0, [], []
    last = max(want) if want else -1
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for i, box in want.get(f, []):
            batch.append(_view_tensor(frame, _crop_boxes(frame, box), (224, 392)))
            idx.append(i)
        if len(batch) >= 64:
            out[idx] = _embed(model, np.stack(batch))
            batch, idx = [], []
        f += 1
    if batch:
        out[idx] = _embed(model, np.stack(batch))
    cap.release()
    return out


def embed() -> None:
    from transformers import AutoModel

    from monitor.settings import DINO_V2_DIR

    model = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    refs = defaultdict(list)                                # state -> [(rid, frame, box)]
    for rid in SPLITS["train"]:
        d = json.loads((LABELS / rid / "OD_labels.json").read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in d["categories"]}
        frames = {im["id"]: int(im["file_name"].split(".")[0]) for im in d["images"]}
        for a in d["annotations"]:
            if names[a["category_id"]].isdigit():
                refs[names[a["category_id"]]].append((rid, frames[a["image_id"]], a["bbox"]))
    keep = []
    for state, rows in refs.items():
        for j in sorted(rng.choice(len(rows), min(REFS_PER_STATE, len(rows)), replace=False)):
            keep.append((state, *rows[j]))
    feats, states, rids = [], [], []
    for rid in sorted({k[1] for k in keep}):
        mine = [k for k in keep if k[1] == rid]
        feats.append(_embed_video(model, rid, [(f, box) for _, _, f, box in mine]))
        states += [k[0] for k in mine]
        rids += [rid] * len(mine)
        print(json.dumps({"refs": rid, "n": len(mine)}), flush=True)
    np.savez(OUT / "refs.npz", feats=np.concatenate(feats), states=np.array(states), rids=np.array(rids))
    for split in ("val", "test"):
        out = {}
        for rid in SPLITS[split]:
            rows = read_jsonl(ROOT / "output/assist" / rid / ASD[split])
            out[rid] = _embed_video(model, rid, [(r["f"], r["box"]) for r in rows])
            print(json.dumps({split: rid, "n": len(rows)}), flush=True)
        np.savez(OUT / f"{split}.npz", **out)


# ---------------------------------------------------------------------------------------------- CPU (laptop)
def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def reference_bank() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """state -> unit embeddings, and state -> median distance to the nearest reference of another train video."""
    r = np.load(OUT / "refs.npz")
    feats, states, rids = _unit(r["feats"]), r["states"], r["rids"]
    bank, scale = {}, {}
    for s in np.unique(states):
        m = states == s
        f, v = feats[m], rids[m]
        sim = f @ f.T
        sim[v[:, None] == v[None, :]] = -1.0                 # never compare with frames of the same video
        nn = 1.0 - sim.max(1)
        ok = sim.max(1) > -1.0
        bank[str(s)] = f
        scale[str(s)] = float(np.median(nn[ok])) if ok.any() else float("nan")
    return bank, scale


def scores(split: str, bank: dict, scale: dict) -> dict[str, np.ndarray]:
    """rid -> anomaly score per detection row (nan when the state has no references)."""
    q = np.load(OUT / f"{split}.npz")
    out = {}
    for rid in SPLITS[split]:
        rows, f = read_jsonl(ROOT / "output/assist" / rid / ASD[split]), _unit(q[rid])
        s = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            if r["state"] in bank and np.isfinite(scale[r["state"]]):
                s[i] = (1.0 - float((bank[r["state"]] @ f[i]).max())) / scale[r["state"]]
        out[rid] = s
    return out


def _covered(rid: str, rows: list[dict], cfg: dict) -> np.ndarray:
    from assist.run_engine import hand_box_cover

    hands = read_jsonl(ROOT / "output/assist" / rid / "hands.jsonl")
    t = np.array([h["t"] for h in hands])
    return np.array([hand_box_cover(hands[int(np.abs(t - r["t"]).argmin())]["hands"], r["box"])
                     >= cfg["engine"]["hand_occlusion_iou"] for r in rows])


def flag(rows: list[dict], score: np.ndarray, covered: np.ndarray, thr: float, k: int) -> list[dict]:
    """Rows with ``state`` replaced by ``error_state`` inside every run of >= k anomalous, visible detections."""
    bad = np.nan_to_num(score, nan=0.0) > thr
    bad &= ~covered
    out, i = [dict(r) for r in rows], 0
    while i < len(rows):
        j = i
        while j < len(rows) and bad[j]:
            j += 1
        if j - i >= k:
            for m in range(i, j):
                out[m]["state"] = "error_state"
        i = max(j, i + 1)
    return out


def _engine_eval(split: str, run_state: str) -> dict:
    from assist.engine.graph import load_tasks
    from assist.evaluation.eval import evaluate
    from assist.run_engine import run

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    for rid in SPLITS[split]:
        write_json(cfg["paths"]["out"] / rid / f"engine_pred_{run_state}.json", run(rid, "pred", run_state, cfg, tasks))
    r = evaluate(f"pred_{run_state}", SPLITS[split], cfg, tasks)
    a = r["rows"]["all"]
    return {"precision": round(100 * (a["precision"] or 0), 1), "recall": round(100 * (a["recall"] or 0), 1),
            "found": a["tp"], "alerts": a["alerts"], "mistakes": a["gt"],
            "install_found": r["rows"]["install"]["tp"], "install": r["rows"]["install"]["gt"],
            "false_alarms_per_h": round(a["false_alarms_per_h"], 1)}


def _write(split: str, sc: dict, thr: float, k: int, cfg: dict, cover: dict) -> None:
    for rid in SPLITS[split]:
        rows = read_jsonl(ROOT / "output/assist" / rid / ASD[split])
        write_jsonl(ROOT / "output/assist" / rid / "asd_anom.jsonl", flag(rows, sc[rid], cover[rid], thr, k))


def _covers(split: str, cfg: dict) -> dict:
    return {rid: _covered(rid, read_jsonl(ROOT / "output/assist" / rid / ASD[split]), cfg) for rid in SPLITS[split]}


def select() -> None:
    cfg = load_config()
    bank, scale = reference_bank()
    sc, cover = scores("val", bank, scale), _covers("val", cfg)
    base = _engine_eval("val", STATE["val"])
    grid = []
    for thr in THRESHOLDS:
        for k in RUNS:
            _write("val", sc, thr, k, cfg, cover)
            grid.append({"thr": thr, "k": k, **_engine_eval("val", "asdanom")})
            print(json.dumps(grid[-1]), flush=True)
    passed = [g for g in grid if g["install_found"] >= 2 and g["false_alarms_per_h"] <= base["false_alarms_per_h"]]
    best = max(passed, key=lambda g: (g["found"], -g["false_alarms_per_h"], g["thr"])) if passed else None
    report = {"rule": "val: >= 2 of the wrongly mounted parts found, false alarms / h <= the engine without it",
              "baseline_val": base, "passed": best is not None, "chosen": best, "grid": grid,
              "score_quantiles_val": {q: float(np.nanquantile(np.concatenate(list(sc.values())), q))
                                      for q in (0.5, 0.9, 0.99)}}
    write_json(OUT / "select_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))


def test() -> None:
    report = json.loads((OUT / "select_val.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise SystemExit("val did not pass the stop rule: test is not scored")
    cfg = load_config()
    bank, scale = reference_bank()
    c = report["chosen"]
    _write("test", scores("test", bank, scale), c["thr"], c["k"], cfg, _covers("test", cfg))
    result = {"chosen_on_val": c, "baseline_test": _engine_eval("test", STATE["test"]),
              "anomaly_test": _engine_eval("test", "asdanom")}
    write_json(OUT / "test.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag_ in ("--embed", "--select", "--test"):
        parser.add_argument(flag_, action="store_true")
    args = parser.parse_args()
    if args.embed:
        embed()
    if args.select:
        select()
    if args.test:
        test()


if __name__ == "__main__":
    main()
