"""Online V-JEPA 2 labels for the live assistant (``ar_vjepa.jsonl``, same format as ``ar.jsonl``); see
``assist/extract/online_vjepa.py`` for how a label is made. Steps:

    python -m assist.extract.run_ar_vjepa --plan --split val          # CPU: update windows from the live model
    python -m assist.extract.run_ar_vjepa --plan --split test
    python -m assist.extract.run_ar_vjepa --benchmark                 # GPU server: ms per update and model
    python -m assist.extract.run_ar_vjepa --windows --split val --part 0/2    # GPU server, one part per GPU
    python -m assist.extract.run_ar_vjepa --select                    # CPU: settings chosen on the 16 val videos
    python -m assist.extract.run_ar_vjepa --select-mistakes           # CPU: final weight by val mistake detection
    python -m assist.extract.run_ar_vjepa --variant prefix --apply-rule   # CPU: the action-beginning model's val rule
    python -m assist.extract.run_ar_vjepa --write --split test        # CPU: test scored once + ar_vjepa.jsonl

Files: ``output/assist/online_vjepa/{split}_plan.npz``, ``{split}_members_{i}.npz``, ``benchmark.json``,
``online_vjepa_val.json``, ``online_vjepa_test.json``. Weights: ``release_vjepa/`` (the 77.2 % ensemble), one
temporal sampling per window (the live budget).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from assist.common import load_config, write_json, write_jsonl
from assist.engine.events import build_events
from assist.extract.online_vjepa import frame_scores, shown_probs, updates
from assist.extract.run_ar import from_cache, topk_rows
from assist.industreal import SPLITS, gt_actions
from monitor.settings import ROOT
from scripts.data.tas_labels import load_mapping
from scripts.recognition.sequence_recognition import SMOOTHING, STRENGTHS, WEIGHTS, transition_counts
from scripts.segmentation.run_official_split_eval import TRAIN

OUT = ROOT / "output/assist/online_vjepa"
WEIGHTS_DIR = ROOT / "release_vjepa"
LABELS = "ar_vjepa.jsonl"    # per-video label file and the engine's name for it
ACTIONS = "pred_vjepa"
VARIANTS = {"release": (OUT, WEIGHTS_DIR, LABELS, ACTIONS),
            # the hand k12 model retrained on action beginnings (vjepa2_finetune --prefix 0.1) + the same VideoMAE
            "prefix": (ROOT / "output/assist/online_vjepa_prefix", ROOT / "release_vjepa_prefix",
                       "ar_vjepa_prefix.jsonl", "pred_vjepap")}
EVERY = 5                    # frames between updates (0.5 s at 10 fps)
DELAY = 5                    # a result is shown 0.5 s after its update runs (the compute budget)
BUDGET_MS = 1000 * DELAY / 10
SUBSETS = {"all5": ("vjft_area_k8", "vjft_hand_k8", "vjft_area_k12", "vjft_hand_k12", "vmae_full"),
           "vjft4": ("vjft_area_k8", "vjft_hand_k8", "vjft_area_k12", "vjft_hand_k12"),
           "hand_k12": ("vjft_hand_k12",),
           "k12": ("vjft_area_k12", "vjft_hand_k12"),
           "vmae+hand_k12": ("vmae_full", "vjft_hand_k12"),
           "hand_k12_p10": ("vjft_hand_k12_p10",),
           "vmae+hand_k12_p10": ("vmae_full", "vjft_hand_k12_p10")}


def live_and_events(rid: str, cfg: dict, actions: list[str]) -> tuple[np.ndarray, list]:
    """The live model's (T, 75) probabilities and its events, exactly as the engine builds them."""
    live = from_cache(rid, cfg, actions)
    e = cfg["events"]
    top = live.argmax(1)
    frames = [(f / cfg["fps"], actions[a], float(live[f, a])) for f, a in enumerate(top)]
    return live, build_events(frames, cfg["fps"], e["min_frames"], e["min_conf"], tuple(e["ignore"]))


def gt_frames(rid: str, cfg: dict, actions: list[str], n: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    index = {a: i for i, a in enumerate(actions)}
    gt, segs = np.zeros(n, np.int64), []
    for r in gt_actions(rid, cfg["paths"]["labels"], cfg["fps"]):
        s, e = round(r["start"] * cfg["fps"]), min(round(r["end"] * cfg["fps"]) - 1, n - 1)
        if s <= e and r["action"] in index:
            gt[s:e + 1] = index[r["action"]]
            segs.append((s, e, index[r["action"]]))
    return gt, segs


def plan(split: str) -> None:
    cfg, actions = load_config(), load_mapping()
    out = {}
    for rid in SPLITS[split]:
        live, events = live_and_events(rid, cfg, actions)
        out[rid] = np.array(updates(events, cfg["fps"], len(live), EVERY), np.int64).reshape(-1, 4)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{split}_plan.npz", **out)
    print(json.dumps({"split": split, "videos": len(out), "updates": int(sum(len(v) for v in out.values()))}))


def _caches() -> dict[str, Path]:
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features import vjepa2_finetune as ft
    return {"vjepa": ft.CACHE, "vmae": vft.CACHE, "hands": ft.HANDS}


def windows(split: str, part: str) -> None:
    from scripts.recognition.predict_vjepa import member_probs

    k, n = (int(x) for x in part.split("/"))
    p = np.load(OUT / f"{split}_plan.npz")
    rids = sorted(p.files)[k::n]
    segments = [(rid, int(s), int(e)) for rid in rids for _, s, e, _ in p[rid]]
    t0 = time.time()
    probs = member_probs(segments, WEIGHTS_DIR, _caches(), fracs=(0.5,))
    np.savez(OUT / f"{split}_members_{k}.npz", rids=np.array(rids), **probs)
    print(json.dumps({"split": split, "part": part, "windows": len(segments),
                      "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


def benchmark(repeats: int = 20) -> None:
    """ms for one update (16 frames from the frame cache, batch 1) per member, on this GPU."""
    import torch
    from types import SimpleNamespace

    from scripts.features import finetune_videomae_ar as vft
    from scripts.features import vjepa2_finetune as ft

    cfg = json.loads((WEIGHTS_DIR / "settings.json").read_text(encoding="utf-8"))
    caches, rid = _caches(), SPLITS["test"][0]
    window = (rid, 300, 340)

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return round(1000 * (time.perf_counter() - t0) / repeats, 1)

    ms = {}
    for run in cfg["vjepa_runs"]:
        model = ft.Finetuned(run["layers"], cfg["size"]).cuda()
        model.load_state_dict(torch.load(WEIGHTS_DIR / run["file"], map_location="cuda", weights_only=True),
                              strict=False)
        args = SimpleNamespace(crop=run["crop"], size=cfg["size"], context=cfg["context"], workers=0)
        ms[Path(run["file"]).stem] = timed(lambda: ft.predict(model, [(*window, 1, 0.5)], args, 1, caches["vjepa"],
                                                              caches["hands"]))
        del model
        torch.cuda.empty_cache()
    model = vft.load_model(len(load_mapping()), WEIGHTS_DIR / cfg["videomae"])
    ms[Path(cfg["videomae"]).stem] = timed(lambda: vft.tta_probs(model, [(*window, 1)], 1, (0.5,),
                                                                 cache=caches["vmae"]))
    report = {"gpu": torch.cuda.get_device_name(0), "ms_per_update": ms,
              "subsets_ms": {name: round(sum(ms[m] for m in members), 1) for name, members in SUBSETS.items()},
              "budget_ms": BUDGET_MS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def benchmark_live() -> None:
    """One update of the first V-JEPA 2 run every 0.5 s, printing its ms, until stopped. Run it beside the TCN
    pipeline (``run_ar --compute``) on the same GPU to see whether both keep up together."""
    import torch
    from types import SimpleNamespace

    from scripts.features import vjepa2_finetune as ft

    cfg = json.loads((WEIGHTS_DIR / "settings.json").read_text(encoding="utf-8"))
    caches, rid, run = _caches(), SPLITS["test"][0], cfg["vjepa_runs"][0]
    model = ft.Finetuned(run["layers"], cfg["size"]).cuda()
    model.load_state_dict(torch.load(WEIGHTS_DIR / run["file"], map_location="cuda", weights_only=True), strict=False)
    args = SimpleNamespace(crop=run["crop"], size=cfg["size"], context=cfg["context"], workers=0)
    while True:
        t0 = time.perf_counter()
        ft.predict(model, [(rid, 300, 340, 1, 0.5)], args, 1, caches["vjepa"], caches["hands"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        print(json.dumps({"t": round(time.time(), 1), "ms": round(1000 * seconds, 1)}), flush=True)
        time.sleep(max(0.0, BUDGET_MS / 1000 - seconds))


def load_members(split: str) -> dict[str, dict[str, np.ndarray]]:
    """rid -> member -> (updates, 74), rows in plan order."""
    p = np.load(OUT / f"{split}_plan.npz")
    out = {}
    for f in sorted(OUT.glob(f"{split}_members_*.npz")):
        m = np.load(f)
        start = 0
        for rid in m["rids"]:
            n = len(p[str(rid)])
            out[str(rid)] = {k: m[k][start:start + n] for k in m.files if k != "rids"}
            start += n
    assert set(out) == set(p.files), f"{split}: member files cover {len(out)} of {len(p.files)} videos"
    return out


_DATA: dict = {}


def _load_split(split: str) -> list[dict]:
    cfg, actions = load_config(), load_mapping()
    p, members = np.load(OUT / f"{split}_plan.npz"), load_members(split)
    videos = []
    for rid in SPLITS[split]:
        live, events = live_and_events(rid, cfg, actions)
        gt, segs = gt_frames(rid, cfg, actions, len(live))
        upd = [tuple(int(x) for x in u) for u in p[rid]]
        videos.append({"rid": rid, "live": live, "events": events, "upd": upd, "members": members[rid],
                       "gt": gt, "segs": segs})
    return videos


def _shown(v: dict, members: tuple, w: float, lam: float, alpha: float, first, pairs) -> np.ndarray:
    m = sum(v["members"][k] for k in members) / len(members)
    return shown_probs(v["live"], v["events"], v["upd"], m, w, lam, alpha, first, pairs, DELAY)


def score(videos: list[dict], shown: list[np.ndarray]) -> dict:
    """Scores pooled over all frames / actions of the videos."""
    r = frame_scores(np.concatenate(shown), np.concatenate([v["gt"] for v in videos]),
                     [(s + off, e + off, a) for v, off in zip(videos, np.cumsum([0] + [len(v["gt"]) for v in videos]))
                      for s, e, a in v["segs"]])
    return {k: round(x, 2) for k, x in r.items()}


def _use_variant(variant: str) -> None:
    global OUT, WEIGHTS_DIR, LABELS, ACTIONS
    OUT, WEIGHTS_DIR, LABELS, ACTIONS = VARIANTS[variant]


def _init(split: str, variant: str | None = None) -> None:
    """``variant`` for spawned workers (Windows), which re-import this module and start on the default folder."""
    if variant:
        _use_variant(variant)
    _DATA["videos"] = _load_split(split)
    _DATA["prior"] = transition_counts(TRAIN, load_mapping())


def _grid_rows(subset: str) -> list[dict]:
    rows = []
    for w, lam in itertools.product(WEIGHTS, STRENGTHS):
        for alpha in (SMOOTHING if lam else SMOOTHING[:1]):
            shown = [_shown(v, SUBSETS[subset], w, lam, alpha, *_DATA["prior"]) for v in _DATA["videos"]]
            rows.append({"members": subset, "weight": w, "strength": lam, "smoothing": alpha,
                         **score(_DATA["videos"], shown)})
    return rows


def select(workers: int, variant: str = "release") -> None:
    bench = OUT / "benchmark.json"
    timing = json.loads(bench.read_text(encoding="utf-8"))["subsets_ms"] if bench.is_file() else None
    have = set(next(iter(load_members("val").values())))
    subsets = [s for s in SUBSETS if set(SUBSETS[s]) <= have and (timing is None or timing[s] < BUDGET_MS)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=("val", variant)) as pool:
        grid = [r for rows in pool.map(_grid_rows, subsets) for r in rows]
    best = max(grid, key=lambda r: (r["frame_acc"], -len(SUBSETS[r["members"]]), -r["strength"]))
    _init("val")
    report = {"protocol": "IndustReal val (16 videos): label shown per frame, background included; live model "
                          "events, ensemble windows end at the current frame, shown 0.5 s after each update",
              "real_time_subsets": subsets, "timing_ms": timing,
              "chosen": best, "live_model": score(_DATA["videos"], [v["live"] for v in _DATA["videos"]]),
              "grid": grid}
    (OUT / "online_vjepa_val.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))


def _write_labels(v: dict, shown: np.ndarray, cfg: dict, actions: list[str]) -> None:
    write_jsonl(cfg["paths"]["out"] / v["rid"] / LABELS, topk_rows(shown, actions, cfg["fps"], 5))


def select_mistakes() -> None:
    """Re-pick the ensemble weight by what the assistant does with the labels: the engine on the 16 val videos
    for each weight in ``WEIGHTS`` (with that weight's best strength / smoothing from ``--select``), with
    our own ASD as the state source (the released ASD has no val predictions). Highest mistake F1 wins, then
    more mistakes warned before they happen, then fewer early warnings per hour (the same mistakes caught with
    fewer warnings), then the smaller weight; weight 0 is the live model itself."""
    from assist.engine.graph import load_tasks
    from assist.evaluation.eval import evaluate
    from assist.run_engine import run

    report = json.loads((OUT / "online_vjepa_val.json").read_text(encoding="utf-8"))
    report["chosen"] = report.get("chosen_by_frame_acc", report["chosen"])
    subset = report["chosen"]["members"]
    cfg, actions = load_config(), load_mapping()
    tasks = load_tasks(cfg["graph_path"])
    _init("val")
    videos, prior = _DATA["videos"], _DATA["prior"]

    def apply(c: dict) -> None:
        for v in videos:
            _write_labels(v, _shown(v, SUBSETS[subset], c["weight"], c["strength"], c["smoothing"], *prior),
                          cfg, actions)
            write_json(cfg["paths"]["out"] / v["rid"] / f"engine_{ACTIONS}_asdours.json",
                       run(v["rid"], ACTIONS, "asdours", cfg, tasks))

    cands = []
    for w in WEIGHTS:
        c = max((r for r in report["grid"] if r["members"] == subset and r["weight"] == w),
                key=lambda r: (r["frame_acc"], -r["strength"]))
        apply(c)
        ev = evaluate(f"{ACTIONS}_asdours", SPLITS["val"], cfg, tasks)
        m = ev["rows"]["all"]
        p, r = m["precision"] or 0.0, m["recall"] or 0.0
        cands.append({**c, "mistake_precision": round(100 * p, 1), "mistake_recall": round(100 * r, 1),
                      "mistake_f1": round(200 * p * r / (p + r), 1) if p + r else 0.0,
                      "false_alarms_per_h": round(m["false_alarms_per_h"], 1), "warned": m["warned"],
                      "early_warnings_per_h": round(ev["early_warnings_per_h"], 1),
                      "gt_mistakes": m["gt"]})
        print(json.dumps(cands[-1]), flush=True)
    best = max(cands, key=lambda c: (c["mistake_f1"], c["warned"], -c["early_warnings_per_h"], -c["weight"]))
    apply(best)
    report.update({"chosen_by_frame_acc": report["chosen"], "chosen": best, "mistake_check": cands})
    (OUT / "online_vjepa_val.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"chosen": best}, indent=2))


def passing_setting(cands: list[dict], live: dict, base: dict) -> dict | None:
    """The rule fixed before the action-beginning run: beat the live model on frames and on the label at action
    end, find at least as many mistakes, raise no more false alarms. Ties as in ``select_mistakes``."""
    passed = [c for c in cands if c["frame_acc"] > live["frame_acc"] and c["end_acc"] > live["end_acc"]
              and c["mistake_recall"] >= base["mistake_recall"]
              and c["false_alarms_per_h"] <= base["false_alarms_per_h"]]
    return max(passed, key=lambda c: (c["mistake_f1"], c["warned"], -c["early_warnings_per_h"], -c["weight"]),
               default=None)


def apply_rule() -> None:
    path = OUT / "online_vjepa_val.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    cands = report["mistake_check"]
    base = next(c for c in cands if c["weight"] == 0.0)     # weight 0 is the live model itself
    best = passing_setting(cands, report["live_model"], base)
    report["rule"] = {"passed": best is not None, "chosen": best, "chosen_by_mistakes": report["chosen"]}
    if best:
        report["chosen"] = best
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["rule"], indent=2))


def write(split: str) -> None:
    cfg, actions = load_config(), load_mapping()
    report = json.loads((OUT / "online_vjepa_val.json").read_text(encoding="utf-8"))
    if split == "test" and not report.get("rule", {"passed": True})["passed"]:
        raise SystemExit("val did not pass the rule: test is not scored")
    chosen = report["chosen"]
    _init(split)
    videos, prior = _DATA["videos"], _DATA["prior"]
    shown = [_shown(v, SUBSETS[chosen["members"]], chosen["weight"], chosen["strength"], chosen["smoothing"], *prior)
             for v in videos]
    for v, s in zip(videos, shown):
        _write_labels(v, s, cfg, actions)
    report = {"split": split, "chosen_on_val": chosen, "live_model": score(videos, [v["live"] for v in videos]),
              "online_vjepa": score(videos, shown)}
    (OUT / f"online_vjepa_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--plan", "--benchmark", "--benchmark-live", "--windows", "--select", "--select-mistakes", "--apply-rule", "--write"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--part", default="0/1", help="i/n: this process's share of the videos (one per GPU).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variant", choices=list(VARIANTS), default="release",
                        help="prefix: the V-JEPA 2 model trained on action beginnings; own folder and label files.")
    args = parser.parse_args()
    _use_variant(args.variant)
    if args.plan:
        plan(args.split)
    if args.benchmark:
        benchmark()
    if args.benchmark_live:
        benchmark_live()
    if args.windows:
        windows(args.split, args.part)
    if args.select:
        select(args.workers, args.variant)
    if args.select_mistakes:
        select_mistakes()
    if args.apply_rule:
        apply_rule()
    if args.write:
        write(args.split)


if __name__ == "__main__":
    main()
