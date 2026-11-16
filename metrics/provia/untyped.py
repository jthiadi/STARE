"""STARE vs PROVIA, untyped mistake detection on our 32 IndustReal test mistakes (PREREGISTRATION_untyped.md).

Ground truth and matching are the teammate's evaluator (assist/evaluation/eval.py: gt_mistakes, collapse window,
+-30 s, one-to-one closest-first) with the type condition removed: an alarm matches a mistake of any type.
PROVIA's alarms: its sequential rule (third_party/PROVIA alarm_protocol) on its anytime score, from its candidate dumps
(runs/industreal/model_seed*/{validation,test}_candidates.npz); thresholds chosen on validation only.
    python metrics/provia/untyped.py      -> metrics/provia/results/untyped.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYS = ROOT / "system"
os.environ.setdefault("INDUSTREAL_DATA", str(SYS / "dataset"))
os.environ.setdefault("TEAMMATE_AR", str(SYS / "assistant"))
sys.path[:0] = [str(SYS / "psr_v14"), str(SYS / "assistant"), str(ROOT / "third_party/PROVIA")]
RUNS = Path(os.environ.get("PROVIA_RUNS", ROOT / "third_party/PROVIA/runs/industreal"))
STARE = {"val": SYS / "psr_v14/results/v14_predictedAR_val.json", "test": SYS / "psr_v14/results/v14_predictedAR_test.json"}
SPLIT = {"val": "validation", "test": "test"}


def setup():
    from assist.common import load_config
    from assist.engine.graph import load_tasks
    from assist.evaluation.eval import gt_mistakes
    sys.path.append(str(SYS / "v14_demo"))
    import frames_compat  # noqa: F401  frame counts from the .mp4 when the dataset has no JPEG frames (as reproduce_all)
    from psr.core import frame_paths, video_dir
    cfg = load_config()
    return cfg, load_tasks(cfg["graph_path"]), gt_mistakes, frame_paths, video_dir


def gts_for(split, videos):
    cfg, tasks, gt_mistakes, frame_paths, video_dir = setup()
    fps, grace = cfg["fps"], cfg["engine"]["order_grace_s"]
    out = {}
    for v in videos:
        n = len(frame_paths(split, v))
        out[v] = (n / fps, gt_mistakes(v, tasks[v.split("_")[1]], cfg["components"], video_dir(split, v).parent, fps,
                                        n / fps, grace))
    return out, cfg["evaluation"]["match_window_s"]


def collapse_untyped(times, window):
    out = []
    for t in sorted(times):
        if not any(0 <= t - u <= window for u in out):
            out.append(t)
    return out


def score(alarms: dict, gts: dict, window: float) -> dict:
    """alarms: video -> list of times (s). Type-free version of the teammate's match (closest first, one-to-one)."""
    tp = fa = 0; hours = 0.0; by_type = {}
    for v, (dur, g) in gts.items():
        hours += dur / 3600
        al = collapse_untyped(alarms.get(v, []), window)
        pairs = sorted((abs(a - m["t"]), i, j) for i, m in enumerate(g) for j, a in enumerate(al) if abs(a - m["t"]) <= window)
        ug, ua = set(), set()
        for _, i, j in pairs:
            if i not in ug and j not in ua:
                ug.add(i); ua.add(j)
        tp += len(ug); fa += len(al) - len(ua)
        for i, m in enumerate(g):
            d, n = by_type.get(m["type"], (0, 0)); by_type[m["type"]] = (d + (i in ug), n + 1)
    n = sum(len(g) for _, g in gts.values())
    return {"found": tp, "mistakes": n, "precision": round(tp / (tp + fa), 3) if tp + fa else None,
            "false_alarms_per_h": round(fa / hours, 2), "found_by_type": {k: f"{d}/{n_}" for k, (d, n_) in sorted(by_type.items())}}


def stare_alarms(split):
    return {v["video"]: [a["t"] for a in v["mistake_alerts"]] for v in json.loads(STARE[split].read_text())["videos"]}


METHODS = {"PROVIA": "model_seed{seed}", "Causal-TCN": "baselines/causal_tcn_seed{seed}",
           "Controlled PREGO": "baselines/prego_seed{seed}"}   # PROVIA repository runs


def provia_videos(seed, split, method="PROVIA"):
    z = np.load(RUNS / METHODS[method].format(seed=seed) / f"{SPLIT[split]}_candidates.npz", allow_pickle=True)
    return list(z["videos"])


def provia_alarms(videos, prior_odds, thr):
    from provia.evaluation.alarm_protocol import alarms_for, evidence
    out = {}
    for v in videos:
        idx = alarms_for(evidence(v["anytime_score"], prior_odds), thr, "sequential")
        t = np.asarray(v["decoded_time"], float)
        out[str(v.get("video_id") or v.get("episode_id"))] = [float(t[i]) for i in idx]
    return out


def main():
    from provia.evaluation.alarm_protocol import ERROR, sweep
    res = {"protocol": "ours, untyped: 32 test mistakes, +-30 s, any type, one-to-one; alarms within 30 s merged"}
    val_ids = list(stare_alarms("val")); test_ids = list(stare_alarms("test"))
    gv, win = gts_for("val", val_ids); gt, _ = gts_for("test", test_ids)
    res["STARE"] = {"val": score(stare_alarms("val"), gv, win), "test": score(stare_alarms("test"), gt, win)}
    target = res["STARE"]["val"]["false_alarms_per_h"]
    for method in METHODS:
        if not (RUNS / METHODS[method].format(seed=2) / "test_candidates.npz").exists():
            print("missing", method); continue
        res[method] = score_method(method, gv, gt, win, target)
    out = Path(os.environ.get("PROVIA_RESULTS", HERE / "results"))      # reproduce_all.py points this elsewhere
    out.mkdir(parents=True, exist_ok=True)
    (out / "untyped.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"STARE": res["STARE"]["test"], **{m: {k: v for k, v in res[m].items() if k != "seeds"}
                                                          for m in METHODS if m in res}}, indent=1))


def score_method(method, gv, gt, win, target):
    from provia.evaluation.alarm_protocol import ERROR, sweep
    seeds = {}
    for seed in (0, 1, 2):
        vv, tv = provia_videos(seed, "val", method), provia_videos(seed, "test", method)
        modes = np.concatenate([np.asarray(v["native_mode"]) for v in vv]); prev = float((modes == ERROR).mean())
        po = prev / (1 - prev)
        # candidate thresholds: the sequential statistic's values on validation (as alarm_protocol.sweep)
        from provia.evaluation.alarm_protocol import evidence
        vals = []
        for v in vv:
            m = 0.0
            for x in evidence(v["anytime_score"], po):
                m = (1.0 + m) * x; vals.append(m)
        grid = np.unique(np.quantile(np.asarray(vals), np.linspace(0, 1, 400)))
        rows = [(float(t), score(provia_alarms(vv, po, float(t)), gv, win)) for t in grid]
        ok = [r for r in rows if r[1]["false_alarms_per_h"] <= target]
        thr = min(ok, key=lambda r: (abs(r[1]["false_alarms_per_h"] - target), -r[1]["found"]))[0] if ok else float(grid[-1])
        entry = {"threshold_matched_to_STARE_val": thr, "val": score(provia_alarms(vv, po, thr), gv, win),
                 "test": score(provia_alarms(tv, po, thr), gt, win)}
        for r in sweep(vv, po, "sequential", 1, [0.1, 0.5, 1.0], "fa_per_min_correct_operation"):
            if "threshold" in r:
                entry[f"test_at_val_budget_{r['budget']}_per_min"] = score(provia_alarms(tv, po, r["threshold"]), gt, win)
        seeds[seed] = entry
        print(method, seed, json.dumps(entry["test"]), flush=True)
    r = {"seeds": seeds}
    for k in ("found", "precision", "false_alarms_per_h"):
        xs = [s["test"][k] for s in seeds.values() if s["test"][k] is not None]
        r[f"test_{k}_mean_sd"] = [round(float(np.mean(xs)), 3), round(float(np.std(xs, ddof=1)), 3)] if len(xs) > 1 else xs
    return r


if __name__ == "__main__":
    main()
