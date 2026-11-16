"""STARE-TCN's mistake alerts scored with PROVIA's IndustReal protocol (PREREGISTRATION.md). CPU, a few seconds.
Needs data/industreal.jsonl (make_manifest.sh). Writes results/provia_protocol.json and prints the table.
    python metrics/provia/score.py"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "third_party/PROVIA"))
from provia.evaluation.alarm_protocol import evaluate                     # noqa: E402  PROVIA, unchanged
from provia.training.completion_metrics import _cell_assignment           # noqa: E402

HZ = 8.0
ERROR = 1
OURS = {"test": ROOT / "system/psr_v14/results/v14_predictedAR_test.json",
        "validation": ROOT / "system/psr_v14/results/v14_predictedAR_val.json"}
SETS = {"install": {"incorrect_install", "possible_incorrect_install", "wrong_part"}, "all": None}
# PROVIA's published IndustReal numbers (arXiv 2609.20638): step-level AP / AUROC (Table II, mean of 3 seeds) and
# alarm-level recall of all mistakes at 0.1 / 0.5 / 1.0 false alarms per minute of correct operation (Table IV)
PUBLISHED = {
    "PROVIA": {"AP": 0.153, "AUROC": 0.625, "recall_at_fa_per_min": {"0.1": 0.098, "0.5": 0.392, "1.0": 0.667}},
    "Causal-TCN": {"AP": 0.148, "AUROC": 0.616, "recall_at_fa_per_min": {"0.1": 0.255, "0.5": 0.529, "1.0": 0.882}},
    "Controlled PREGO": {"AP": 0.133, "AUROC": 0.612, "recall_at_fa_per_min": {"0.1": 0.137, "0.5": 0.392, "1.0": 0.863}},
    "MistSense-RGB-style": {"AP": 0.101, "AUROC": 0.567, "recall_at_fa_per_min": {"0.1": 0.118, "0.5": 0.529, "1.0": 0.745}},
    "Position control (time), no perception": {"recall_at_fa_per_min": {"0.1": 0.294, "0.5": 0.588, "1.0": 0.765}},
}


def grid(t: float) -> float:
    return round(t * HZ) / HZ


def videos_for(split: str, types) -> tuple[list[dict], dict]:
    manifest = {r["video_id"]: r for r in map(json.loads, open(HERE / "data/industreal.jsonl", encoding="utf-8"))
                if r["split"] == split}
    ours = {v["video"]: v for v in json.loads(OURS[split].read_text())["videos"]}
    assert set(manifest) == set(ours), "recordings differ"
    out, dropped = [], 0
    for vid in sorted(manifest):
        ep, mine = manifest[vid], ours[vid]
        segs = sorted(ep["segments"], key=lambda s: s["end_sec"])
        native_time = np.array([grid(s["end_sec"]) for s in segs])
        native_mode = np.array([ERROR if s["correctness"] == "error" else 0 for s in segs])
        steps = sorted({grid(f / 10.0) for f, _ in mine["predicted_steps"]})
        end = grid(ep["duration_sec"])
        times = sorted({0.0, end, *steps})
        score = np.zeros(len(times))
        for a in mine["mistake_alerts"]:
            if types is not None and a["type"] not in types:
                continue
            if not steps:
                dropped += 1
                continue
            before = [s for s in steps if s <= a["t"] + 1e-9]
            score[times.index(before[-1] if before else steps[0])] = 1.0
        ts = np.array(times)
        idx = np.round(ts * HZ).astype(int)
        nat = np.round(native_time * HZ).astype(int)
        assignment = _cell_assignment(nat, idx, np.arange(max(idx.max(), nat.max()) + 1) / HZ)
        out.append({"video_id": vid, "assignment": assignment, "native_mode": native_mode, "native_time": native_time,
                    "decoded_time": ts, "anytime_score": score,
                    "native_error_types": [tuple(s.get("error_types") or ()) for s in segs]})
    return out, {"alerts_dropped": dropped}


def step_level(videos: list[dict]) -> dict:
    y, s = [], []
    for v in videos:
        for j in range(len(v["native_mode"])):
            c = np.flatnonzero(v["assignment"] == j)
            y.append(int(v["native_mode"][j] == ERROR)); s.append(float(v["anytime_score"][c].max()) if len(c) else 0.0)
    return {"AP": round(float(average_precision_score(y, s)), 3), "AUROC": round(float(roc_auc_score(y, s)), 3),
            "steps": len(y), "mistakes": int(sum(y))}


def main() -> None:
    res = {"protocol": "PROVIA IndustReal (arXiv 2609.20638), alarm-level grace 1, instantaneous rule at score 0.5",
           "published": PUBLISHED}
    keys = ("recall_all", "fa_per_min_correct_operation", "recall_first", "recall_after_first", "median_delay_sec",
            "zero_alarm_correct_share", "n_mistakes", "n_videos")
    for split in ("validation", "test"):
        for name, types in SETS.items():
            vids, info = videos_for(split, types)
            r = evaluate(vids, 1.0, "instantaneous", 0.5, 1)
            row = {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in keys}
            row.update(step_level(vids), **info, recall_by_type=r["recall_by_type"])
            res[f"{split}/{name}"] = row
            print(split, name, json.dumps(row), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/provia_protocol.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
