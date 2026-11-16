"""PSR + mistake metrics of the FULL SYSTEM on the 32 IndustReal test recordings, from the outputs of the on-demand
pipeline (system/assistant/live: their TCN + V-JEPA 2 live actions, v14 run live on the video, their engine).

Scored with the teammate's port of the official IndustReal PSR evaluator (assist/evaluation/psr_eval.py; checked equal to
the official psr_utils.determine_performance on 224 video/system pairs, and it gives v14's frozen numbers exactly) and their
mistake scoring (assist/evaluation/eval.py: same ground-truth rules, +-30 s matching). Ground truth is read only here.
  v14 live          v14's own steps and alerts (v14_live.json)
  engine + v14      the engine with the live actions (pred_vjepap) and v14's steps as its state
Run from system/assistant (server: source ../../server/env.sh):  python ../../metrics/psr/score_full_system.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from assist.common import load_config, write_json
from assist.engine.graph import load_tasks
from assist.evaluation.eval import TYPE_OF, TYPES, collapse, gt_mistakes, match
from assist.evaluation.psr_eval import Data, engine_steps, score
from assist.industreal import SPLITS, task_of
from scripts.mistakes.compare_v9 import _mistakes, _summary, step_entries

RUN = "pred_vjepap_asdv14"
OUT_FILE = Path(__file__).resolve().parent / "results" / "full_system_test.json"


def mistakes_of(alerts_by_rid: dict, cfg: dict, tasks: dict) -> dict:
    ev, hours = cfg["evaluation"], 0.0
    st = {k: {"gt": 0, "alerts": 0, "tp": 0} for k in TYPES}
    for rid, (dur, alerts) in alerts_by_rid.items():
        hours += dur / 3600
        g = gt_mistakes(rid, tasks[task_of(rid)], cfg["components"], cfg["paths"]["labels"], cfg["fps"], dur,
                        cfg["engine"]["order_grace_s"])
        m = collapse([a for a in alerts if a.get("kind", "mistake") == "mistake"], ev["match_window_s"])
        for k in TYPES:
            st[k]["gt"] += sum(1 for x in g if x["type"] == k)
            st[k]["alerts"] += sum(1 for a in m if TYPE_OF[a["type"]] == k)
        for a, _ in match(g, m, ev["match_window_s"]):
            st[a["type"]]["tp"] += 1
    tot = {f: sum(st[k][f] for k in TYPES) for f in ("gt", "alerts", "tp")}
    return {"found": tot["tp"], "of": tot["gt"], "alerts": tot["alerts"],
            "precision": round(tot["tp"] / tot["alerts"], 3) if tot["alerts"] else None,
            "false_alarms_per_h": round((tot["alerts"] - tot["tp"]) / hours, 1),
            "by_type": {k: f"{st[k]['tp']}/{st[k]['gt']} ({st[k]['alerts']} alerts)" for k in TYPES}}


def main() -> None:
    cfg = load_config(); tasks = load_tasks(cfg["graph_path"]); data = Data(cfg, "release"); rids = SPLITS["test"]
    out = cfg["paths"]["out"]
    missing = [r for r in rids if not (out / r / "v14_live.json").exists() or not (out / r / f"engine_{RUN}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test recordings not computed yet: {missing[:5]} ...")
    v14 = {r: json.loads((out / r / "v14_live.json").read_text(encoding="utf-8")) for r in rids}
    eng = {r: json.loads((out / r / f"engine_{RUN}.json").read_text(encoding="utf-8")) for r in rids}
    report = {
        "protocol": "IndustReal test, 32 recordings, official PSR evaluator (port), POS / F1 / delay; mistakes: teammate rules",
        "inputs": "live actions = their TCN + action-beginning V-JEPA 2 (w 0.1), computed from the video; v14 frozen, run "
                  "live on the video (verifiers on even frames, ASD on odd frames)",
        "v14_live": {"steps": _summary(score(rids, data, lambda rid, v: step_entries(v14[rid]["predicted_steps"], data.proc_info))),
                     "mistakes": mistakes_of({r: (v14[r]["frames"] / cfg["fps"], v14[r]["mistake_alerts"]) for r in rids}, cfg, tasks),
                     "latency_ms_median_of_videos": sorted(v14[r]["latency_ms"]["median"] for r in rids)[len(rids) // 2]},
        "engine_with_v14_state": {"steps": _summary(score(rids, data, lambda rid, v: engine_steps(eng[rid], cfg["components"],
                                                                                                 data.proc_info, data.graph[v["task"]]))),
                                  "mistakes": _mistakes(RUN, rids, cfg, tasks)},
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_FILE, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
