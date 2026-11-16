"""v14 PSR (frozen, C:/Project/industreal-psr-v14) inside this assistant, on the 32 test videos. Same idea as compare_v9.py.

v14 = per-part DINOv2 verifiers + the released IndustReal ASD + live actions in a causal state filter. Its frozen test run
saved, per video, the completed steps and its own mistake alerts (``output/assist/v14/v14_predictedAR_test.json``,
copied from ``industreal-psr-v14/results``). No model runs here. Three things are written / scored:

  v14 alone        v14's saved steps through this repository's port of the official scoring, and v14's saved alerts
                   through this repository's mistake scoring (a check that both scorers agree with v14's own)
  engine + v14     v14's steps as the engine's state input (``asd_v14.jsonl``, one reading every 0.5 s), with the
                   live actions ``--actions`` (default ``pred``: the causal TCN, the same AR v14 used); nothing tuned
  engine_<a>_v14   ``engine_<actions>_v14.json``: v14's own steps and alerts in the engine's result format, so the
                   dashboard can replay them (step status from v14's steps, alerts = v14's alerts)

    python -m scripts.mistakes.compare_v14 [--actions pred]     # -> output/assist/v14/compare.json
"""
from __future__ import annotations

import argparse
import json

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from monitor.settings import ROOT
from scripts.mistakes.compare_v9 import _mistakes, _summary, state_rows, step_entries

SAVED = ROOT / "output/assist/v14/v14_predictedAR_test.json"
STATE = "asdv14"
ALERT_TEXT = {"wrong_order": "Wrong order", "skipped_step": "Skipped step", "incorrect_install": "Installed wrong",
              "possible_incorrect_install": "Check the install"}


def v14_engine_result(rid: str, video: dict, task, duration: float, base: dict, proc: dict, components: list) -> dict:
    """v14's own output in the engine's format; the engine's events (actions) are kept from ``base`` for the replay."""
    alerts = []
    for a in video["mistake_alerts"]:
        step = a.get("step")
        name = task.steps[step].name if step in task.steps else (a.get("component") or "").replace("_", " ")
        alerts.append({"t": a["t"], "type": a["type"], "kind": "mistake", "step": step, "component": a.get("component"),
                       "severity": "stop" if a["type"] != "possible_incorrect_install" else "check",
                       "title": f"{ALERT_TEXT.get(a['type'], a['type'])}: {name}",
                       "why": "Raised by the v14 PSR (task-graph rules on its reported steps).", "fix_steps": [],
                       "rule": a["type"], "evidence": {}, "source": "v14", "missing": []})
    done = {}
    for frame, text in sorted(video["predicted_steps"]):
        p = proc[text]
        if p["id"] % 3 == 1:                     # "incorrectly installed": not a completed step
            continue
        sids = task.steps_for(components[p["state_idx"]], 1 if p["install"] else 0)
        if sids:
            done.setdefault(sids[0], frame / 10)
    log = [{"t": t, "step": sid, "status": "done", "source": "v14"} for sid, t in sorted(done.items(), key=lambda x: x[1])]
    return {**base, "state": "v14", "alerts": alerts, "log": log,
            "steps": [{**s, "final": "done" if s["id"] in done else s["final"]} for s in base["steps"]],
            "duration_s": round(duration, 1)}


def main() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data, engine_steps, score
    from assist.industreal import SPLITS, n_frames, task_of
    from assist.run_engine import run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", default="pred")
    a = ap.parse_args()
    cfg = load_config(); tasks = load_tasks(cfg["graph_path"]); data = Data(cfg, "release"); rids = SPLITS["test"]
    saved = json.loads(SAVED.read_text(encoding="utf-8")); vids = {v["video"]: v for v in saved["videos"]}
    if set(vids) != set(rids):
        raise ValueError("v14's test videos are not the 32 test videos")
    report = {"v14_published": {k: saved["summary"][k] for k in ("all", "with_errors", "mistakes")}}
    report["v14_steps_our_scoring"] = _summary(score(rids, data, lambda rid, v: step_entries(vids[rid]["predicted_steps"], data.proc_info)))
    results = {}
    for rid in rids:
        out_dir = cfg["paths"]["out"] / rid; frames = n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"])
        write_jsonl(out_dir / "asd_v14.jsonl", state_rows(vids[rid]["predicted_steps"], list(tasks[task_of(rid)].initial_state),
                                                          data.proc_info, frames, read_jsonl(out_dir / "asd.jsonl")))
        results[rid] = run(rid, a.actions, STATE, cfg, tasks)
        write_json(out_dir / f"engine_{a.actions}_{STATE}.json", results[rid])
        write_json(out_dir / f"engine_{a.actions}_v14.json",
                   v14_engine_result(rid, vids[rid], tasks[task_of(rid)], frames / cfg["fps"], results[rid],
                                     {q["description"]: q for q in data.proc_info}, cfg["components"]))
    report["v14_alerts_our_scoring"] = _mistakes(f"{a.actions}_v14", rids, cfg, tasks)
    steps_of = lambda res: lambda rid, v: engine_steps(res[rid], cfg["components"], data.proc_info, data.graph[v["task"]])   # noqa: E731
    report["engine_with_v14_state"] = {"actions": a.actions, "steps": _summary(score(rids, data, steps_of(results))),
                                       "mistakes": _mistakes(f"{a.actions}_{STATE}", rids, cfg, tasks)}
    write_json(ROOT / "output/assist/v14/compare.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
