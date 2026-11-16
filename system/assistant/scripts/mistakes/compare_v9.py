"""Compare the v9 PSR system (branch ``psr``, folder ``PSR``) with the live system on the same scoring.

v9 fuses predicted actions with its own per-part verifiers and reports completed steps. It saves those steps per video,
so no model runs here. Two comparisons on the 32 test videos:

  steps     v9's saved steps through our port of the official scoring, to check its published table
  mistakes  v9's steps as the state input of our engine, with the same live actions as the live system, so only the
            state source differs. Nothing is tuned: the engine's settings are the live system's.

v9 reports steps, not a stream of views, so its state is given to the engine as one reading every 0.5 s. The engine then
always "sees" the assembly, unlike the released detector. Hand occlusion uses the released detector's last box.

    git archive origin/psr PSR/results | tar -x -C output/assist/v9      # after: git fetch origin psr
    python -m scripts.mistakes.compare_v9       # -> output/assist/v9/compare.json
"""
from __future__ import annotations

import json
from pathlib import Path

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.engine.psr import make_entry
from monitor.settings import ROOT

RESULTS = ROOT / "output/assist/v9/PSR/results"
ACTIONS = "pred_vjepap"
LIVE = f"{ACTIONS}_asd"
STATE = "asdv9"


def _by_description(proc_info: list[dict]) -> dict[str, dict]:
    return {p["description"]: p for p in proc_info}


def step_entries(steps: list, proc_info: list[dict]) -> list[dict]:
    """v9's [frame, description] rows as entries of our scoring code."""
    table = _by_description(proc_info)
    return [make_entry(int(frame), table[text]["id"], proc_info, conf=1) for frame, text in steps]


def state_rows(steps: list, initial: list[int], proc_info: list[dict], frames: int, boxes: list[dict],
               fps: float = 10, every: int = 5) -> list[dict]:
    """The state v9 had reported so far, every ``every`` frames, in the format of the state-detector files."""
    table = _by_description(proc_info)
    order = sorted(steps, key=lambda s: s[0])
    state, k, b, box, out = list(initial), 0, 0, None, []
    for f in range(0, frames, every):
        while k < len(order) and order[k][0] <= f:
            step = table[order[k][1]]
            state[step["state_idx"]] = 1 if step["install"] else 0
            k += 1
        t = round(f / fps, 2)
        while b < len(boxes) and boxes[b]["t"] <= t:
            box = boxes[b]["box"]
            b += 1
        out.append({"f": f, "t": t, "state": list(state), "conf": 1.0, "box": box})
    return out


def _mistakes(run: str, rids: list[str], cfg: dict, tasks: dict) -> dict:
    from assist.evaluation.eval import evaluate

    r = evaluate(run, rids, cfg, tasks)
    rows = r["rows"]
    return {"found": rows["all"]["tp"], "of": rows["all"]["gt"], "alerts": rows["all"]["alerts"],
            "false_alarms_per_h": round(rows["all"]["false_alarms_per_h"], 1), "warned": rows["all"]["warned"],
            "early_warnings_per_h": round(r["early_warnings_per_h"], 1),
            "by_type": {k: {"found": rows[k]["tp"], "of": rows[k]["gt"], "alerts": rows[k]["alerts"],
                            "false_alarms_per_h": round(rows[k]["false_alarms_per_h"], 1),
                            "delay_s": None if rows[k]["delay_s"] is None else round(rows[k]["delay_s"], 1)}
                        for k in ("skipped", "order", "install")}}


def _summary(result: dict) -> dict:
    s = result["summary"]
    keep = lambda d: {"f1": round(d["f1"], 3), "pos": round(d["pos"], 3), "delay_s": round(d["delay_s"], 1)}   # noqa: E731
    return {"all": keep(s["all"]), "with_errors": keep(s["with_errors"])}


def main() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data, engine_steps, score
    from assist.industreal import SPLITS, n_frames, task_of
    from assist.run_engine import run

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "release")
    rids = SPLITS["test"]
    saved = json.loads((RESULTS / "v9_predictedAR_test.json").read_text(encoding="utf-8"))
    steps = {v["video"]: v["predicted_steps"] for v in saved["videos"]}
    if set(steps) != set(rids):
        raise ValueError("v9's test videos are not the 32 test videos")
    report = {"v9_published": {k: saved["summary"][k] for k in ("all", "with_errors")}}
    report["v9_steps_our_scoring"] = _summary(score(rids, data, lambda rid, v: step_entries(steps[rid], data.proc_info)))
    results = {}
    for rid in rids:
        out_dir = cfg["paths"]["out"] / rid
        write_jsonl(out_dir / "asd_v9.jsonl", state_rows(
            steps[rid], list(tasks[task_of(rid)].initial_state), data.proc_info,
            n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"]), read_jsonl(out_dir / "asd.jsonl")))
        results[rid] = run(rid, ACTIONS, STATE, cfg, tasks)
        write_json(out_dir / f"engine_{ACTIONS}_{STATE}.json", results[rid])
    live = {rid: json.loads((cfg["paths"]["out"] / rid / f"engine_{LIVE}.json").read_text(encoding="utf-8")) for rid in rids}
    steps_of = lambda res: lambda rid, v: engine_steps(res[rid], cfg["components"], data.proc_info, data.graph[v["task"]])   # noqa: E731
    report["live"] = {"steps": _summary(score(rids, data, steps_of(live))), "mistakes": _mistakes(LIVE, rids, cfg, tasks)}
    report["engine_with_v9_state"] = {"steps": _summary(score(rids, data, steps_of(results))),
                                      "mistakes": _mistakes(f"{ACTIONS}_{STATE}", rids, cfg, tasks)}
    write_json(ROOT / "output/assist/v9/compare.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
