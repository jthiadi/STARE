"""Combined check: the state detector keeps completing steps; the per-part model only adds "installed wrong" alerts.

The per-part model lost on test when it replaced the state detector (``part_state.py``), mostly on step recognition.
Here it never touches a step. After a step's install finishes (by actions or by state), its part is watched for
``window_s`` seconds, and the alert fires after ``looks`` readings in a row with P(mounted wrong) above ``thr``, on
looks with a good detection and clear hands.

Val only: the live actions + our detector (the released detector has no val predictions), with the two per-part models
averaged, as val chose before. The rule and the 12 settings were fixed before any result:

  at least 2 of the 5 wrong installs newly found, no more new false alarms than new finds, step F1 and POS not lower.

Thresholds below 0.5 were added after the rule failed, to see whether the fixed settings were simply too strict. They
are reported as ``after_the_rule`` and can never be chosen. Test is not scored here.

    python -m scripts.mistakes.part_check            # CPU; writes output/assist/part_state/part_check_val.json
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from assist.common import load_config, read_jsonl, write_json
from assist.engine.graph import load_tasks
from assist.evaluation.eval import evaluate, gt_mistakes
from assist.evaluation.psr_eval import Data, engine_steps, score
from assist.industreal import SPLITS, task_of
from assist.run_engine import run
from scripts.mistakes.part_state import ASD, OUT, STATE, probs

SPLIT = "val"
ACTIONS = "pred_vjepap"
MODEL = "both"
SETTINGS = [{"thr": thr, "window_s": window, "looks": looks}
            for thr, window, looks in itertools.product((0.5, 0.7, 0.9), (10, 30), (3, 8))]
AFTER_THE_RULE = [{"thr": thr, "window_s": window, "looks": looks}
                  for thr, window, looks in itertools.product((0.1, 0.15, 0.2, 0.3), (10, 30), (3, 8))]
MIN_NEW_FINDS = 2


def wrong_rows(rows: list[dict], p: np.ndarray) -> list[tuple[float, list[float], float]]:
    """(time, P(mounted wrong) per component, detection confidence) for every state-detector row."""
    if len(rows) != len(p):
        raise ValueError(f"{len(rows)} state-detector rows but {len(p)} predictions")
    return [(r["t"], q[:, 2].tolist(), r["conf"]) for r, q in zip(rows, p)]


def passes(base: dict, row: dict) -> bool:
    new_finds = row["install_found"] - base["install_found"]
    new_false_alarms = row["false_alarms"] - base["false_alarms"]
    return new_finds >= MIN_NEW_FINDS and new_false_alarms <= new_finds \
        and row["psr_f1"] >= base["psr_f1"] and row["psr_pos"] >= base["psr_pos"]


def choose(base: dict, rows: list[dict]) -> dict | None:
    passed = [r for r in rows if passes(base, r)]
    return max(passed, key=lambda r: (r["install_found"], -r["false_alarms"])) if passed else None


def scored(name: str, results: dict, cfg: dict, tasks: dict, data: Data) -> dict:
    rids = SPLITS[SPLIT]
    for rid in rids:
        write_json(cfg["paths"]["out"] / rid / f"engine_{name}.json", results[rid])
    r = evaluate(name, rids, cfg, tasks)["rows"]
    psr = score(rids, data, lambda rid, v: engine_steps(results[rid], cfg["components"], data.proc_info,
                                                        data.graph[v["task"]]))["summary"]["all"]
    return {"found": r["all"]["tp"], "mistakes": r["all"]["gt"], "alerts": r["all"]["alerts"],
            "install_found": r["install"]["tp"], "install": r["install"]["gt"],
            "false_alarms": r["all"]["alerts"] - r["all"]["tp"],
            "false_alarms_per_h": round(r["all"]["false_alarms_per_h"], 1),
            "part_alerts": sum(a.get("source") in ("part", "match") for res in results.values() for a in res["alerts"]),
            "psr_f1": round(psr["f1"], 3), "psr_pos": round(psr["pos"], 3), "psr_delay_s": round(psr["delay_s"], 1)}


def wrong_moments(rows: dict, cfg: dict, tasks: dict, results: dict) -> list[dict]:
    """For every annotated wrong install: what the model read around it, and whether the engine was watching."""
    out = []
    for rid in SPLITS[SPLIT]:
        task = tasks[task_of(rid)]
        for g in gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], cfg["fps"],
                             results[rid]["duration_s"], cfg["engine"]["order_grace_s"]):
            if g["type"] != "install":
                continue
            k = cfg["components"].index(g["component"])
            near = [w[k] for t, w, _ in rows[rid] if abs(t - g["t"]) <= cfg["evaluation"]["match_window_s"]]
            done = [e["t"] for e in results[rid]["log"] if e["step"] == g["step"] and e["status"] == "done"]
            out.append({"video": rid, "step": g["step"], "t": g["t"], "looks_within_30s": len(near),
                        "max_p_wrong": round(max(near), 3) if near else None,
                        "share_above_0_5": round(float(np.mean(np.array(near) > 0.5)), 3) if near else None,
                        "step_done_at": done})
    return out


def main() -> None:
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "ours")
    rids = SPLITS[SPLIT]
    p = probs(SPLIT, MODEL)
    rows = {rid: wrong_rows(read_jsonl(cfg["paths"]["out"] / rid / ASD[SPLIT]), p[rid]) for rid in rids}
    plain = {rid: run(rid, ACTIONS, STATE[SPLIT], cfg, tasks) for rid in rids}
    base = scored(f"{ACTIONS}_{STATE[SPLIT]}", plain, cfg, tasks, data)
    print(json.dumps({"baseline": base}), flush=True)

    def try_settings(settings: list[dict]) -> list[dict]:
        out = []
        for setting in settings:
            results = {rid: run(rid, ACTIONS, STATE[SPLIT], cfg, tasks, part=(rows[rid], setting)) for rid in rids}
            for rid in rids:
                if results[rid]["log"] != plain[rid]["log"]:
                    raise RuntimeError(f"{rid}: the combined check changed a step's status")
            out.append({**setting, **scored(f"{ACTIONS}_partcheck", results, cfg, tasks, data)})
            print(json.dumps(out[-1]), flush=True)
        return out

    grid = try_settings(SETTINGS)
    best = choose(base, grid)
    after = try_settings(AFTER_THE_RULE)
    report = {"rule": f"val: at least {MIN_NEW_FINDS} wrong installs newly found, no more new false alarms than new "
                      "finds, step F1 and POS not lower", "model": MODEL, "actions": ACTIONS, "baseline_val": base,
              "passed": best is not None, "chosen": best, "wrong_moments": wrong_moments(rows, cfg, tasks, plain),
              "grid": grid, "after_the_rule": after}
    write_json(OUT / "part_check_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("grid", "after_the_rule")}, indent=2))


if __name__ == "__main__":
    main()
