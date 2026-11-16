"""Baseline for live mistake detection on IndustReal: the official PSR, followed by the standard mistake rules.

No published method reports live mistake detection on IndustReal, so this builds the obvious one from the
paper's own pipeline. The official IndustReal PSR (released ASD predictions, accumulated confidence; the base
of STORM-PSR) outputs completed steps as it goes. Those steps are replayed into a state sequence and checked with
exactly the rules that define the ground-truth mistakes (``eval.gt_mistakes``): order, skipped, install. The
official PSR only ever completes correct steps (it ignores ``error_state``), so it cannot report an
incorrectly installed part.

Writes ``output/assist/<rid>/engine_baseline_psr.json`` in the engine's result format, so it is scored like any
engine run:

    python -m assist.evaluation.baseline
    python -m assist.evaluation.eval --runs baseline_psr pred_asd
"""
from __future__ import annotations

import argparse
import json

from assist.common import load_config, write_json
from assist.engine.graph import load_tasks
from assist.evaluation.eval import gt_mistakes
from assist.evaluation.psr_eval import FPS, Data, run_fused
from assist.industreal import SPLITS

ALERT_TYPE = {"order": "wrong_order", "skipped": "skipped_step", "install": "incorrect_install"}


def state_changes(steps: list[dict], initial: list[int]) -> list[tuple[int, list[int]]]:
    """The state after each frame's steps, as ``[(frame, state), ...]`` from the initial state. Official step
    ids: component k is installed (3k), installed incorrectly (3k + 1, state -1) or removed (3k + 2)."""
    state, out = list(initial), [(0, list(initial))]
    for s in steps:
        state[s["id"] // 3] = (1, -1, 0)[s["id"] % 3]
        if out[-1][0] == s["frame"]:
            out[-1] = (s["frame"], list(state))
        else:
            out.append((s["frame"], list(state)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("test",), help="Released ASD predictions: test only.")
    args = parser.parse_args()
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "release")
    for rid in SPLITS[args.split]:
        v = data.video(rid)
        task = tasks[v["task"]]
        duration = v["frames"] / FPS
        changes = state_changes(run_fused(v, data), task.initial_state)
        mistakes = gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], FPS, duration,
                               cfg["engine"]["order_grace_s"], changes=changes)
        alerts = [{"t": m["t"], "type": ALERT_TYPE[m["type"]], "kind": "mistake", "step": m["step"],
                   "component": m["component"]} for m in mistakes]
        write_json(cfg["paths"]["out"] / rid / "engine_baseline_psr.json",
                   {"rid": rid, "task": v["task"], "actions": "none", "state": "asd", "duration_s": round(duration, 1),
                    "alerts": alerts})
        print(json.dumps({"video": rid, "steps": len(changes) - 1, "alerts": len(alerts)}), flush=True)


if __name__ == "__main__":
    main()
