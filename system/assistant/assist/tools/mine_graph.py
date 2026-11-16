"""Draft the task graph from correct training recordings.

For every correct train + val recording (no incorrect install, ends in the task's goal state) the PSR
labels give the frame at which each catalog step was completed. Edge A -> B ("A is a prerequisite of B")
when A never finished after B and finished strictly before it at least once; steps that always finish at
the same moment get no edge. Edges implied by others are dropped (transitive reduction).

Severity: ``stop`` for steps other steps depend on (a mistake there blocks the rest), else ``warn``.
The output is a draft for hand review; the engine only reads the written YAML.

    python -m assist.tools.mine_graph
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from assist.common import ROOT, load_config, load_yaml
from assist.industreal import SPLITS, psr_states, task_of


def goal_state(task: dict, components: list[str]) -> list[int]:
    state = [int(c) for c in task["initial_state"]]
    for step in task["steps"].values():
        state[components.index(step["component"])] = step["value"]
    return state


def completion_frames(rid: str, task: dict, components: list[str], labels: Path) -> dict[str, int]:
    """First frame at which each step's component *changes to* the step's value (PSR labels).

    A change, not mere presence: in maintenance the rear wheel is on at the start, removed, then put back.
    """
    done: dict[str, int] = {}
    prev = [int(c) for c in task["initial_state"]]
    for frame, state in psr_states(rid, labels):
        for name, step in task["steps"].items():
            c = components.index(step["component"])
            if name not in done and state[c] == step["value"] != prev[c]:
                done[name] = frame
        prev = state
    return done


def correct_recordings(task_name: str, task: dict, components: list[str], labels: Path) -> list[str]:
    goal = goal_state(task, components)
    out = []
    for rid in SPLITS["train"] + SPLITS["val"]:
        if task_of(rid) != task_name:
            continue
        states = psr_states(rid, labels)
        if any(v == -1 for _, s in states for v in s) or states[-1][1] != goal:
            continue
        out.append(rid)
    return out


def mine(task_name: str, task: dict, components: list[str], labels: Path) -> tuple[dict[str, list[str]], list[str]]:
    rids = correct_recordings(task_name, task, components, labels)
    runs = [completion_frames(r, task, components, labels) for r in rids]
    names = list(task["steps"])
    # A never finished after B, and strictly before it at least once (always-simultaneous pairs get no edge)
    before = {(a, b) for a, b in itertools.permutations(names, 2)
              if all(a in run and b in run and run[a] <= run[b] for run in runs)
              and any(run[a] < run[b] for run in runs)}
    reduced = {(a, b) for a, b in before if not any((a, c) in before and (c, b) in before for c in names)}
    requires = {n: [a for a in names if (a, n) in reduced] for n in names}
    return requires, rids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=ROOT / "assist/configs/industreal_catalog.yaml", type=Path)
    parser.add_argument("--config", default=ROOT / "assist/configs/industreal.yaml", type=Path)
    parser.add_argument("--out", type=Path, help="Defaults to the config's graph path.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    catalog = load_yaml(args.catalog)
    out = {"fasten_actions": catalog["fasten_actions"], "tasks": {}}
    for task_name, task in catalog["tasks"].items():
        requires, rids = mine(task_name, task, cfg["components"], cfg["paths"]["labels"])
        dependents = {n for reqs in requires.values() for n in reqs}
        dependents |= {r for step in task["steps"].values() for r in step.get("extra_requires", [])}
        steps = {}
        for name, step in task["steps"].items():
            extra = [r for r in step.pop("extra_requires", []) if r not in requires[name]]
            steps[name] = {**step, "requires": requires[name] + extra,
                           "severity": "stop" if name in dependents else "warn"}
        out["tasks"][task_name] = {"title": task["title"], "initial_state": task["initial_state"],
                                   "mined_from": rids, "subgoals": task["subgoals"], "steps": steps}
        print(f"{task_name}: mined from {len(rids)} correct recordings")
        for name in steps:
            print(f"  {name:32s} <- {', '.join(requires[name]) or '-'}")
    path = args.out or cfg["graph_path"]
    header = ("# DRAFT written by assist/tools/mine_graph.py -- review and edit by hand.\n"
              "# requires: steps that must be done (or at least placed) before this one.\n"
              "# severity: info / warn / stop, used for mistakes on this step.\n")
    path.write_text(header + yaml.safe_dump(out, sort_keys=False, width=110), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
