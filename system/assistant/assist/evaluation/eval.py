"""Evaluate engine alerts against ground-truth mistakes and write a markdown table.

Ground truth comes from the official PSR labels (assembly state over time), replayed through the same
task graph:
  install   a component becomes -1 (installed incorrectly), or a component this task never uses appears
  order     a step finishes while a prerequisite of the same subgoal is not finished (10 s grace, as the engine)
  skipped   a step of a later subgoal finishes while an earlier-subgoal step was never done, or the
            recording ends with a step not done
An alert matches an unmatched ground-truth mistake of the same type and step (component for wrong parts)
within +-``match_window_s``. Reported per type: precision, recall, mean detection delay (alert - mistake),
false alarms per hour (unmatched alerts), and for early warnings the mean lead time before the mistakes
they announced (warning for the same step within ``warning_horizon_s`` before it).

    python -m assist.evaluation.eval --runs gt_gt gt_none pred_asd pred_none
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from assist.common import ROOT, load_config, write_json
from assist.engine.graph import Task, load_tasks
from assist.industreal import SPLITS, n_frames, psr_states, task_of

TYPE_OF = {"wrong_order": "order", "skipped_step": "skipped", "incorrect_install": "install",
           "possible_incorrect_install": "install", "wrong_part": "install"}
TYPES = ("skipped", "order", "install")
RUN_NAMES = {"gt_gt": "oracle: GT actions + GT state", "gt_none": "GT actions, no state",
             "gt_asd": "GT actions + ASD", "pred_asd": "predicted actions + ASD", "pred_vjepa_asd": "online V-JEPA 2 actions + ASD",
             "pred_vjepap_asd": "live system: TCN + V-JEPA 2 actions + ASD",
             "pred_vjepap_gt": "live actions (TCN + V-JEPA 2) + GT state",
             "pred_vjepap_none": "live actions (TCN + V-JEPA 2), no state",
             "pred_gt": "predicted actions + GT state", "pred_none": "predicted actions, no state",
             "baseline_psr": "baseline: official IndustReal PSR on ASD + the same mistake rules"}


def gt_mistakes(rid: str, task: Task, components: list[str], labels, fps: float, duration: float,
                grace: float, changes: list | None = None) -> list[dict]:
    """Mistakes in a state sequence ``[(frame, state), ...]``: the annotated one unless ``changes`` is given."""
    done: dict[str, float] = {}
    out, seen = [], set()

    def add(kind, step, t, component=None):
        key = (kind, step, component)
        if key not in seen:
            seen.add(key)
            out.append({"type": kind, "step": step, "component": component, "t": round(t, 2)})

    prev = list(task.initial_state)
    pending = []                                     # (deadline, step, missing, earlier)
    for frame, state in psr_states(rid, labels) if changes is None else changes:
        t = frame / fps
        for d, s, missing, earlier in [p for p in pending if p[0] <= t]:
            for m in missing:
                if m not in done:
                    add("order", s, d)
            for e in earlier:
                if e not in done:
                    add("skipped", e, d)
        pending = [p for p in pending if p[0] > t]
        changed = sorted((i for i in range(len(state)) if state[i] != prev[i]),
                         key=lambda i: min((task.order.index(s) for s in task.steps_for(components[i], state[i])),
                                           default=99))
        for i in changed:
            comp, v = components[i], state[i]
            if v == -1:                              # done, but wrong: an install mistake, not a skipped step
                sid = (task.steps_for(comp, 1) or [None])[0]
                add("install", sid, t, comp)
                if sid:
                    done[sid] = t
                continue
            steps = [s for s in task.steps_for(comp, v) if s not in done]
            if steps:
                s = steps[0]
                done[s] = t
                sg = task.steps[s].subgoal
                missing = [p for p in task.ancestors(s) if p not in done and task.steps[p].subgoal == sg]
                earlier = [e for e in task.order if task.steps[e].subgoal < sg and e not in done]
                if missing or earlier:
                    pending.append((t + grace, s, missing, earlier))
            elif not task.steps_for(comp, v) and v == 1 and task.initial_state[i] == 0:
                add("install", None, t, comp)
            elif not task.steps_for(comp, v):
                for s in task.steps_for(comp, 1 - v):
                    done.pop(s, None)
        prev = state
    for d, s, missing, earlier in pending:
        for m in missing:
            if m not in done:
                add("order", s, d)
        for e in earlier:
            if e not in done:
                add("skipped", e, d)
    for s in task.order:
        if s not in done:
            add("skipped", s, duration)
    return out


def collapse(alerts: list[dict], window: float) -> list[dict]:
    """One alert per mistake: a "possible" incorrect install followed by the confirmed one (same step, within
    the window) is a single detection, timed by the first."""
    out = []
    for a in alerts:
        if any(TYPE_OF[b["type"]] == TYPE_OF[a["type"]] and b["step"] == a["step"] and b.get("component") ==
               a.get("component") and a["t"] - b["t"] <= window for b in out):
            continue
        out.append(a)
    return out


def match(gts: list[dict], alerts: list[dict], window: float) -> list[tuple[dict, dict]]:
    """Greedy one-to-one matching, closest in time first."""
    pairs = []
    for g in gts:
        for a in alerts:
            if TYPE_OF.get(a["type"]) != g["type"]:
                continue
            same = a["step"] == g["step"] if g["step"] else a.get("component") == g["component"]
            if g["type"] == "install" and not same:
                same = a.get("component") == g["component"]
            if same and abs(a["t"] - g["t"]) <= window:
                pairs.append((abs(a["t"] - g["t"]), id(g), id(a), g, a))
    used_g, used_a, out = set(), set(), []
    for _, ig, ia, g, a in sorted(pairs, key=lambda p: p[0]):
        if ig not in used_g and ia not in used_a:
            used_g.add(ig)
            used_a.add(ia)
            out.append((g, a))
    return out


def evaluate(run: str, rids: list[str], cfg: dict, tasks: dict) -> dict:
    ev = cfg["evaluation"]
    hours = 0.0
    stats = {k: {"gt": 0, "alerts": 0, "tp": 0, "delays": [], "leads": [], "warned": 0} for k in TYPES}
    warnings = cannot = 0
    per_video = {}
    for rid in rids:
        res = json.loads((cfg["paths"]["out"] / rid / f"engine_{run}.json").read_text(encoding="utf-8"))
        task = tasks[task_of(rid)]
        duration = res["duration_s"]
        hours += duration / 3600
        gts = gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], cfg["fps"], duration,
                          cfg["engine"]["order_grace_s"])
        mistakes = collapse([a for a in res["alerts"] if a["kind"] == "mistake"], ev["match_window_s"])
        warns = [a for a in res["alerts"] if a["type"] == "early_warning"]
        warnings += len(warns)
        cannot += sum(1 for a in res["alerts"] if a["type"] == "cannot_verify")
        pairs = match(gts, mistakes, ev["match_window_s"])
        for k in TYPES:
            stats[k]["gt"] += sum(1 for g in gts if g["type"] == k)
            stats[k]["alerts"] += sum(1 for a in mistakes if TYPE_OF[a["type"]] == k)
        for g, a in pairs:
            stats[g["type"]]["tp"] += 1
            stats[g["type"]]["delays"].append(a["t"] - g["t"])
        for g in gts:
            before = [w for w in warns if (w["step"] == g["step"] or g["step"] in w.get("missing", []))
                      and 0 < g["t"] - w["t"] <= ev["warning_horizon_s"]]
            if before:
                stats[g["type"]]["warned"] += 1
                stats[g["type"]]["leads"].append(g["t"] - min(w["t"] for w in before))
        per_video[rid] = {"gt": gts, "matched": [[g["type"], g["step"], g["t"], a["t"]] for g, a in pairs],
                          "alerts": len(mistakes)}
    rows = {}
    for k, s in stats.items():
        fa = s["alerts"] - s["tp"]
        rows[k] = {"gt": s["gt"], "alerts": s["alerts"], "tp": s["tp"],
                   "precision": s["tp"] / s["alerts"] if s["alerts"] else None,
                   "recall": s["tp"] / s["gt"] if s["gt"] else None,
                   "delay_s": sum(s["delays"]) / len(s["delays"]) if s["delays"] else None,
                   "lead_s": sum(s["leads"]) / len(s["leads"]) if s["leads"] else None,
                   "warned": s["warned"], "false_alarms_per_h": fa / hours}
    tot = {key: sum(rows[k][key] for k in TYPES) for key in ("gt", "alerts", "tp", "warned")}
    all_delays = [d for k in TYPES for d in stats[k]["delays"]]
    all_leads = [d for k in TYPES for d in stats[k]["leads"]]
    rows["all"] = {**tot, "precision": tot["tp"] / tot["alerts"] if tot["alerts"] else None,
                   "recall": tot["tp"] / tot["gt"] if tot["gt"] else None,
                   "delay_s": sum(all_delays) / len(all_delays) if all_delays else None,
                   "lead_s": sum(all_leads) / len(all_leads) if all_leads else None,
                   "false_alarms_per_h": (tot["alerts"] - tot["tp"]) / hours}
    return {"run": run, "videos": len(rids), "hours": round(hours, 2), "rows": rows,
            "early_warnings_per_h": warnings / hours, "cannot_verify_per_h": cannot / hours,
            "per_video": per_video}


def _f(x, pct=False, digits=1):
    if x is None:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def markdown(results: list[dict], split: str) -> str:
    lines = [f"# Mistake detection on the IndustReal {split} set", "",
             f"{results[0]['videos']} videos, {results[0]['hours']:.2f} h. Ground truth from the official PSR "
             "labels; counts are shown so small rows are not over-read.", ""]
    for r in results:
        lines += [f"## {RUN_NAMES.get(r['run'], r['run'])}", "",
                  "| Mistake type | GT | Alerts | Precision | Recall | Mean delay (s) | Warned before | "
                  "Mean lead (s) | False alarms / h |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in TYPES + ("all",):
            row = r["rows"][k]
            lines.append(f"| {k} | {row['gt']} | {row['alerts']} | {_f(row['precision'], True)} "
                         f"({row['tp']}/{row['alerts']}) | {_f(row['recall'], True)} ({row['tp']}/{row['gt']}) | "
                         f"{_f(row['delay_s'])} | {row['warned']}/{row['gt']} | {_f(row['lead_s'])} | "
                         f"{_f(row['false_alarms_per_h'])} |")
        lines += ["", f"Early warnings: {r['early_warnings_per_h']:.1f} / h. "
                      f"\"Cannot verify\": {r['cannot_verify_per_h']:.1f} / h.", ""]
    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m assist.evaluation.eval`._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["gt_gt"], help="engine_<actions>_<state> runs to score.")
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    args = parser.parse_args()
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    results = [evaluate(run, SPLITS[args.split], cfg, tasks) for run in args.runs]
    out = cfg["paths"]["out"] / "eval"
    write_json(out / f"eval_{args.split}.json", results)
    md = markdown(results, args.split)
    (out / f"eval_{args.split}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {(out / f'eval_{args.split}.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
