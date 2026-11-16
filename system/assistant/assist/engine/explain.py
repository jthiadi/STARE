"""Plain-language explanations: ``{title, why, fix_steps[], rule, evidence}`` from templates and the graph.

No language model: every sentence is built from the alert and the task graph, so the fix steps are always
consistent with the procedure (missing prerequisites in order, plus whatever must be removed first).
"""
from __future__ import annotations

from assist.engine.graph import Task

DONE = 3          # engine status of a finished step (see engine.py)
PLACED = 2


def _name(task: Task, sid: str) -> str:
    return task.steps[sid].name


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _verb(task: Task, sid: str) -> str:
    """'Install the front bracket' / 'Remove the rear wheel assembly' (removal names already say remove)."""
    name = _name(task, sid)
    return name[0].upper() + name[1:] if task.steps[sid].value == 0 else f"Install the {name}"


def _undo(task: Task, sid: str) -> str:
    name = _name(task, sid)
    if task.steps[sid].value == 0:                       # undoing a removal = putting it back
        return f"Put back the {name.removeprefix('remove the ')}"
    return f"Remove the {name}"


def redo_plan(task: Task, status: dict[str, int], redo: list[str], missing: list[str]) -> list[str]:
    """Fix steps: undo every placed step that has to come off (latest first), do the missing steps (and
    their own unfinished prerequisites) in order, then redo the undone ones in order."""
    missing = set(missing) | {a for m in missing for a in task.ancestors(m) if status.get(a, 0) < DONE}
    undo = set(redo)
    for m in missing:
        undo |= {d for d in task.descendants(m) if status.get(d, 0) >= PLACED}
    undo_order = [s for s in reversed(task.order) if s in undo]
    plan = [_undo(task, s) for s in undo_order]
    plan += [_verb(task, m) for m in task.order if m in missing]
    plan += [_verb(task, s) for s in task.order if s in undo]
    return plan


def explain(alert: dict, task: Task, status: dict[str, int]) -> dict:
    kind, sid, missing = alert["type"], alert.get("step"), alert.get("missing", [])
    step = task.steps[sid] if sid else None
    note = f" {step.note}" if step and step.note else ""
    if kind == "early_warning":
        title = f"Wait: {_name(task, sid)} is not next"
        why = f"You picked up the part for the {_name(task, sid)}, but {_join([_name(task, m) for m in missing])} " \
              f"must be done first.{note}"
        fix = [_verb(task, m) for m in missing] + [f"Then {_verb(task, sid).lower()}"]
    elif kind == "wrong_order":
        title = f"Out of order: {_name(task, sid)}"
        why = f"The {_name(task, sid)} was done before {_join([_name(task, m) for m in missing])}.{note}"
        fix = redo_plan(task, status, [sid], missing)
    elif kind == "skipped_step":
        title = f"Skipped step: {_name(task, sid)}"
        later = alert.get("trigger")
        why = (f"The {_name(task, sid)} was never done, but the procedure has moved on"
               + (f" ({_name(task, later)} is already done)." if later else " (the task has ended).") + note)
        fix = redo_plan(task, status, [], [sid])
    elif kind == "incorrect_install":
        title = f"Installed incorrectly: {_name(task, sid)}"
        why = (f"The assembly-state check sees the {_name(task, sid)} in a wrong position."
               f"{note}")
        fix = redo_plan(task, status, [sid], []) + ["Check the orientation against the instructions"]
    elif kind == "possible_incorrect_install":
        title = f"Check the {_name(task, sid)}"
        why = (f"The {_name(task, sid)} was just finished, but within {alert['window_s']:.0f} s the camera "
               f"never saw the assembly in the expected state.")
        fix = [f"Look at the {_name(task, sid)}: is it fully seated and the right way round?",
               "If not: " + "; ".join(redo_plan(task, status, [sid], [])).lower()]
    elif kind == "wrong_part":
        comp = alert["component"].replace("_", " ")
        title = f"Wrong part: {comp}"
        why = f"The camera sees a {comp} on the model, but {task.title.lower()} does not use it."
        nxt = [s for s in task.order if status.get(s, 0) < DONE
               and all(status.get(p, 0) >= DONE for p in task.steps[s].requires)]
        fix = [f"Remove the {comp}"] + ([f"Continue with: {_join([_name(task, s) for s in nxt])}"] if nxt else [])
    elif kind == "cannot_verify":
        title = f"Cannot verify: {_name(task, sid)}" if sid else "Cannot verify"
        why = alert.get("reason", "The evidence is too weak to decide.")
        fix = ["Keep the assembly in view of the camera for a moment"]
    else:
        raise ValueError(f"unknown alert type {kind}")
    return {"title": title, "why": why, "fix_steps": fix, "rule": RULES[kind], "evidence": alert.get("evidence", {})}


RULES = {
    "early_warning": "TAKE or start of a step whose prerequisites have not been started",
    "wrong_order": "step finished while a prerequisite step was not finished",
    "skipped_step": "a later subgoal (or the task) finished while this step was never done",
    "incorrect_install": "assembly-state detector reports an incorrect state after this install",
    "possible_incorrect_install": "expected assembly state not seen within T seconds after the install",
    "wrong_part": "assembly state contains a component this task does not use",
    "cannot_verify": "low action confidence, or the assembly is hidden (hands / not detected)",
}
