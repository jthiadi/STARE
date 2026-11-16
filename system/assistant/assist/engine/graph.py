"""The task graph (one per task: steps, prerequisites, subgoals) as read from the graph YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from assist.common import load_yaml


@dataclass
class Step:
    id: str
    name: str
    component: str
    value: int                 # component value once the step is done (1 installed, 0 removed)
    take: list[str]
    do: list[str]
    done_on: str               # "fasten" or "do"
    requires: list[str]
    severity: str
    subgoal: int               # index into Task.subgoals
    note: str = ""


@dataclass
class Task:
    name: str
    title: str
    initial_state: list[int]
    steps: dict[str, Step]
    subgoals: list[tuple[str, list[str]]]
    fasten_actions: list[str]
    order: list[str] = field(default_factory=list)      # topological order of the steps

    def __post_init__(self):
        self.order = _topological(self.steps)
        self._ancestors = {s: self._collect(s) for s in self.steps}

    def _collect(self, sid: str) -> set[str]:
        out, todo = set(), list(self.steps[sid].requires)
        while todo:
            p = todo.pop()
            if p not in out:
                out.add(p)
                todo.extend(self.steps[p].requires)
        return out

    def ancestors(self, sid: str) -> list[str]:
        """Every step that must come before ``sid`` (direct and indirect), in topological order."""
        return [s for s in self.order if s in self._ancestors[sid]]

    def descendants(self, sid: str) -> list[str]:
        """Every step that needs ``sid`` first, in topological order."""
        return [s for s in self.order if sid in self._ancestors[s]]

    def candidates(self, action: str, field_name: str) -> list[str]:
        """Steps whose ``take`` or ``do`` list contains the action, in graph order."""
        return [s for s in self.order if action in getattr(self.steps[s], field_name)]

    def steps_for(self, component: str, value: int) -> list[str]:
        return [s for s in self.order if self.steps[s].component == component and self.steps[s].value == value]


def _topological(steps: dict[str, Step]) -> list[str]:
    order, seen = [], set()

    def visit(s: str, stack: tuple = ()) -> None:
        if s in seen:
            return
        if s in stack:
            raise ValueError(f"cycle in task graph through {s}")
        for p in steps[s].requires:
            visit(p, stack + (s,))
        seen.add(s)
        order.append(s)

    for s in steps:                      # file order breaks ties, so the YAML order is kept where possible
        visit(s)
    return order


def load_tasks(path: Path | str) -> dict[str, Task]:
    data = load_yaml(path)
    tasks = {}
    for name, t in data["tasks"].items():
        sub = list(t["subgoals"].items())
        index = {s: i for i, (_, members) in enumerate(sub) for s in members}
        steps = {sid: Step(id=sid, name=s["name"], component=s["component"], value=int(s["value"]),
                           take=list(s.get("take", [])), do=list(s.get("do", [])), done_on=s["done_on"],
                           requires=list(s.get("requires", [])), severity=s.get("severity", "warn"),
                           subgoal=index.get(sid, len(sub)), note=s.get("note", ""))
                 for sid, s in t["steps"].items()}
        tasks[name] = Task(name, t["title"], [int(c) for c in t["initial_state"]], steps, sub,
                           list(data["fasten_actions"]))
    return tasks
