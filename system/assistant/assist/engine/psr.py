"""Procedure step recognition (which steps were completed correctly, and when) from the state detector,
optionally fused with the action stream.

``AccumulatedConfidencePSR`` is the official IndustReal "expected" PSR, ported line for line from
TimSchoonbeek/IndustReal ``PSR/psr_utils.py``: every frame, the steps implied by the detected state (relative to
the current state) gain the detection's confidence, all other steps decay by ``decay``; a step whose
accumulated confidence passes ``threshold`` is completed.

``FusedPSR`` adds a second stream, in the spirit of STORM-PSR's two-stream accumulation: when the action
stream says a step was just finished (a nut fitted / tightened after the step's part was placed), that step
gains ``action_weight`` x the action's confidence on each of the next ``action_frames`` frames. Action
evidence only counts for a step whose prerequisites (task graph) are already completed, so a misread action
cannot complete a step out of order on its own. With ``action_weight = 0`` this is exactly the official PSR.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def make_entry(frame: int, action_id: int, proc_info: list, conf=1) -> dict:
    return {"frame": frame, "id": action_id, "description": proc_info[action_id]["description"], "conf": conf}


def convert_states_to_steps(prev: list, curr: list, frame: int, proc_info: list, conf=None):
    actions = []
    for k, (p, c) in enumerate(zip(prev, curr)):
        if p == c or (p == -1 and c == 0):
            continue
        if (p == -1 and c == 1) or (p == 0 and c == 1):
            action_id = k * 3 + 0
        elif p in (0, 1) and c == -1:
            action_id = k * 3 + 1
        else:                                   # 1 -> 0
            action_id = k * 3 + 2
        actions.append(make_entry(frame, action_id, proc_info, conf))
    return actions


def process_asd_predictions(file: Path, frames: int, min_conf: float = 0.0) -> list:
    """Per frame: [[class, conf, [x, y, w, h]], ...] from a released-format ASD CSV (boxes below min_conf dropped)."""
    data = [[i, []] for i in range(frames)]
    with open(file, encoding="utf-8") as fp:
        reader = csv.reader(fp)
        next(reader, None)
        for row in reader:
            f = int(row[1])
            if f < frames and float(row[3]) >= min_conf:
                data[f][1].append([int(row[2]), round(float(row[3]), 3), [float(v) for v in row[4:8]]])
    return data                                 # the official PSR reads each frame's first box (best first)


class AccumulatedConfidencePSR:
    """The official "expected" ASD-based PSR (accumulated confidence with decay)."""

    def __init__(self, proc_info: list, procedure: str, categories: list, initial_state: list[int],
                 threshold: float = 6.0, decay: float = 0.75):
        self.proc_info, self.threshold, self.decay, self.categories = proc_info, threshold, decay, categories
        self.cum_confs = np.zeros(len(proc_info))
        self.updated: list[int] = []
        self.y_hat: list[dict] = []
        self.expected = {a["id"] for a in proc_info if a[f"expected_in_{procedure}"]}
        self.current_state = list(initial_state)

    def add(self, step_id: int, conf: float) -> None:
        self.cum_confs[step_id] += conf
        self.updated.append(step_id)

    def update(self, pred: list, frame_n: int) -> None:
        if pred:
            state_str = self.categories[int(pred[0][0])]
            if state_str not in ("error_state", "background"):
                state = [int(c) for c in state_str]
                for s in convert_states_to_steps(self.current_state, state, frame_n, self.proc_info, conf=1):
                    self.add(s["id"], pred[0][1])
        self.complete(frame_n)
        for idx in range(len(self.proc_info)):
            if idx not in self.updated:
                self.cum_confs[idx] *= self.decay
        self.updated = []

    def complete(self, frame_n: int) -> None:
        for idx in np.nonzero(self.cum_confs > self.threshold)[0]:
            self.cum_confs[idx] = 0
            if idx in self.expected:
                info = self.proc_info[idx]
                self.current_state[info["state_idx"]] = 1 if info["install"] else 0
                self.y_hat.append(make_entry(frame_n, int(idx), self.proc_info))


class FusedPSR(AccumulatedConfidencePSR):
    def __init__(self, *args, requires: dict[int, list[int]], action_weight: float = 0.0, action_frames: int = 30,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.requires, self.action_weight, self.action_frames = requires, action_weight, action_frames
        self.boosts: list[list] = []            # [step id, per-frame amount, frames left]

    def action_done(self, step_id: int, conf: float) -> None:
        if self.action_weight > 0 and step_id in self.expected:
            self.boosts.append([step_id, self.action_weight * conf, self.action_frames])

    def _done(self, step_id: int) -> bool:
        info = self.proc_info[step_id]
        return self.current_state[info["state_idx"]] == (1 if info["install"] else 0)

    def update(self, pred: list, frame_n: int) -> None:
        for b in self.boosts:
            if not self._done(b[0]) and all(self._done(r) for r in self.requires.get(b[0], [])):
                self.add(b[0], b[1])
            b[2] -= 1
        self.boosts = [b for b in self.boosts if b[2] > 0]
        super().update(pred, frame_n)


def official_ids(graph_task: dict, components: list[str]) -> dict[str, int]:
    """Graph step -> official PSR step id (component k: 3k install, 3k + 2 removal)."""
    return {sid: 3 * components.index(s["component"]) + (0 if s["value"] == 1 else 2)
            for sid, s in graph_task["steps"].items()}


def requires_by_id(graph_task: dict, components: list[str]) -> dict[int, list[int]]:
    ids = official_ids(graph_task, components)
    return {ids[sid]: [ids[r] for r in s.get("requires", [])] for sid, s in graph_task["steps"].items()}


def action_completions(engine_result: dict, graph_task: dict, components: list[str], fps: float) -> dict[int, list]:
    """frame -> [(official step id, confidence)] for every step the actions-only engine finished from actions."""
    ids = official_ids(graph_task, components)
    out: dict[int, list] = {}
    for e in engine_result["log"]:
        if e["status"] == "done" and e["source"] == "actions":
            out.setdefault(int(round(e["t"] * fps)), []).append((ids[e["step"]], 1.0))
    return out
