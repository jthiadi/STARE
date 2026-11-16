"""Causal AR event extraction, shared by table building (train) and online inference.
from __future__ import annotations

A *trigger* fires at the first frame after a fastening-type action ends.  The *window* is the set
of action names seen (active at any frame) since the previous trigger, which tells us which
objects were handled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Actions whose end marks a likely step completion (chosen from train statistics, see build_tables.py)
TRIGGERS = {
    "tighten_nut", "tighten_acorn_nut", "loosen_nut", "loosen_acorn_nut",
    "fit_wheel", "plug_wheel", "pull_wheel", "fit_pulley",
}
USE_REMOVAL_TRIGGERS = False
# Removal completes when the part comes out, not when the nut is loosened
REMOVAL_TRIGGERS = {"pull_pin_short", "pull_pin_middle", "pull_pin_long", "pull_screw_pin",
                    "pull_small_screw_pin", "pull_objects", "pull_partial_model"}
# An action STARTING from this set means the worker moved on from the current fastening
MOVE_ON = {"check_instruction", "browse_instruction", "take_partial_model", "put_partial_model",
           "take_short_brace", "take_long_brace", "take_pin_short", "take_pin_long", "take_pin_middle",
           "take_screw_pin", "take_small_screw_pin", "take_wheel", "take_wing", "take_wing_beam",
           "take_pulley", "take_objects", "take_instruction", "check_partial_model", "put_objects"}
GENERIC = {"check_instruction", "browse_instruction", "align_objects", "take_instruction",
           "put_instruction", "check_partial_model"}


# For these actions we count how many fastening triggers happened since the action was last seen
KEY_ACTS = ["fit_wheel", "fit_pulley", "plug_pin_middle", "take_wheel", "fit_short_brace", "fit_long_brace",
            "plug_short_pin", "plug_screw_pin", "fit_wing_beam", "take_partial_model", "put_partial_model",
            "pull_pin_short", "pull_pin_middle", "pull_wheel", "take_long_brace", "take_short_brace"]


@dataclass
class ARTrigger:
    frame: int
    name: str
    window: frozenset
    duration: int = 0          # length of the action that just ended (known at its end)
    tsince: dict = field(default_factory=dict)   # KEY_ACT -> # earlier triggers since it was last seen


@dataclass
class EventExtractor:
    prev: set = field(default_factory=set)
    window: set = field(default_factory=set)
    use_removal: bool | None = None      # None -> module setting USE_REMOVAL_TRIGGERS
    started: dict = field(default_factory=dict)
    last_seen: dict = field(default_factory=dict)
    trig_frames: list = field(default_factory=list)

    def update(self, t: int, active_names: list[str]) -> list[ARTrigger]:
        cur = set(active_names)
        for nm in cur:
            self.last_seen[nm] = t
        self.window |= {n for n in cur if n not in GENERIC}
        ended = self.prev - cur
        for n in cur - self.prev:
            self.started[n] = t
        self.prev = cur
        out = []
        for n in sorted(ended & (TRIGGERS | (REMOVAL_TRIGGERS if (USE_REMOVAL_TRIGGERS if self.use_removal is None else self.use_removal) else set()))):
            ts = {a: sum(1 for f in self.trig_frames if f > self.last_seen[a]) for a in KEY_ACTS if a in self.last_seen}
            out.append(ARTrigger(t, n, frozenset(self.window), t - self.started.get(n, t), ts))
        self.trig_frames += [t] * len(out)
        if out:
            self.window = {n for n in cur if n not in GENERIC}
        return out
