"""IndustReal ``AR_labels.csv`` rows as time intervals, and the kit object each AR action refers to."""
from __future__ import annotations

import csv
import re
from pathlib import Path


def _frame_index(name: str) -> int:
    return int(Path(str(name)).stem)


def load_ar_intervals(path: Path, *, source_fps: float = 10.0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 5:
                continue
            start_f = _frame_index(row[3])
            end_f = _frame_index(row[4])
            rows.append({
                "action_id": int(row[1]),
                "action": str(row[2]),
                "start_s": start_f / source_fps,
                "end_s": end_f / source_fps,
            })
    rows.sort(key=lambda item: (item["start_s"], item["end_s"], item["action"]))
    return rows


def active_ar(intervals: list[dict], timestamp_s: float) -> list[dict]:
    return [
        row for row in intervals
        if row["start_s"] - 1e-9 <= timestamp_s <= row["end_s"] + 1e-9
    ]


# AR action label → object noun used by our kit timeline (or None = no object GT).
_AR_ACTION_OBJECT: dict[str, str | None] = {
    "take_nut": "nut",
    "fit_nut": "nut",
    "tighten_nut": "nut",
    "take_tooth_washer": "tooth_washer",
    "fit_tooth_washer": "tooth_washer",
    "take_round_washer": "round_washer",
    "fit_round_washer": "round_washer",
    "take_acorn_nut": "acorn_nut",
    "fit_acorn_nut": "acorn_nut",
    "put_acorn_nut": "acorn_nut",
    "tighten_acorn_nut": "acorn_nut",
    "take_pin_short": "pin_short",
    "plug_short_pin": "pin_short",
    "take_pin_middle": "pin_middle",
    "plug_pin_middle": "pin_middle",
    "take_pin_long": "pin_long",
    "pull_pin_long": "pin_long",
    "plug_screw_pin": "screw_pin",
    "take_short_brace": "short_brace",
    "fit_short_brace": "short_brace",
    "take_wheel": "wheel",
    "fit_wheel": "wheel",
    "take_pulley": "pulley",
    "fit_pulley": "pulley",
    "take_wing": "wing",
    "fit_wing": "wing",
    "take_wing_beam": "wing",
    "fit_wing_beam": "wing",
    "take_partial_model": "partial_model",
    "put_partial_model": "partial_model",
    # No unique part in our 15-class kit head:
    "check_instruction": None,
    "browse_instruction": None,
    "align_objects": None,
    "take_objects": None,
    "plug_objects": None,
}

_FALLBACK_OBJECT = re.compile(
    r"(?:take|fit|plug|put|pull|tighten)_"
    r"(pin_short|pin_middle|pin_long|screw_pin|short_brace|long_brace|"
    r"tooth_washer|round_washer|acorn_nut|partial_model|wing_beam|"
    r"front_bracket|nut|wheel|pulley|wing)"
)


def gt_object_from_action(action: str) -> str | None:
    key = str(action or "").strip()
    if key in _AR_ACTION_OBJECT:
        return _AR_ACTION_OBJECT[key]
    match = _FALLBACK_OBJECT.fullmatch(key)
    if not match:
        return None
    noun = match.group(1)
    return "wing" if noun == "wing_beam" else noun
