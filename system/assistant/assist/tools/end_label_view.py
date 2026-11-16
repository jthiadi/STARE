"""The action-end label on the overlay: a chip in the video's corner, and a thin strip on the timeline.

A label only appears once its pass over the action has finished, ``ACTION_PASS_S`` after the action's end (measured on a free
RTX 2080 Ti, ``run_action_end --benchmark``), or later if the previous pass is still running. The running label is unchanged.
"""
from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw

from assist.extract.run_action_end import label_delays
from assist.tools.render_overlay import CARD, INK, VH, describe, ui, verb_color

ACTION_PASS_S = 1.21
HOLD_S = 4.0
STRIP_Y, STRIP_H = 66, 12    # between the timeline's live row and its ground-truth row


def with_times(rows: list[dict], service_s: float = ACTION_PASS_S) -> list[dict]:
    """The rows of ``action_end.jsonl``, each with ``label_at``: when its label is ready."""
    delays = label_delays([r["ended_at"] for r in rows], service_s)
    return [{**r, "label_at": round(r["ended_at"] + d, 6)} for r, d in zip(rows, delays)]


def shown_final(rows: list[dict], t: float, hold_s: float = HOLD_S) -> dict | None:
    """The newest label ready by t, for ``hold_s`` after it became ready. Never one that is not ready yet."""
    ready = [r for r in rows if r["label_at"] <= t]
    return ready[-1] if ready and t < ready[-1]["label_at"] + hold_s else None


def draw_strip(image: Image.Image, rows: list[dict], t: float, duration: float, left: float, right: float) -> None:
    """Each action whose label is ready by t, at its place in the video, in its verb's colour."""
    pen = ImageDraw.Draw(image)
    y0, y1 = STRIP_Y - STRIP_H // 2, STRIP_Y + STRIP_H // 2

    def x(s: float) -> float:
        return left + (right - left) * min(max(s / duration, 0.0), 1.0)

    for row in rows:
        if row["label_at"] <= t:
            pen.rectangle([x(row["start"]), y0, max(x(row["start"]) + 1, x(row["end"])), y1], fill=verb_color(row["label"]))


def final_events(rows: list[dict]) -> list[dict]:
    """For a replay made after every label is final: each action shows its final label over its whole span. Live, the label
    only exists ``ACTION_PASS_S`` or more after the action ends, so a video using this must say the labels were filled in."""
    return [{"label": r["label"], "start": r["start"], "end": r["end"], "known_at": r["start"], "ended_at": r["end"],
             "conf": r["conf"]} for r in rows]


def chip_text(label: str) -> str:
    return f"Final label: {describe(label)}"


@lru_cache(maxsize=None)
def _chip(label: str) -> Image.Image:
    text, font = chip_text(label), ui(24, "semibold")
    width, height = int(font.getlength(text)) + 44, 50
    card = Image.new("RGB", (width, height), INK)
    ImageDraw.Draw(card).text((22, height / 2), text, font=font, fill=CARD, anchor="lm")
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, width - 1, height - 1], radius=14, fill=235)
    card.putalpha(mask)
    return card


def draw_chip(image: Image.Image, label: str) -> None:
    chip = _chip(label)
    image.paste(chip, (16, VH - chip.height - 16), chip)
