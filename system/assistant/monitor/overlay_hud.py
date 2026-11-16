"""Shared OpenCV HUD helpers for SOP / GT video overlays."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def fit_text(text: str, maximum: int = 118) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def short_step_title(step_id: str, instruction: str) -> str:
    text = " ".join(str(instruction or "").split())
    if not text:
        return str(step_id)
    lowered = text.lower()
    for prefix in ("install the ", "install a ", "install "):
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    return fit_text(f"{step_id} {text}", 70)


def draw_check_mark(frame: np.ndarray, x: int, y: int, size: int, color) -> None:
    """Simple two-stroke check inside a checkbox at (x, y)."""
    p1 = (x + int(size * 0.18), y + int(size * 0.55))
    p2 = (x + int(size * 0.42), y + int(size * 0.78))
    p3 = (x + int(size * 0.82), y + int(size * 0.22))
    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    cv2.line(frame, p2, p3, color, 2, cv2.LINE_AA)


def draw_sop_step_rail(
    frame: np.ndarray,
    step_order: list[str],
    step_labels: dict[str, str],
    completed: list[str] | set[str] | None,
    current_step: str | None,
    ready: list[str] | set[str] | None = None,
) -> int:
    """Left-side SOP checklist: tick completed steps, color the current step.

    Returns the panel width in pixels (0 if nothing drawn).
    """
    if not step_order:
        return 0
    height, width = frame.shape[:2]
    n = len(step_order)
    header_h = 28
    row_h = max(22, min(34, (height - header_h - 8) // max(n, 1)))
    panel_w = min(300, max(210, width // 4))
    panel_h = min(height, header_h + row_h * n + 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (16, 20, 26), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (55, 65, 78), 1)
    cv2.putText(
        frame, "SOP STEPS", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 220, 230), 1, cv2.LINE_AA,
    )

    done = {str(sid) for sid in (completed or [])}
    ready_set = {str(sid) for sid in (ready or [])}
    current = str(current_step or "")
    if current in {"", "unknown", "None"}:
        current = ""

    box = 14
    for index, sid in enumerate(step_order):
        y0 = header_h + index * row_h
        y1 = y0 + row_h - 2
        is_done = sid in done
        is_current = bool(current) and sid == current
        is_ready = (not is_done) and (not is_current) and sid in ready_set

        if is_current:
            cv2.rectangle(frame, (2, y0), (panel_w - 2, y1), (35, 70, 55), -1)
            cv2.rectangle(frame, (2, y0), (6, y1), (80, 230, 140), -1)
            text_color = (120, 255, 180)
            mark_color = (120, 255, 180)
        elif is_done:
            text_color = (140, 210, 160)
            mark_color = (60, 210, 120)
        elif is_ready:
            text_color = (160, 200, 230)
            mark_color = (140, 180, 210)
        else:
            text_color = (130, 140, 150)
            mark_color = (90, 100, 110)

        bx = 10
        by = y0 + max(2, (row_h - box) // 2 - 1)
        cv2.rectangle(frame, (bx, by), (bx + box, by + box), mark_color, 1 if not is_done else -1)
        if is_done:
            draw_check_mark(frame, bx, by, box, (18, 28, 22))
        elif is_current:
            cv2.circle(frame, (bx + box // 2, by + box // 2), 3, mark_color, -1, cv2.LINE_AA)

        label = str(step_labels.get(sid) or "")
        short = short_step_title(sid, label) if label else sid
        max_chars = max(18, (panel_w - 36) // 7)
        line = fit_text(short, max_chars)
        cv2.putText(
            frame, line, (bx + box + 8, y0 + int(row_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA,
        )
    return panel_w


def step_labels_from_sop(sop: dict[str, Any] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for step in (sop or {}).get("steps") or []:
        sid = str(step.get("id") or "")
        if sid:
            labels[sid] = str(step.get("instruction") or sid).strip()
    return labels


def step_order_from_sop(sop: dict[str, Any] | None, fallback: list[str] | None = None) -> list[str]:
    order = [
        str(step.get("id") or "")
        for step in (sop or {}).get("steps") or []
        if step.get("id")
    ]
    return order or list(fallback or [])
