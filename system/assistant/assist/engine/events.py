"""Turn noisy per-frame action labels into clean, causal start / end events.

A label becomes an event once it has been the top label for ``min_frames`` frames in a row, each with
confidence >= ``min_conf``. The event ends once another label (or nothing) has held that long. Both
moments are recorded twice: when the action really started / ended (``start`` / ``end``) and when an
online system could first know it (``known_at`` / ``ended_at``), which is what the engine uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    label: str
    start: float          # seconds, first frame of the action
    end: float            # seconds, just after its last frame
    known_at: float       # when the start was confirmed (start + debounce)
    ended_at: float       # when the end was confirmed
    conf: float = 1.0     # mean confidence over the frames it was the top label
    confs: list[float] = field(default_factory=list, repr=False)


def build_events(frames: list[tuple[float, str, float]], fps: float, min_frames: int, min_conf: float,
                 ignore: tuple[str, ...] = ("background",)) -> list[Event]:
    """``frames``: ``(time_s, top_label, confidence)`` per video frame, in time order."""
    dt = 1.0 / fps
    events: list[Event] = []
    active: Event | None = None
    last_seen = 0.0                       # last frame time at which the active label was on top
    run_label, run_start, run_len, run_confs = None, 0.0, 0, []

    def close(now: float) -> None:
        nonlocal active
        if active is not None:
            active.end, active.ended_at = last_seen + dt, now
            active.conf = round(sum(active.confs) / len(active.confs), 3)
            events.append(active)
            active = None

    for t, label, conf in frames:
        cand = label if conf >= min_conf and label not in ignore else None
        if cand == run_label:
            run_len += 1
            run_confs.append(conf)
        else:
            run_label, run_start, run_len, run_confs = cand, t, 1, [conf]
        if active is not None and cand == active.label:
            last_seen = t
            active.confs.append(conf)
        if run_len == min_frames and (active is None or cand != active.label):
            close(t)
            if cand is not None:
                active = Event(cand, run_start, t + dt, known_at=t, ended_at=t, confs=list(run_confs))
                last_seen = t
    if frames:
        close(frames[-1][0] + dt)
    return events


def events_from_segments(segments: list[dict]) -> list[Event]:
    """Ground-truth action segments (``{action, start, end}``) as events known the moment they happen."""
    return [Event(s["action"], s["start"], s["end"], known_at=s["start"], ended_at=s["end"]) for s in segments]
