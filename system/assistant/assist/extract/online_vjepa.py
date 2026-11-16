"""Online V-JEPA 2 labels for the live assistant: the 77.2 % ensemble without future frames.

The live model's events (``build_events``) give each action's start. From the moment an action is confirmed
(``known_at``) the ensemble scores the window [start, now] every ``every`` frames, and once more over the action's
own frames when its end is confirmed (``ended_at``); the action-order prior adds the earlier actions only (forward
filtering). Each frame then shows the live model's own output with its action scores pulled towards the latest
ensemble result: ``(1 - w) log p_live(frame) + w (log p_ensemble + prior)``, background probability kept from the
live model, so ``w = 0`` is exactly the live model. A result is used from ``delay`` frames after its update runs
(the compute budget). Pure numpy; the ensemble scores come from ``run_ar_vjepa``.
"""
from __future__ import annotations

import numpy as np

from assist.engine.events import Event
from scripts.recognition.sequence_recognition import _lse, log_prior


def updates(events: list[Event], fps: float, n_frames: int, every: int = 5) -> list[tuple[int, int, int, int]]:
    """``(event index, window start, window end, frame the update runs)``; window frames inclusive, end <= run frame."""
    last = n_frames - 1
    out = []
    for i, e in enumerate(events):
        start, known, ended = (min(round(t * fps), last) for t in (e.start, e.known_at, e.ended_at))
        out += [(i, start, at, at) for at in range(known, ended, every)]
        out.append((i, start, max(start, min(round(e.end * fps) - 1, ended)), ended))
    return out


def shown_probs(live: np.ndarray, events: list[Event], upd: list[tuple[int, int, int, int]], member: np.ndarray,
                w: float, strength: float, smoothing: float, first: np.ndarray, pairs: np.ndarray,
                delay: int = 5) -> np.ndarray:
    """(T, 75) probabilities shown per frame (column 0 = background). ``live``: the live model's (T, 75);
    ``member``: the ensemble's (len(upd), 74) per update."""
    shown = live.copy()
    start_lp, trans = log_prior(first, pairs, smoothing)
    prev = None                                        # filtered log posterior of the previous action at its end
    for j, (i, s, e, at) in enumerate(upd):
        ens = np.log(member[j] + 1e-8)
        if strength:
            ens = ens + strength * (start_lp if prev is None else _lse(prev[:, None] + trans, 0))
        if j + 1 == len(upd) or upd[j + 1][0] != i:     # the action's final window: the filter state for the next
            la = live[s:e + 1, 1:].mean(0)
            post = (1 - w) * np.log(la / la.sum() + 1e-8) + w * ens
            prev = post - _lse(post, 0)
            continue
        stop = min(upd[j + 1][3] + delay, _ended(upd, j), len(live))   # next result, or the end is confirmed
        if at + delay < stop:
            frames = live[at + delay:stop]
            a = (1 - w) * np.log(frames[:, 1:] + 1e-12) + w * ens
            a = np.exp(a - a.max(1, keepdims=True))
            shown[at + delay:stop, 1:] = (1 - frames[:, :1]) * a / a.sum(1, keepdims=True)
    return shown


def _ended(upd: list[tuple[int, int, int, int]], j: int) -> int:
    """Run frame of the final update of update ``j``'s event (when its end is confirmed)."""
    i = upd[j][0]
    while j + 1 < len(upd) and upd[j + 1][0] == i:
        j += 1
    return upd[j][3]


def frame_scores(shown: np.ndarray, gt: np.ndarray, gt_segments: list[tuple[int, int, int]]) -> dict:
    """Top-1 of the shown label: every frame, action frames only, and at each annotated action's last frame."""
    pred = shown.argmax(1)
    act = gt != 0
    return {"frame_acc": float((pred == gt).mean() * 100),
            "action_frame_acc": float((pred[act] == gt[act]).mean() * 100),
            "end_acc": float(np.mean([pred[e] == label for _, e, label in gt_segments]) * 100)}
