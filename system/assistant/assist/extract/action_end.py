"""The action-end label: one final label per action, fixed when the action ends.

While an action runs, the assistant's running label stays close to the TCN, because the clip models have seen little of the
action yet. Once the action ends they have seen all of it, so they lead here. The action runs once through the models below,
so this label does not have to fit the 0.5 s budget of the running label.

The live assistant and the paper's test (``scripts.recognition.action_end_benchmark``) both call ``end_labels``. The only
difference between them is where the spans come from: the TCN's detected events live, the annotated segments on the paper's test.
It uses each action's own frames and the earlier actions, never anything after the action's end.

The recipe was chosen on the 16 val videos (val top-1 73.81), the best of the candidates that see no frames outside the action.
"""
from __future__ import annotations

import numpy as np

from scripts.recognition.pipeline_on_segments import tcn_on_segment
from scripts.recognition.sequence_recognition import decode_online, log_prior

MEMBERS = ("vmae_full", "vjft_area_k12", "vjft_hand_k12", "vjft_hand_k12_s1")
SAMPLINGS = (0.5, 0.25, 0.75)     # temporal samplings per member, averaged
WEIGHT = 0.95                     # (1 - WEIGHT) log p_tcn + WEIGHT log p_clip
STRENGTH = 0.2                    # the action-order prior over earlier actions
SMOOTHING = 0.01


def mean_clip(probs: dict[str, np.ndarray], members: tuple[str, ...] = MEMBERS) -> np.ndarray:
    """(spans, actions) mean of the members' probabilities; a missing member raises, so the recipe cannot go lighter silently."""
    return sum(probs[m] for m in members) / len(members)


def end_labels(frames: np.ndarray, spans: list[tuple[int, int]], clip: np.ndarray, w: float, strength: float,
               smoothing: float, first: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """1-based action per span of one video, spans in time order and frames inclusive.

    ``frames``: the TCN's (T, 1 + actions) probabilities, background in column 0. ``clip``: (spans, actions).
    """
    tcn = np.stack([tcn_on_segment(frames, s, e) for s, e in spans])
    scores = (1 - w) * np.log(tcn + 1e-8) + w * np.log(clip + 1e-8)
    if strength == 0:
        return scores.argmax(1) + 1
    start, trans = log_prior(first, pairs, smoothing)
    return decode_online(scores, strength * start, strength * trans, "filter") + 1
