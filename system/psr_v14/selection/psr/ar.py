"""Action-recognition providers.

Every provider answers one question: at frame t, which actions are active?  It may only use
information available at frame t.  Swap ``OracleARProvider`` for a real model by implementing
``ARProvider.observe``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import ARSegment, load_ar_labels


@dataclass
class ARObs:
    name: str
    cls: int
    conf: float


class ARProvider:
    kind = "none"          # "none" | "oracle" | "predicted"
    label = "No AR (visual-only)"

    def reset(self, split: str, video: str, n_frames: int) -> None:
        pass

    def observe(self, t: int) -> list[ARObs]:
        return []


class OracleARProvider(ARProvider):
    """ORACLE (diagnostic only): ground-truth AR labels, revealed causally.

    At frame t it returns exactly the GT segments with start <= t <= end.  It never reveals a
    segment before it starts, nor when it will end; the tracker only learns that a segment ended
    at the first frame after its end.
    """
    kind = "oracle"
    label = "ORACLE AR (ground truth, diagnostic)"

    def reset(self, split, video, n_frames):
        self.segs: list[ARSegment] = sorted(load_ar_labels(split, video), key=lambda s: s.start)

    def observe(self, t):
        return [ARObs(s.name, s.cls, 1.0) for s in self.segs if s.start <= t <= s.end]


class PredictedARProvider(ARProvider):
    """Placeholder for the user's causal AR model.

    Implement ``observe`` to return the model's current top action(s) for frame t, computed only
    from frames <= t.
    """
    kind = "predicted"
    label = "Predicted AR"

    def __init__(self, model=None):
        self.model = model

    def observe(self, t):  # pragma: no cover - to be implemented with the real model
        raise NotImplementedError("Plug the causal AR model in here.")


def make_provider(mode: str) -> ARProvider:
    return {"oracle": OracleARProvider, "none": ARProvider}[mode]()
