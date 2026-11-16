"""v14 = v13 + the official IndustReal assembly state detector (ASD) as extra whole-state evidence in the state filter.

At a frame where the ASD's top box has confidence >= cmin and names an assembly state c (not "background" and not
"error_state"), every belief state s whose installed/absent code differs from c is down-weighted by w * log(eps).
Everything else (verifiers, step models, triggers, reporting, mistake alerts) is v13 (psr/v11.py). Causal: the ASD
output of the current frame only.  Selected on val only (EXPERIMENT_LOG "v14 PRE-REGISTRATION").
"""
from __future__ import annotations

import math

import numpy as np

from .v11 import V11Tracker

# ASD class index -> state (order of the released IndustReal ASD model, asd_best_IndustRealandSynthetic.pt)
ASD_CLASSES = ["background", "10000000000", "10010010000", "10010100000", "10010110000", "11100000000",
               "11110010000", "11110100000", "11110110000", "11110111100", "11110111110", "11110110001",
               "11110111101", "11110111111", "11110101111", "11110011111", "11110011110", "11110101110",
               "11100001110", "11101101110", "11101011110", "11101111110", "11101111111", "error_state"]


def asd_states(rows: np.ndarray, cmin: float) -> dict[int, str]:
    """rows (N, 7): frame, class, conf, x, y, w, h -> frame -> state of the most confident box (conf >= cmin)."""
    best = {}
    for f, cl, c, *_ in rows:
        f = int(f)
        if f not in best or c > best[f][1]:
            best[f] = (ASD_CLASSES[int(cl)], float(c))
    return {f: s for f, (s, c) in best.items() if c >= cmin and s not in ("background", "error_state")}


class V14Tracker(V11Tracker):
    def __init__(self, proc, fcfg, model, tcfg, d0: float = 0.0, asd: dict | None = None, w: float = 1.0,
                 eps: float = 0.05, rate: int = 1, vision=None):
        super().__init__(proc, fcfg, model, tcfg, d0=d0, vision=vision)
        self.asd, self.w, self.eps, self.rate = asd or {}, w, eps, rate

    def observe_asd(self, t: int, state: str | None):
        if state is None or self.w <= 0 or (self.rate == 2 and t % 2 == 0):
            return
        le = self.w * math.log(self.eps)
        for s in list(self.belief):
            if "".join("1" if x != 0 else "0" for x in s) != state:
                self.belief[s] += le
        self._norm()

    def update(self, t, ar_names, feat=None):
        self.observe_asd(t, self.asd.get(t))
        return super().update(t, ar_names, feat)
