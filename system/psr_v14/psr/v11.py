"""v11 additions on top of the frozen v9 state filter (psr/filter_tracker.py is unchanged).

1. Short-trigger weighting (live AR).  A fastening trigger moves the belief with probability
   p = min(1, d / d0), where d is the length in frames of the action segment that just ended (known at its end);
   with probability 1 - p the state stays as it was.  Live AR produces many 1-4 frame flickers of fastening
   actions that are not real actions; this keeps them from moving the filter.  d0 = 0 is exactly v9.

2. Mistake alerts (no ASD).  The reported steps are replayed into a state sequence and checked with the
   teammate's procedure rules (task graph: wrong order / skipped step / incorrect install; their
   ``assist.evaluation.eval.gt_mistakes``, as in their ``baseline_psr``).  At the end of the recording, a
   "skipped step" alert is kept only when the filter does not believe the step done (posterior < 0.5) and the last
   20 verifier readings do not show it (mean log-odds <= 0).  An install step the filter believes done (>= 0.9) but
   never confirmed visually is reported as "possible_incorrect_install" instead.

Everything is causal: a trigger's weight uses only its own (finished) segment; alerts use only the steps
reported so far, and the end-of-recording check uses the state at the end.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from .filter_tracker import FilterTracker, _lse


class V11Tracker(FilterTracker):
    """v9 filter + short-trigger weighting (d0) and the state needed for the end-of-recording mistake check."""

    def __init__(self, proc, fcfg, model, tcfg, d0: float = 0.0, vision=None):
        super().__init__(proc, fcfg, model, tcfg, vision)
        self.d0 = d0

    def _trigger(self, t, trig, ctx):
        p = 1.0 if self.d0 <= 0 else min(1.0, trig.duration / self.d0)
        if p >= 0.999:
            return super()._trigger(t, trig, ctx)
        old = dict(self.belief)
        super()._trigger(t, trig, ctx)
        lp, lq = math.log(max(p, 1e-6)), math.log(max(1 - p, 1e-6))
        new = {}
        for s in set(old) | set(self.belief):
            xs = [x for x in ((lp + self.belief[s]) if s in self.belief else None,
                              (lq + old[s]) if s in old else None) if x is not None]
            new[s] = _lse(xs)
        self.belief = new
        self._norm()

    def marginal(self, k: int, value: int) -> float:
        return sum(math.exp(lp) for s, lp in self.belief.items() if s[k] == value)

    def recent_logodds(self, k: int, n: int = 20) -> float:
        h = self.hist[-n:]
        return sum(x[k][1] - x[k][0] for x in h) / max(1, len(h))


# ---------------------------------------------------------------- mistake alerts
TEAMMATE = Path(os.environ.get("TEAMMATE_AR", Path(__file__).resolve().parents[2] / "teammate_ar"))


def _assist():
    if str(TEAMMATE) not in sys.path:
        sys.path.insert(0, str(TEAMMATE))
    from assist.common import load_config
    from assist.engine.graph import load_tasks
    from assist.evaluation.baseline import ALERT_TYPE, state_changes
    from assist.evaluation.eval import gt_mistakes
    cfg = load_config()
    return cfg, load_tasks(cfg["graph_path"]), gt_mistakes, state_changes, ALERT_TYPE


_A = None
B_DONE = 0.9      # "believed done" for the possible-incorrect-install conversion


def mistake_alerts(video: str, tr: V11Tracker, n_frames: int, end_check: bool = True) -> list[dict]:
    """Alerts from the reported steps (tr.y_hat_err) + the end-of-recording check.  Uses no labels: gt_mistakes is
    called with ``changes`` (the predicted state sequence), so it never opens a label file."""
    global _A
    if _A is None:
        _A = _assist()
    cfg, tasks, gt_mistakes, state_changes, ALERT_TYPE = _A
    task = tasks[video.split("_")[1]]
    fps = cfg["fps"]; dur = n_frames / fps
    ch = state_changes(sorted(tr.y_hat_err, key=lambda s: s["frame"]), task.initial_state)
    ms = gt_mistakes(video, task, cfg["components"], Path("/nonexistent"), fps, dur, cfg["engine"]["order_grace_s"],
                     changes=ch)
    out = []
    for m in ms:
        a = {"t": m["t"], "type": ALERT_TYPE[m["type"]], "kind": "mistake", "step": m["step"], "component": m["component"]}
        if end_check and a["type"] == "skipped_step" and a["t"] >= dur - 1:
            st = task.steps[a["step"]]
            k = cfg["components"].index(st.component)
            done = tr.marginal(k, st.value)
            lo = tr.recent_logodds(k)
            seen = lo > 0 if st.value == 1 else lo < 0
            if done >= 0.5 or seen:
                if st.value == 1 and done >= B_DONE:
                    out.append(dict(a, type="possible_incorrect_install", component=st.component))
                continue
        out.append(a)
    return sorted(out, key=lambda a: a["t"])


def score_mistakes(per_video: list) -> dict:
    """EVALUATION ONLY (reads PSR_labels_raw.csv).  per_video: [(split, video, n_frames, alerts)].
    Same ground-truth rules, collapsing and matching as the teammate's assist/evaluation/eval.py
    (match window 30 s, order grace 10 s)."""
    global _A
    if _A is None:
        _A = _assist()
    cfg, tasks, gt_mistakes, _, _ = _A
    from assist.evaluation.eval import TYPE_OF, TYPES, collapse, match
    from .core import video_dir
    win, grace, fps = cfg["evaluation"]["match_window_s"], cfg["engine"]["order_grace_s"], cfg["fps"]
    st = {k: {"gt": 0, "alerts": 0, "tp": 0, "d": []} for k in TYPES}
    hours = 0.0
    for split, v, n, al in per_video:
        hours += n / fps / 3600
        g = gt_mistakes(v, tasks[v.split("_")[1]], cfg["components"], video_dir(split, v).parent, fps, n / fps, grace)
        m = collapse(al, win)
        for k in TYPES:
            st[k]["gt"] += sum(1 for x in g if x["type"] == k)
            st[k]["alerts"] += sum(1 for x in m if TYPE_OF[x["type"]] == k)
        for a, b in match(g, m, win):
            st[a["type"]]["tp"] += 1
            st[a["type"]]["d"].append(b["t"] - a["t"])
    out = {}
    for k in list(TYPES) + ["all"]:
        ks = TYPES if k == "all" else [k]
        gt_, al_, tp_ = (sum(st[x][f] for x in ks) for f in ("gt", "alerts", "tp"))
        d = [y for x in ks for y in st[x]["d"]]
        out[k] = {"gt": gt_, "alerts": al_, "tp": tp_, "precision": round(tp_ / al_, 3) if al_ else None,
                  "recall": round(tp_ / gt_, 3) if gt_ else None, "false_alarms_per_h": round((al_ - tp_) / hours, 2),
                  "mean_delay_s": round(sum(d) / len(d), 1) if d else None}
    out["hours"] = round(hours, 3)
    return out
