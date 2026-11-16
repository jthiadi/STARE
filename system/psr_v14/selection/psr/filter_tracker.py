"""v6: causal state filter (forward algorithm) over assembly states.

  * belief: log-probabilities over 11-bit states (pruned to the most likely ones)
  * transitions: at an AR fastening trigger, mass moves s -> apply(s, g) with the step model's probability
    p(g | clues, s) ("no step" keeps it); between triggers a tiny prior eps per frame for any legal step group
    (recovers steps whose fastening was not seen)
  * emissions: every verifier reading reweights states by tau * sum_k log p(component k as in state)
  * report: when the most likely state differs from the reported state with posterior >= theta for R readings,
    AND the changed components are visually confirmed (mean signed log-odds of the last N readings >= conf_thr),
    the difference is reported as PSR steps at the current frame.
Only frames/AR/readings up to the current frame are used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .core import INITIAL_STATE, PROC_INFO, STEP_DESC, apply_steps, bits, code, group_is_legal
from .events import GENERIC, EventExtractor

PART_OUT = {"pull_pin_short", "pull_pin_middle", "pull_pin_long", "pull_screw_pin", "pull_small_screw_pin",
            "pull_objects", "pull_partial_model", "pull_wheel", "put_pin_short", "put_pin_middle", "put_pin_long",
            "put_screw_pin", "put_short_brace", "put_long_brace", "put_wheel", "put_objects", "put_nut"}


@dataclass
class FilterConfig:
    tau: float = 0.2              # emission temperature (readings are correlated)
    eps: float = 1e-4             # per-frame prior of a step without a fastening trigger
    theta: float = 0.8            # posterior needed to report
    stable: int = 3               # consecutive readings the new state must stay best
    conf_n: int = 6               # visual confirmation: last N readings ...
    conf_thr: float = 1.0         # ... mean signed log-odds of the changed parts
    conf_min: int = 3
    keep: int = 40                # beam size
    w_model: float = 1.0          # sharpness of the step model's transition probabilities
    min_wait: int = 0             # frames after a trigger before its transitions may be reported
    model_null_floor: float = 0.0  # extra "no step" mass at triggers (log)
    drift_expected: bool = True   # drift only along steps expected in this procedure
    p_err: float = 0.0            # prior that an install lands incorrectly (needs the 3-state verifier)
    err_parts: tuple = ()         # components (1-based) where p_err applies; () = all
    conf_thr_err: float | None = None  # visual confirmation threshold when a part becomes incorrect (None = conf_thr)
    eps_aux: float = 0.0          # prior of an expected step at the end of ANY (non-generic) action
    rec_n: int = 0                # recovery: >0 = this many consecutive readings must contradict the reported
    rec_thr: float = 3.0          #   state of one part with |log-odds| >= rec_thr ...
    rec_expected: bool = True     #   ... and the implied step must be expected in this procedure
    q_ref: float = 0.0            # >0: weight each reading by clip(view quality / q_ref, q_min, 1)
    q_min: float = 0.2
    rm_any: float = 0.0           # at a loosen/pull trigger: prior that any one installed part was taken off
    uni: float = 0.0              # mix this share of a uniform distribution over legal step groups into transitions
    rm_wait: int = 0              # removals: report only >= rm_wait frames after the last trigger ...
    rm_partout: bool = False      # ... or once a pull/put-away action has ended since that trigger


CLS = {0: 0, 1: 1, -1: 2}


def _lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class FilterTracker:
    def __init__(self, proc, fcfg: FilterConfig, model, tcfg, vision=None):
        self.proc, self.f, self.model, self.tcfg = proc, fcfg, model, tcfg
        self.vision = vision
        s0 = bits(INITIAL_STATE[proc])
        self.belief = {s0: 0.0}
        self.reported = s0
        self.events = EventExtractor()
        self.since: set = set()
        self.ntrig = 0
        self.last_commit = 0
        self.last_trigger = -10**9
        self.hist = []                 # recent per-component log-odds readings
        self.best_run = (None, 0)
        self.y_hat: list[dict] = []          # official PSR output (correct steps only)
        self.y_hat_err: list[dict] = []      # error-aware output (adds "Incorrectly installed X")
        self.log: list[dict] = []
        self.groups = [g for g in model.nb.groups if g] if hasattr(model, "nb") else [g for g in model.groups if g]
        exp = {p["id"] for p in PROC_INFO if p.get(f"expected_in_{proc}")}
        self.drift_groups = [g for g in self.groups if all(x in exp for x in g)] if fcfg.drift_expected else self.groups
        self.expected = exp
        self.contra = [0] * 11
        self.pending = None            # interface compatibility with the demo server
        self.confirming = None

    @property
    def state(self):
        return self.reported

    # ------------------------------------------------------------------ helpers
    def _norm(self):
        z = _lse(list(self.belief.values()))
        b = {s: v - z for s, v in self.belief.items()}
        keep = sorted(b.items(), key=lambda kv: -kv[1])[: self.f.keep]
        self.belief = {s: v for s, v in keep if v > -25}

    def _trigger(self, t, trig, ctx):
        new = {}
        for s, lp in self.belief.items():
            sc = self.model.score(self.proc, s, trig.name, trig.window, frozenset(self.since), self.tcfg, self.ntrig, ctx=ctx)
            sc = {g: self.f.w_model * v for g, v in sc.items()}
            if self.f.model_null_floor and () in sc:
                sc[()] += self.f.model_null_floor
            z = _lse(list(sc.values()))
            if self.f.uni > 0:
                legal = [g for g in sc if g]
                u = math.log(self.f.uni / max(1, len(legal)))
                sc = {g: _lse([math.log(1 - self.f.uni) + v - z, u]) if g else math.log(1 - self.f.uni) + v - z
                      for g, v in sc.items()}
                z = _lse(list(sc.values()))
            for g, v in sc.items():
                x = lp + v - z
                outs = [(apply_steps(s, g) if g else s, 0.0)]
                inst = [st // 3 for st in g if st % 3 == 0 and s[st // 3] == 0
                        and (not self.f.err_parts or st // 3 in self.f.err_parts)]
                if self.f.p_err > 0 and inst:
                    s_bad = list(apply_steps(s, g))
                    for k in inst:
                        s_bad[k] = -1
                    outs = [(outs[0][0], math.log(1 - self.f.p_err)), (tuple(s_bad), math.log(self.f.p_err))]
                for s2, extra in outs:
                    y = x + extra
                    new[s2] = _lse([new[s2], y]) if s2 in new else y
        if self.f.rm_any > 0 and (trig.name.startswith("loosen") or trig.name.startswith("pull")):
            keep = math.log(1 - self.f.rm_any)
            new = {s: v + keep for s, v in new.items()}
            for s, lp in self.belief.items():
                on = [k for k in range(1, 11) if s[k] != 0]
                for k in on:
                    s2 = list(s); s2[k] = 0; s2 = tuple(s2)
                    y = lp + math.log(self.f.rm_any / len(on))
                    new[s2] = _lse([new[s2], y]) if s2 in new else y
        self.belief = new
        self._norm()

    def _drift(self, eps):
        if eps <= 0:
            return
        le = math.log(eps)
        new = dict(self.belief)
        for s, lp in list(self.belief.items()):
            if lp < -8:
                continue
            legal = [g for g in self.drift_groups if group_is_legal(s, g)]
            if not legal:
                continue
            each = le - math.log(len(legal))
            for g in legal:
                s2 = apply_steps(s, g)
                x = lp + each
                new[s2] = _lse([new[s2], x]) if s2 in new else x
        self.belief = new
        self._norm()

    def _emit(self, lp, w=1.0):
        """lp[k] = (log p absent, log p correct, log p incorrect) for component k."""
        tau = self.f.tau * w
        for s in list(self.belief):
            e = 0.0
            for k in range(1, 11):
                e += lp[k][CLS[s[k]]]
            self.belief[s] += tau * e
        self._norm()

    def _confirmed(self, s_new):
        changed = [k for k in range(1, 11) if s_new[k] != self.reported[k]]
        if not changed:
            return False, 0.0
        recent = self.hist[-self.f.conf_n:]
        if len(recent) < self.f.conf_min:
            return False, 0.0
        vals = []
        for lp in recent:
            tot = 0.0
            for k in changed:
                pc = min(max(math.exp(lp[k][CLS[s_new[k]]]), 1e-3), 1 - 1e-3)
                tot += math.log(pc / (1 - pc))
            vals.append(tot / len(changed))
        m = sum(vals) / len(vals)
        thr = self.f.conf_thr
        if self.f.conf_thr_err is not None and any(s_new[k] == -1 for k in changed):
            thr = self.f.conf_thr_err
        return m >= thr, m

    @staticmethod
    def _steps_between(a, b):
        """official IndustReal conversion (psr_utils.convert_states_to_steps), incl. error steps 3k+1."""
        out = []
        for k in range(len(a)):
            x, y = a[k], b[k]
            if x == y or (x == -1 and y == 0):
                continue
            if y == 1:
                out.append(3 * k)
            elif y == -1:
                out.append(3 * k + 1)
            else:
                out.append(3 * k + 2)
        return out

    # ------------------------------------------------------------------ main
    def update(self, t, ar_names, feat=None):
        self.since |= {n for n in ar_names if n not in GENERIC}
        cur = set(ar_names)
        ended = getattr(self, "_prev", set()) - cur
        self._prev = cur
        if ended & PART_OUT:
            self.part_out_at = t
        if self.f.eps_aux > 0 and (ended - GENERIC):
            self._drift(self.f.eps_aux)
        for trig in self.events.update(t, ar_names):
            self.ntrig += 1
            self.last_trigger = t
            ctx = {"duration": trig.duration, "elapsed": t - self.last_commit, "tsince": trig.tsince}
            self._trigger(t, trig, ctx)
        self._drift(self.f.eps)
        if feat is not None and feat.get("cls") is not None:
            p = feat["cls"]
            lp = {}
            for k in range(10):
                if len(p) == 30:
                    q = [min(max(float(p[3 * k + c]), 1e-4), 1.0) for c in range(3)]
                else:
                    on = min(max(float(p[k]), 1e-3), 1 - 1e-3)
                    q = [1 - on, on, 1e-4]
                lp[k + 1] = tuple(math.log(x) for x in q)
            self.hist.append(lp)
            self.hist = self.hist[-50:]
            w = 1.0
            if self.f.q_ref > 0:
                w = min(1.0, max(self.f.q_min, float(feat.get("q", 0.0)) / self.f.q_ref))
            self._emit(lp, w)
            out = self._decide(t)
            if not out and self.f.rec_n:
                out = self._recover(t, lp)
            return out
        return []

    def _decide(self, t):
        best = max(self.belief, key=self.belief.get)
        post = math.exp(self.belief[best])
        prev, n = self.best_run
        self.best_run = (best, n + 1 if prev == best else 1)
        if best == self.reported or post < self.f.theta or self.best_run[1] < self.f.stable:
            return []
        if t - self.last_trigger < self.f.min_wait:
            return []
        ok, m = self._confirmed(best)
        if not ok:
            return []
        removal = any(best[k] == 0 and self.reported[k] != 0 for k in range(11))
        if removal and (self.f.rm_wait or self.f.rm_partout):
            waited = t - self.last_trigger >= self.f.rm_wait if self.f.rm_wait else False
            out = self.f.rm_partout and getattr(self, "part_out_at", -1) > self.last_trigger
            if not (waited or out):
                return []
        steps = self._steps_between(self.reported, best)
        before = self.reported
        self.reported = best
        self.since = set(); self.ntrig = 0; self.last_commit = t
        all_entries = [{"frame": t, "id": s, "description": STEP_DESC[s], "conf": 1} for s in steps]
        entries = [e for e in all_entries if e["id"] % 3 != 1]
        self.y_hat += entries
        self.y_hat_err += all_entries
        self.log.append({"frame": t, "trigger_frame": self.last_trigger, "trigger": "filter", "window": [],
                         "decision": steps, "steps": [STEP_DESC[s] for s in steps], "views": len(self.hist),
                         "forced": False, "prob": round(post, 3), "state_before": code(before), "state": code(best),
                         "top": [], "confirm": {"status": "confirmed", "views": min(len(self.hist), self.f.conf_n),
                                                "mean_logodds": round(m, 2), "proposed": [STEP_DESC[s] for s in steps],
                                                "proposed_at": t}})
        return entries

    def _recover(self, t, lp):
        """slow visual recovery of a step no fastening trigger produced (all frames up to t only)"""
        for k in range(1, 11):
            on = lp[k][1] - lp[k][0]                   # log p(correct) - log p(absent)
            cur = self.reported[k]
            against = (on >= self.f.rec_thr and cur != 1) or (on <= -self.f.rec_thr and cur == 1)
            self.contra[k] = self.contra[k] + 1 if against else 0
        cands = [k for k in range(1, 11) if self.contra[k] >= self.f.rec_n]
        if not cands:
            return []
        k = max(cands, key=lambda c: self.contra[c])
        self.contra[k] = 0
        new = list(self.reported)
        new[k] = 1 if self.reported[k] != 1 else 0
        new = tuple(new)
        steps = self._steps_between(self.reported, new)
        if not steps or (self.f.rec_expected and not all(s in self.expected for s in steps)):
            return []
        before = self.reported
        self.reported = new
        self.belief = {new: 0.0}                       # the filter restarts from the visually recovered state
        self.since = set(); self.ntrig = 0; self.last_commit = t
        all_entries = [{"frame": t, "id": s, "description": STEP_DESC[s], "conf": 1} for s in steps]
        entries = [e for e in all_entries if e["id"] % 3 != 1]
        self.y_hat += entries
        self.y_hat_err += all_entries
        self.log.append({"frame": t, "trigger_frame": t, "trigger": "visual recovery", "window": [], "decision": steps,
                         "steps": [STEP_DESC[s] for s in steps], "views": self.f.rec_n, "forced": False, "prob": 1.0,
                         "state_before": code(before), "state": code(new), "top": [],
                         "confirm": {"status": "confirmed (recovery)", "views": self.f.rec_n, "mean_logodds": None,
                                     "proposed": [STEP_DESC[s] for s in steps], "proposed_at": t}})
        return entries

    def finish(self, last_frame):
        return []
