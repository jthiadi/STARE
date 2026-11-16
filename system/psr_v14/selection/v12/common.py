"""v12 dev helpers (train+val only)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v11.h as H
from psr.filter_tracker import FilterTracker, FilterConfig, _lse
class D0Tracker(FilterTracker):
    d0 = 8.0
    def _trigger(self, t, trig, ctx):
        p = 1.0 if self.d0 <= 0 else min(1.0, trig.duration / self.d0)
        if p >= 0.999:
            return super()._trigger(t, trig, ctx)
        old = dict(self.belief); super()._trigger(t, trig, ctx)
        lp, lq = math.log(max(p, 1e-6)), math.log(max(1 - p, 1e-6)); new = {}
        for s in set(old) | set(self.belief):
            xs = [x for x in ((lp + self.belief[s]) if s in self.belief else None, (lq + old[s]) if s in old else None) if x is not None]
            new[s] = _lse(xs)
        self.belief = new; self._norm()

CLS = {0: 0, 1: 1, -1: 2}
class OrderTracker(D0Tracker):
    """Same-frame multi-step reports ordered by visual onset (earliest sustained agreement with the new state, past
    readings only); ties keep canonical order."""
    order_mode = "onset"
    def _decide(self, t):
        before = self.reported
        out = super()._decide(t)
        if len(out) < 2 or self.order_mode == "canon":
            return out
        new = self.reported
        def onset(e):
            k = e["id"] // 3; val = new[k]; run = 0
            for lp in reversed(self.hist):
                c = lp[k][CLS[val]]; o = max(x for j, x in enumerate(lp[k]) if j != CLS[val])
                if c > o: run += 1
                else: break
            return -run
        srt = sorted(out, key=onset)
        n = len(out)
        assert self.y_hat[-n:] == out
        self.y_hat[-n:] = srt
        m = len([e for e in self.y_hat_err if e["frame"] == t])
        tail = self.y_hat_err[-m:]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: onset(e) if e["id"] % 3 != 1 else 0)
        return srt

import collections as _c
from psr.core import load_psr_gt as _gt
_PRE = {}
def precedence(exclude_sub, proc=None):
    """P(a before b) from PSR labels of train+val videos of OTHER subjects (training labels only)."""
    key = (exclude_sub, proc)
    if key not in _PRE:
        cnt = _c.Counter()
        for sp, v in H.E.ITEMS:
            if v[:2] == exclude_sub or (proc and v.split("_")[1] != proc): continue
            first = {}
            for i, e in enumerate(_gt(sp, v)): first.setdefault(e["id"], i)
            for a in first:
                for b in first:
                    if a != b: cnt[a, b] += first[a] < first[b]
        _PRE[key] = cnt
    return _PRE[key]
class PriorOrderTracker(D0Tracker):
    sub = None; amb = 0.0; proc = None
    def _decide(self, t):
        out = super()._decide(t)
        if len(out) < 2: return out
        cnt = precedence(self.sub, self.proc)
        import functools
        def cmp(x, y):
            a, b = x["id"], y["id"]; n = cnt[a, b] + cnt[b, a]
            if n == 0: return 0
            p = cnt[a, b] / n
            return -1 if p > 0.5 + self.amb else (1 if p < 0.5 - self.amb else 0)
        srt = sorted(out, key=functools.cmp_to_key(cmp))
        n = len(out); self.y_hat[-n:] = srt
        m = len([e for e in self.y_hat_err if e["frame"] == t]); tail = self.y_hat_err[-m:]
        ids = [e["id"] for e in srt]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: ids.index(e["id"]) if e["id"] in ids else -1)
        return srt

import itertools as _it
_SEQ = {}
def train_seqs(exclude_sub, proc):
    key = (exclude_sub, proc)
    if key not in _SEQ:
        out = []
        for sp, v in H.E.ITEMS:
            if v[:2] == exclude_sub or v.split("_")[1] != proc: continue
            first = {}
            for i, e in enumerate(_gt(sp, v)): first.setdefault(e["id"], i)
            out.append(first)
        _SEQ[key] = out
    return _SEQ[key]
class CondOrderTracker(D0Tracker):
    """Order of steps reported in the same frame = the permutation most consistent with training-subject label
    sequences that agree with the order of the steps reported so far (labels of OTHER subjects only)."""
    sub = None; proc = None; hist_k = 3
    def _decide(self, t):
        n_prev = len(self.y_hat)
        prev_ids = [e["id"] for e in self.y_hat]
        out = super()._decide(t)
        if len(out) < 2 or len(out) > 5: return out
        seqs = train_seqs(self.sub, self.proc)
        recent = prev_ids[-self.hist_k:] if self.hist_k else []
        def agrees(s):
            r = [i for i in recent if i in s]; g = [i for i in (e["id"] for e in out) if i in s]
            return all(s[a] < s[b] for a, b in zip(r, r[1:])) and all(s[a] < s[b] for a in r for b in g)
        S = [s for s in seqs if agrees(s)] or seqs
        def sc(perm):
            tot = 0
            for s in S:
                p = [i for i in list(recent) + list(perm) if i in s]
                tot += sum(s[a] < s[b] for a, b in _it.combinations(p, 2)) - sum(s[a] > s[b] for a, b in _it.combinations(p, 2))
            return tot
        ids = [e["id"] for e in out]
        best = max(_it.permutations(ids), key=lambda p: (sc(p), p == tuple(ids)))
        by = {e["id"]: e for e in out}
        srt = [by[i] for i in best]
        self.y_hat[n_prev:] = srt
        m = len([e for e in self.y_hat_err if e["frame"] == t]); tail = self.y_hat_err[-m:]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: best.index(e["id"]) if e["id"] in best else -1)
        return srt

class FlushTracker(D0Tracker):
    """At the end of the stream (known at that moment), report the most likely state if its posterior >= th_end and
    (optionally) the changed parts' recent readings agree (mean log-odds >= c_end)."""
    th_end = 0.5; c_end = None
    def finish(self, last_frame):
        best = max(self.belief, key=self.belief.get)
        if best == self.reported or math.exp(self.belief[best]) < self.th_end:
            return []
        if self.c_end is not None:
            changed = [k for k in range(1, 11) if best[k] != self.reported[k]]
            if not changed: return []
            ok, m = self._confirmed(best)
            if m < self.c_end: return []
        steps = self._steps_between(self.reported, best); self.reported = best
        allE = [{"frame": last_frame, "id": s, "description": "", "conf": 1} for s in steps]
        E = [e for e in allE if e["id"] % 3 != 1]
        self.y_hat += E; self.y_hat_err += allE
        return E

class RmOnsetTracker(D0Tracker):
    """Same-frame report: removal steps ordered by how long the part has looked absent (longest first, past readings
    only); other steps keep canonical order and stay in place."""
    margin = 0.0; mode = "rm"
    def _absent_run(self, k):
        run = 0
        for lp in reversed(self.hist):
            if lp[k][0] - lp[k][1] > self.margin: run += 1
            else: break
        return run
    def _decide(self, t):
        n_prev = len(self.y_hat)
        out = super()._decide(t)
        if len(out) < 2: return out
        rm = [e for e in out if e["id"] % 3 == 2]
        if len(rm) < 2 and self.mode == "rm": return out
        runs = {e["id"]: self._absent_run(e["id"] // 3) for e in rm}
        rs = iter(sorted(rm, key=lambda e: -runs[e["id"]]))
        srt = [next(rs) if e["id"] % 3 == 2 else e for e in out]
        self.y_hat[n_prev:] = srt
        ids = [e["id"] for e in srt]
        m = len([e for e in self.y_hat_err if e["frame"] == t]); tail = self.y_hat_err[-m:]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: ids.index(e["id"]) if e["id"] in ids else -1)
        return srt

class MinFixOrderTracker(D0Tracker):
    """Keep the canonical order of a same-frame report unless that relative order (with the last reported step as
    prefix) never occurs in the training-subject label sequences; then use the most frequent observed order."""
    sub = None; proc = None; min_support = 1
    def _decide(self, t):
        n_prev = len(self.y_hat); prev = [e["id"] for e in self.y_hat][-1:]
        out = super()._decide(t)
        if len(out) < 2 or len(out) > 5: return out
        seqs = train_seqs(self.sub, self.proc)
        ids = [e["id"] for e in out]
        def support(perm):
            full = list(prev) + list(perm); c = 0
            for s in seqs:
                p = [i for i in full if i in s]
                if len(p) == len(full) and all(s[a] < s[b] for a, b in zip(p, p[1:])): c += 1
            return c
        if support(ids) >= self.min_support: return out
        best = max(_it.permutations(ids), key=lambda p: (support(p), p == tuple(ids)))
        if support(best) == 0: return out
        by = {e["id"]: e for e in out}; srt = [by[i] for i in best]
        self.y_hat[n_prev:] = srt
        m = len([e for e in self.y_hat_err if e["frame"] == t]); tail = self.y_hat_err[-m:]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: best.index(e["id"]) if e["id"] in best else -1)
        return srt

from psr.official_eval import psr_utils as _pu
def _dl(a, b):
    return _pu.procedure_order_similarity(list(a), list(b))[1] if a else 0.0
class BayesOrderTracker(D0Tracker):
    """Order of a same-frame report = the permutation with the lowest expected POS edit distance against the orders
    of these steps in training-subject label sequences (those containing all of them and agreeing with the last
    reported step); fewer than min_n such sequences -> canonical order.  Ties -> canonical."""
    sub = None; proc = None; min_n = 3
    def _decide(self, t):
        n_prev = len(self.y_hat); prev = [e["id"] for e in self.y_hat][-1:]
        out = super()._decide(t)
        if len(out) < 2 or len(out) > 5: return out
        ids = [e["id"] for e in out]
        S = []
        for s in train_seqs(self.sub, self.proc):
            if not all(i in s for i in ids): continue
            if prev and prev[0] in s and not all(s[prev[0]] < s[i] for i in ids): continue
            S.append(sorted(ids, key=lambda i: s[i]))
        if len(S) < self.min_n: return out
        cost = lambda p: sum(_dl(o, p) for o in S) / len(S)
        best = min(_it.permutations(ids), key=lambda p: (round(cost(p), 9), p != tuple(ids)))
        if list(best) == ids: return out
        by = {e["id"]: e for e in out}; srt = [by[i] for i in best]
        self.y_hat[n_prev:] = srt
        m = len([e for e in self.y_hat_err if e["frame"] == t]); tail = self.y_hat_err[-m:]
        self.y_hat_err[-m:] = sorted(tail, key=lambda e: best.index(e["id"]) if e["id"] in best else -1)
        return srt
