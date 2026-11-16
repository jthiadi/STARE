"""v14 (VAL only; ASD was trained on train): v13 filter + ASD as extra whole-state evidence.
At each frame where ASD's top box (conf >= CMIN) names a state c (not background / error_state), every belief state s
gets + W * (0 if code(s) == c else log EPS). Causal: the ASD output of frame t only.
    python v14/asd_filter.py <W> <CMIN> [EPS]"""
import os, sys, math, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import v14.shadow as S
from v14.shadow import T, V13, D0_13, run_mode, score_run, H
from analyze_errvids import models, score
import v11.mistakes_eval as ME

CATS = ME.CFG["asd_categories"]
_A = {}


def asd_top(v, cmin):
    if v not in _A:
        if not os.path.exists(f"../cache/asd_val/{v}.npz"): return {}      # train recordings: not scored here
        z = np.load(f"../cache/asd_val/{v}.npz"); best = {}
        for f, cl, c, *_ in z["rows"]:
            f = int(f)
            if f not in best or c > best[f][1]: best[f] = (CATS[int(cl)], float(c))
        _A[v] = best
    return {f: s for f, (s, c) in _A[v].items() if c >= cmin and s not in ("background", "error_state")}


class AT(T):
    W, CMIN, EPS = 1.0, 0.5, 0.05
    _asd = {}
    def update(self, t, ar_names, feat=None):
        st = self._asd.get(t)
        if st is not None:
            le = math.log(self.EPS)
            for s in list(self.belief):
                c = "".join("1" if x != 0 else "0" for x in s)
                if c != st: self.belief[s] += self.W * le
            self._norm()
        return super().update(t, ar_names, feat)


if __name__ == "__main__":
    W, CMIN = float(sys.argv[1]), float(sys.argv[2]); EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    AT.W, AT.CMIN, AT.EPS = W, CMIN, EPS
    orig_run = H.run
    def run(sp, v, fcfg, model, armode="oracle", tracker_cls=None, obs_fn=None):
        cls = type("ATv", (tracker_cls,), {"_asd": asd_top(v, CMIN)}) if W > 0 else tracker_cls
        return orig_run(sp, v, fcfg, model, armode, tracker_cls=cls, obs_fn=obs_fn)
    H.run = run; S.T = AT
    crit = lambda s: (s["all"]["POS"] + s["all"]["F1"] + s["err"]["POS"] + s["err"]["F1"]) / 2 - 0.02 * s["all"]["delay_frames"] / 10
    M = models()
    for mode in ("oracle", "live"):
        R = run_mode(V13, D0_13, mode, M)
        R = {v: r for v, r in R.items() if r["sp"] == "val"}
        s = score([(r["sp"], v, [x for x in r["steps"] if x["id"] % 3 != 1], r["steps"]) for v, r in R.items()])
        print(f"W={W} CMIN={CMIN} EPS={EPS} val {mode:6s} {H.fmt(s)} crit {crit(s):.4f} | {ME.fmt(score_run(R))[:95]}", flush=True)
