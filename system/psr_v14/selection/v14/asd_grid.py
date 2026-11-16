"""v14 pre-registered grid (VAL only): v13 + ASD whole-state evidence.  Run from selection/: python v14/asd_grid.py <shard> <nshards>
Writes v14/agrid_<shard>.pkl: {config_json: {"oracle"|"live": {v: dict(sp, n, steps, end)}}}"""
import os, sys, math, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v14.shadow import T, V13, D0_13, H
from v14.asd_filter import asd_top
from analyze_errvids import models
from psr.filter_tracker import FilterConfig

GRID = dict(W=[0.3, 1.0, 3.0], CMIN=[0.3, 0.5, 0.7], EPS=[0.02, 0.05, 0.1], RATE=[1, 2])


class AT(T):
    W, EPS, RATE = 1.0, 0.05, 1
    _asd = {}
    def update(self, t, ar_names, feat=None):
        st = self._asd.get(t) if (self.RATE == 1 or t % 2 == 1) else None
        if st is not None:
            le = self.W * math.log(self.EPS)
            for s in list(self.belief):
                if "".join("1" if x != 0 else "0" for x in s) != st: self.belief[s] += le
            self._norm()
        return super().update(t, ar_names, feat)


def run_cfg(c, M):
    out = {}
    for mode in ("oracle", "live"):
        out[mode] = {}
        for sp, v in H.items("live"):                       # the 16 val recordings
            cls = type("A", (AT,), dict(W=c["W"], EPS=c["EPS"], RATE=c["RATE"], _asd=asd_top(v, c["CMIN"]), d0=D0_13))
            tr = H.run(sp, v, FilterConfig(**V13), M[v[:2]], mode, tracker_cls=cls)
            n = len(H.frame_paths(sp, v))
            end = {k: (tr.marginal(k, 1), tr.marginal(k, 0), tr.recent_logodds(k)) for k in range(1, 11)}
            out[mode][v] = dict(sp=sp, n=n, steps=list(tr.y_hat_err), end=end)
    return out


if __name__ == "__main__":
    shard, ns = int(sys.argv[1]), int(sys.argv[2]); M = models(); res = {}
    cfgs = [dict(zip(GRID, x)) for x in itertools.product(*GRID.values())]
    if shard == 0: cfgs_extra = [dict(W=0.0, CMIN=0.5, EPS=0.05, RATE=1)]       # v13 control
    else: cfgs_extra = []
    for i, c in enumerate(cfgs_extra + [c for j, c in enumerate(cfgs) if j % ns == shard]):
        res[json.dumps(c)] = run_cfg(c, M); pickle.dump(res, open(f"v14/agrid_{shard}.pkl", "wb")); print(i, c, flush=True)
    print("SHARD DONE", flush=True)
