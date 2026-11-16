"""v13 pre-registered grid (train+val only). Saves per-video outputs. python v13/grid.py <shard> <nshards>"""
import sys, os, itertools, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v12.common import D0Tracker, H
from analyze_errvids import models
from psr.filter_tracker import FilterConfig
import v13.simar as SA
GRID = dict(tau=[0.04, 0.05, 0.06], theta=[0.6, 0.7, 0.8, 0.9], d0=[0, 4, 8, 12])
def cfgs():
    for vals in itertools.product(*GRID.values()): yield dict(zip(GRID, vals))
class T(D0Tracker): pass
def run_cfg(c, M):
    fc = FilterConfig(**dict(H.V9, tau=c["tau"], theta=c["theta"])); T.d0 = float(c["d0"]); out = {}
    for sp, v in H.items("oracle"):
        tr = H.run(sp, v, fc, M[v[:2]], "oracle", tracker_cls=T); out[("oracle", v)] = (sp, tr.y_hat, tr.y_hat_err)
    for sp, v in H.items("live"):
        tr = H.run(sp, v, fc, M[v[:2]], "live", tracker_cls=T); out[("live", v)] = (sp, tr.y_hat, tr.y_hat_err)
    for seed in range(3):
        for sp, v in H.items("oracle"):
            tr = H.run(sp, v, fc, M[v[:2]], "live", tracker_cls=T, obs_fn=lambda v_, sp=sp, s=seed: SA.simulate(sp, v_, s))
            out[(f"sim{seed}", v)] = (sp, tr.y_hat, tr.y_hat_err)
    return out
if __name__ == "__main__":
    shard, ns = int(sys.argv[1]), int(sys.argv[2]); M = models(); res = {}
    for i, c in enumerate(cfgs()):
        if i % ns != shard: continue
        res[json.dumps(c)] = run_cfg(c, M); print(i, c, flush=True)
        pickle.dump(res, open(f"v13/grid_{shard}.pkl", "wb"))
    print("SHARD DONE", flush=True)
