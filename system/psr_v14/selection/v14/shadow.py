"""v14 (train+val only): a second ("shadow") state filter used only for mistake alerts.
Reported PSR steps stay those of v13; the shadow filter's own step stream is replayed through the mistake rules.
    python v14/shadow.py probe          # a few hand-picked configs
    python v14/shadow.py grid <shard> <nshards>
Per run we keep the shadow steps and the end-of-recording belief / recent log-odds, so alert rules can vary offline."""
import os, sys, math, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v12.common import D0Tracker, H
from analyze_errvids import models
from psr.filter_tracker import FilterConfig
import v11.mistakes_eval as ME

V13 = dict(H.V9, tau=0.05, theta=0.6); D0_13 = 12.0
MODES = ["oracle", "live", "sim0", "sim1", "sim2"]


class T(D0Tracker):
    def marginal(self, k, value): return sum(math.exp(lp) for s, lp in self.belief.items() if s[k] == value)
    def recent_logodds(self, k, n=20):
        h = self.hist[-n:]; return sum(x[k][1] - x[k][0] for x in h) / max(1, len(h))


def run_mode(fc, d0, mode, M):
    out = {}
    if mode.startswith("sim"):
        import v13.simar as SA; seed = int(mode[3:])
    T.d0 = float(d0)
    for sp, v in H.items("live" if mode == "live" else "oracle"):
        kw = {"obs_fn": (lambda v_, sp=sp: SA.simulate(sp, v_, seed))} if mode.startswith("sim") else {}
        tr = H.run(sp, v, FilterConfig(**fc), M[v[:2]], "oracle" if mode == "oracle" else "live", tracker_cls=T, **kw)
        n = len(H.frame_paths(sp, v))
        end = {k: (tr.marginal(k, 1), tr.marginal(k, 0), tr.recent_logodds(k)) for k in range(1, 11)}
        out[v] = dict(sp=sp, n=n, steps=list(tr.y_hat_err), end=end)
    return out


def alerts(v, r, b_keep=0.5, b_pii=0.9, use_end=True):
    """v13 alert rule on a (shadow) step stream; r = run record"""
    out = []; dur = r["n"] / ME.FPS; task = ME.TASKS[ME.task_of(v)]
    for a in ME.alerts_from_steps(v, r["steps"], r["n"]):
        if use_end and a["type"] == "skipped_step" and a["t"] >= dur - 1:
            st = task.steps[a["step"]]; k = ME.CFG["components"].index(st.component)
            m1, m0, lo = r["end"][k]; done = m1 if st.value == 1 else m0
            seen = lo > 0 if st.value == 1 else lo < 0
            if done >= b_keep or seen:
                if st.value == 1 and done >= b_pii: out.append(dict(a, type="possible_incorrect_install", component=st.component))
                continue
        out.append(a)
    return out


def score_run(R, **kw):
    return ME.score([(v, r["n"], alerts(v, r, **kw)) for v, r in R.items()])


PROBE = {"v13": (V13, D0_13), "d0=20": (V13, 20), "d0=30": (V13, 30), "theta.9": (dict(V13, theta=0.9), 12),
         "conf6": (dict(V13, conf_thr=6.0), 12), "stable6": (dict(V13, stable=6), 12), "tau.08": (dict(V13, tau=0.08), 12),
         "tau.03": (dict(V13, tau=0.03), 12)}
GRID = dict(tau=[0.03, 0.05, 0.08], theta=[0.6, 0.8, 0.9], d0=[12, 20, 30], conf_thr=[5.0, 6.0], stable=[3, 6])


def grid_cfgs():
    for vals in itertools.product(*GRID.values()):
        c = dict(zip(GRID, vals)); d0 = c.pop("d0"); yield json.dumps(dict(c, d0=d0)), (dict(V13, **c), d0)


if __name__ == "__main__":
    M = models()
    if sys.argv[1] == "probe":
        name = sys.argv[2]; fc, d0 = PROBE[name]; res = {}
        for mode in MODES:
            res[mode] = run_mode(fc, d0, mode, M); print(name, mode, ME.fmt(score_run(res[mode])), flush=True)
        pickle.dump(res, open(f"v14/probe_{name}.pkl", "wb"))
    else:
        shard, ns = int(sys.argv[2]), int(sys.argv[3]); res = {}
        for i, (key, (fc, d0)) in enumerate(grid_cfgs()):
            if i % ns != shard: continue
            res[key] = {mode: run_mode(fc, d0, mode, M) for mode in MODES}
            pickle.dump(res, open(f"v14/sgrid_{shard}.pkl", "wb")); print(i, key, flush=True)
        print("SHARD DONE", flush=True)
