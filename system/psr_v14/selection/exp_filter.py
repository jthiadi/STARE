"""v6 state filter: LOSO over train+val subjects, out-of-fold verifier probabilities.  Test untouched."""
import itertools
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from analyze_errvids import models, score
import os
from exp_verifier import ITEMS, PROBS
ENS = [x for x in os.environ.get("VENS", "").split(",") if x]     # e.g. "verif_oof,verif3_oof,verif_base_oof"
QV = {}
if ENS:
    import numpy as np
    from psr.vision import CACHE
    for _sp, _v in ITEMS:
        acc, cnt = {}, {}
        for d in ENS:
            _z = np.load(CACHE / d / f"{_v}.npz")
            for t, p, q in zip(_z["t"], _z["p"], _z["q"]):
                QV[(_v, int(t))] = float(q)
                p = p.astype(np.float32)
                if len(p) == 10:     # binary -> (absent, correct, incorrect)
                    p3 = np.stack([1 - p, p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
                else:
                    p3 = p
                acc[int(t)] = acc.get(int(t), 0) + p3; cnt[int(t)] = cnt.get(int(t), 0) + 1
        PROBS[_v] = {t: {"cls": acc[t] / cnt[t], "q": QV.get((_v, t), 0.0)} for t in acc}
if os.environ.get("VERIF3") == "1":                     # out-of-fold 3-class verifier probabilities
    import numpy as np
    from psr.vision import CACHE
    for _sp, _v in ITEMS:
        _z = np.load(CACHE / "verif3_oof" / f"{_v}.npz")
        PROBS[_v] = {int(t): {"cls": p.astype(np.float32), "q": float(q)} for t, p, q in zip(_z["t"], _z["p"], _z["q"])}
if os.environ.get("VCAL") == "1":
    from calibrate import calibrate
    for _sp, _v in ITEMS:      # make sure every vector is 30-dim
        for _t, _f in PROBS[_v].items():
            _p = np.asarray(_f["cls"], np.float32)
            if len(_p) == 10:
                _f["cls"] = np.stack([1 - _p, _p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
    calibrate(PROBS)
if os.environ.get("PIN_ORACLE") == "1":     # DIAGNOSTIC ONLY: perfect which-pin identity (uses labels)
    from train_verifier import state_series
    for _sp, _v in ITEMS:
        ts = sorted(PROBS[_v]); y, near = state_series(_sp, _v, np.array(ts))
        for t, yy in zip(ts, y):
            if yy[4] != yy[5]:
                p = np.asarray(PROBS[_v][t]["cls"], np.float32).copy()
                if len(p) == 10:
                    on = max(p[4], p[5]); p[4], p[5] = (on, 1 - on) if yy[4] else (1 - on, on)
                else:
                    q = p.reshape(10, 3); on = max(1 - q[4, 0], 1 - q[5, 0])
                    a, b = (4, 5) if yy[4] else (5, 4)
                    q[a] = [1 - on, on, 0.0]; q[b] = [on, 1 - on, 0.0]; p = q.reshape(-1)
                PROBS[_v][t] = dict(PROBS[_v][t], cls=p)
if os.environ.get("VIS_ORACLE"):     # DIAGNOSTIC ONLY: near-perfect verifier from state labels (incorrect = absent)
    from train_verifier import state_series
    _c = float(os.environ["VIS_ORACLE"])
    for _sp, _v in ITEMS:
        ts = sorted(PROBS[_v]); y, _ = state_series(_sp, _v, np.array(ts))
        for t, yy in zip(ts, y):
            on = np.where(yy > 0, _c, 1 - _c).astype(np.float32)
            PROBS[_v][t] = {"cls": np.stack([1 - on, on, np.full(10, 1e-3, np.float32)], 1).reshape(-1), "q": 0.0}
if os.environ.get("PART_ORACLE"):     # DIAGNOSTIC ONLY: near-perfect readings for the listed parts (0-based), others real
    from train_verifier import state_series
    _ks = [int(k) for k in os.environ["PART_ORACLE"].split(",")]
    for _sp, _v in ITEMS:
        ts = sorted(PROBS[_v]); y, _ = state_series(_sp, _v, np.array(ts))
        for t, yy in zip(ts, y):
            p = np.asarray(PROBS[_v][t]["cls"], np.float32).copy()
            if len(p) == 10:
                p = np.stack([1 - p, p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
            q = p.reshape(10, 3)
            for k in _ks:
                on = 0.995 if yy[k] > 0 else 0.005
                q[k] = [1 - on, on, 1e-3]
            PROBS[_v][t] = dict(PROBS[_v][t], cls=q.reshape(-1))
from psr.ar import OracleARProvider
from psr.core import frame_paths, procedure_type
from psr.filter_tracker import FilterConfig, FilterTracker
if os.environ.get('V9') == '1':
    from psr.v9_filter import V9Config as FilterConfig
from psr.tracker import TrackerConfig

TCFG = TrackerConfig(w_action=0.6, w_since=0.3, null_bias=0.0)
_obs = {}


def run(sp, v, fcfg, model):
    n = len(frame_paths(sp, v))
    if (sp, v) not in _obs:
        ar = OracleARProvider(); ar.reset(sp, v, n); _obs[sp, v] = [[o.name for o in ar.observe(t)] for t in range(n)]
    if os.environ.get("ZRULES") in ("1", "2"):   # zero-shot track: procedure rules (1: expected + once, 2: expected)
        from zs.run_z0 import ZFilterTracker
        tr = ZFilterTracker(procedure_type(v), fcfg, model, TCFG)
        tr.once = os.environ["ZRULES"] == "1"
    elif type(fcfg).__name__ == "V9Config":     # v9: put-away check (psr/v9_filter.py)
        from psr.v9_filter import V9FilterTracker
        tr = V9FilterTracker(procedure_type(v), fcfg, model, TCFG)
    elif type(fcfg).__name__ == "V8Config":     # v8: sequential visual confirmation (psr/v8_filter.py)
        from psr.v8_filter import V8FilterTracker
        tr = V8FilterTracker(procedure_type(v), fcfg, model, TCFG)
    elif type(fcfg).__name__ == "FastConfig":   # zero-shot track: low-delay fast path (zs/fast_filter.py)
        from zs.fast_filter import FastFilterTracker
        tr = FastFilterTracker(procedure_type(v), fcfg, model, TCFG)
    else:
        tr = FilterTracker(procedure_type(v), fcfg, model, TCFG)
    P = PROBS[v]
    for t in range(n):
        tr.update(t, _obs[sp, v][t], P.get(t))
    tr.finish(n - 1)
    return tr


if __name__ == "__main__":
    M = models()
    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else dict(tau=[0.1, 0.3], eps=[1e-5, 1e-4], theta=[0.7, 0.9],
                                                                   stable=[3], conf_thr=[0.5, 1.0])
    res = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals)); t0 = time.time()
        fc = FilterConfig(**kw)
        trs = [(sp, v, run(sp, v, fc, M[v[:2]])) for sp, v in ITEMS]
        sc = score([(sp, v, tr.y_hat, tr.y_hat_err) for sp, v, tr in trs])
        sel = (sc["all"]["POS"] + sc["all"]["F1"] + sc["err"]["POS"] + sc["err"]["F1"]) / 2
        res.append({"cfg": kw, "sel": sel, **sc})
        a, e, w = sc["all"], sc["err"], sc["with_errors_labels"]
        print(f"all {a['POS']:.3f}/{a['F1']:.3f}/{a['delay_frames']:5.1f} FP{a['FP']} FN{a['FN']} | err {e['POS']:.3f}/{e['F1']:.3f} | w/err {w['POS']:.3f}/{w['F1']:.3f} | {kw} {time.time() - t0:.0f}s", flush=True)
    res.sort(key=lambda r: -r["sel"])
    json.dump(res[:20], open("exp_filter.json", "w"), indent=1)
    print("reference v5 (LOSO): all 0.787/0.869/71 | err 0.687/0.798")
