"""v14 re-tuned on validation for one shifted stream (PREREGISTRATION.md, "Re-tuning"). Test is never read.
    python metrics/e2e/retune.py grid <tcn|ens|oracle> <shard> <nshards>    # live half of the pre-registered grid, odd-frame ASD
    python metrics/e2e/retune.py select <tcn|ens>                     # asd_select.py's rule -> work/models/<name>/
The oracle half of the criterion does not depend on the AR input: it is read from the original runs
(system/psr_v14/selection/v14/agrid_*.pkl). The v13 control is rerun live like every other config.
"oracle" (PREREGISTRATION_ablations.md): the live half uses the annotated val actions shifted by v14's p95 compute."""
import glob
import itertools
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("E2E_WORK", HERE / "work"))           # outputs (reproduce_all.py points these elsewhere)
RES = Path(os.environ.get("E2E_RESULTS", HERE / "results"))
SYS = HERE.parents[1] / "system"
SEL = SYS / "psr_v14" / "selection"
GRID = dict(W=[0.3, 1.0, 3.0], CMIN=[0.3, 0.5, 0.7], EPS=[0.02, 0.05, 0.1], RATE=[2])
CTRL = dict(W=0.0, CMIN=0.5, EPS=0.05, RATE=1)


def setup(name: str) -> None:
    os.environ["LIVE_AR_DIR"] = "" if name == "oracle" else str(WORK / "streams" / f"{name}_val")
    os.environ.setdefault("INDUSTREAL_DATA", str(SYS / "dataset"))
    os.environ.setdefault("TEAMMATE_AR", str(SYS / "assistant"))
    os.chdir(SEL)                                   # the selection code reads ../cache relative to here
    sys.path[:0] = [str(SEL), str(SYS / "v14_demo")]
    import frames_compat  # noqa: F401  frame counts from the .mp4 when the dataset has no JPEG frames (as reproduce_all.py)


def grid(name: str, shard: int, ns: int) -> None:
    setup(name)
    from v14.asd_grid import AT, H, V13, D0_13
    from v14.asd_filter import asd_top
    from analyze_errvids import models
    from psr.filter_tracker import FilterConfig
    M = models()
    obs_fn = None
    if name == "oracle":
        sys.path.insert(0, str(HERE))
        from oracle_shift import d_oracle, shifted_oracle
        from psr.ar import OracleARProvider
        D, cache = d_oracle(), {}
        obs_fn = lambda v: cache.setdefault(v, shifted_oracle(OracleARProvider, "val", v, len(H.frame_paths("val", v)), D))
    cfgs = [CTRL] + [dict(zip(GRID, x)) for x in itertools.product(*GRID.values())]
    out = WORK / "retune" / name / f"live_{shard}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    for j, c in enumerate(cfgs):
        if j % ns != shard:
            continue
        runs = {}
        for sp, v in H.items("live"):              # the 16 val recordings
            cls = type("A", (AT,), dict(W=c["W"], EPS=c["EPS"], RATE=c["RATE"], _asd=asd_top(v, c["CMIN"]), d0=D0_13))
            tr = H.run(sp, v, FilterConfig(**V13), M[v[:2]], "live", tracker_cls=cls, obs_fn=obs_fn)
            n = len(H.frame_paths(sp, v))
            end = {k: (tr.marginal(k, 1), tr.marginal(k, 0), tr.recent_logodds(k)) for k in range(1, 11)}
            runs[v] = dict(sp=sp, n=n, steps=list(tr.y_hat_err), end=end)
        res[json.dumps(c)] = {"live": runs}
        pickle.dump(res, open(out, "wb"))
        print(name, j, c, flush=True)
    print("SHARD DONE", name, shard, flush=True)


def select(name: str) -> None:
    setup(name)
    from v12.common import H
    from analyze_errvids import score
    import v11.mistakes_eval as ME
    from v14.shadow import alerts
    orig = {}
    for f in glob.glob(str(SEL / "v14/agrid_*.pkl")):
        orig.update(pickle.load(open(f, "rb")))
    res = {}
    for f in glob.glob(str(WORK / "retune" / name / "live_*.pkl")):
        for k, o in pickle.load(open(f, "rb")).items():
            res[k] = {"live": o["live"], "oracle": orig[k]["oracle"]}
    assert len(res) == 28, f"{len(res)} configs (expected 27 + control)"
    crit = lambda s: (s["all"]["POS"] + s["all"]["F1"] + s["err"]["POS"] + s["err"]["F1"]) / 2 - 0.02 * s["all"]["delay_frames"] / 10
    psr = lambda R: score([(r["sp"], v, [x for x in r["steps"] if x["id"] % 3 != 1], r["steps"]) for v, r in R.items()])
    mis = lambda R: ME.score([(v, r["n"], alerts(v, r)) for v, r in R.items()])
    total = lambda o: crit(psr(o["live"])) + crit(psr(o["oracle"]))
    ck = json.dumps(CTRL)
    ref = mis(res[ck]["live"])
    elig = lambda o: (lambda m: m["tp"] >= ref["tp"] and m["fa_h"] <= ref["fa_h"] + 1e-9)(mis(o["live"]))
    T = {k: total(o) for k, o in res.items()}
    E = {k: t for k, t in T.items() if k != ck and elig(res[k])}
    best = max(E, key=E.get) if E else None
    chosen = best if best and E[best] > T[ck] else ck                # no eligible config beats v13 -> v13 (ASD off)
    rows = [{"config": json.loads(k), "total": round(T[k], 4), "eligible": k in E, "live": H.fmt(psr(res[k]["live"]))}
            for k in sorted(T, key=T.get, reverse=True)]
    c = json.loads(chosen)
    rep = {"stream": name, "chosen": c, "chosen_total": round(T[chosen], 4), "control_total": round(T[ck], 4), "ranking": rows}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"retune_{name}.json").write_text(json.dumps(rep, indent=1))
    m = WORK / "models" / name
    shutil.rmtree(m, ignore_errors=True)
    shutil.copytree(SYS / "psr_v14/model", m)
    cfg = json.loads((m / "config.json").read_text())
    cfg["asd"].update(w=c["W"], cmin=c["CMIN"], eps=c["EPS"], rate=c["RATE"])
    cfg["selection"] = f"e2e re-tune on val with the {name} stream (metrics/e2e/PREREGISTRATION.md): {c}"
    (m / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: rep[k] for k in ("stream", "chosen", "chosen_total", "control_total")}), flush=True)
    for r in rows[:5]:
        print(r, flush=True)


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    assert name in ("tcn", "ens", "oracle")
    grid(name, int(sys.argv[3]), int(sys.argv[4])) if cmd == "grid" else select(name)
