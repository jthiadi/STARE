"""Ablation: STARE without action recognition (PREREGISTRATION_noar.md). Vision only: no action input, the tracker's
drift prior eps selected on validation (v14 selection harness, out-of-fold verifiers), test scored once.
    python metrics/e2e/noar.py grid <shard> <nshards>   # validation grid -> work/noar/grid_<shard>.pkl
    python metrics/e2e/noar.py select                   # -> results/retune_noar.json, work/models/noar
    python metrics/e2e/noar.py test                     # one test score -> results/test_noar, results/e2e_summary.json"""
import glob
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("E2E_WORK", HERE / "work"))
RES = Path(os.environ.get("E2E_RESULTS", HERE / "results"))
SYS = HERE.parents[1] / "system"
SEL = SYS / "psr_v14" / "selection"
GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ASD = dict(W=1.0, CMIN=0.3, EPS=0.02, RATE=2)          # v14's frozen state-detector setting


def setup() -> None:
    os.environ["LIVE_AR_DIR"] = ""
    os.environ.setdefault("INDUSTREAL_DATA", str(SYS / "dataset"))
    os.environ.setdefault("TEAMMATE_AR", str(SYS / "assistant"))
    os.chdir(SEL)
    sys.path[:0] = [str(SEL), str(SYS / "v14_demo")]
    import frames_compat  # noqa: F401


def grid(shard: int, ns: int) -> None:
    setup()
    from v14.asd_grid import AT, H, V13, D0_13
    from v14.asd_filter import asd_top
    from analyze_errvids import models
    from psr.filter_tracker import FilterConfig
    M = models()
    out = WORK / "noar" / f"grid_{shard}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    for j, eps in enumerate(GRID):
        if j % ns != shard:
            continue
        runs = {}
        for sp, v in H.items("live"):                     # the 16 validation recordings
            n = len(H.frame_paths(sp, v))
            cls = type("A", (AT,), dict(W=ASD["W"], EPS=ASD["EPS"], RATE=ASD["RATE"], _asd=asd_top(v, ASD["CMIN"]), d0=D0_13))
            tr = H.run(sp, v, FilterConfig(**{**V13, "eps": eps}), M[v[:2]], "live", tracker_cls=cls,
                       obs_fn=lambda _v, n=n: [[] for _ in range(n)])
            runs[v] = dict(sp=sp, n=n, steps=list(tr.y_hat_err))
        res[eps] = runs
        pickle.dump(res, open(out, "wb"))
        print("eps", eps, flush=True)
    print("SHARD DONE", shard, flush=True)


def select() -> None:
    setup()
    from v12.common import H
    from analyze_errvids import score
    res = {}
    for f in glob.glob(str(WORK / "noar" / "grid_*.pkl")):
        res.update(pickle.load(open(f, "rb")))
    assert len(res) == len(GRID), f"{len(res)} configs"
    crit = lambda s: (s["all"]["POS"] + s["all"]["F1"] + s["err"]["POS"] + s["err"]["F1"]) / 2 - 0.02 * s["all"]["delay_frames"] / 10
    psr = lambda R: score([(r["sp"], v, [x for x in r["steps"] if x["id"] % 3 != 1], r["steps"]) for v, r in R.items()])
    T = {e: crit(psr(r)) for e, r in res.items()}
    best = max(T, key=T.get)
    rep = {"ablation": "no action recognition (vision only)", "chosen_eps": best,
           "ranking": [{"eps": e, "criterion": round(T[e], 4), "val": H.fmt(psr(res[e]))} for e in sorted(T, key=T.get, reverse=True)]}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "retune_noar.json").write_text(json.dumps(rep, indent=1))
    m = WORK / "models" / "noar"
    shutil.rmtree(m, ignore_errors=True)
    shutil.copytree(SYS / "psr_v14/model", m)
    cfg = json.loads((m / "config.json").read_text())
    cfg["filter"]["eps"] = best
    cfg["selection"] = f"no-AR ablation (metrics/e2e/PREREGISTRATION_noar.md): v14 frozen + drift prior eps {best}, chosen on val"
    (m / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


def test() -> None:
    sys.path.insert(0, str(HERE))
    import test_once as T
    summ = RES / "e2e_summary.json"
    old = json.loads(summ.read_text()) if summ.exists() else {}
    assert "noar" not in old, "noar was already scored on test (one look only)"
    sys.path.insert(0, str(SYS / "assistant"))
    from assist.industreal import SPLITS
    from streams import read, write
    for rid in SPLITS["test"]:                            # all-background action stream (no action input)
        n = len(read(SYS / "assistant/output/assist" / rid / "ar.jsonl"))
        write(WORK / "streams" / "noar_test" / rid / "ar.jsonl", [["background", 1.0]] * n)
    r = {"chosen": {"eps": json.loads((RES / "retune_noar.json").read_text())["chosen_eps"]}, **T.evaluate("noar")}
    print("noar", json.dumps(r), flush=True)
    summ.write_text(json.dumps({**old, "noar": r}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    grid(int(sys.argv[2]), int(sys.argv[3])) if cmd == "grid" else select() if cmd == "select" else test()
