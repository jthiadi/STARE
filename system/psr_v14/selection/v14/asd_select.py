"""v14 selection exactly as pre-registered (EXPERIMENT_LOG 'v14 PRE-REGISTRATION'). VAL only."""
import os, sys, glob, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v12.common import H
from analyze_errvids import score
import v11.mistakes_eval as ME
from v14.shadow import alerts
res = {}
for f in glob.glob("v14/agrid_*.pkl"): res.update(pickle.load(open(f, "rb")))
crit = lambda s: (s["all"]["POS"] + s["all"]["F1"] + s["err"]["POS"] + s["err"]["F1"]) / 2 - 0.02 * s["all"]["delay_frames"] / 10
def psr(R, keep=lambda v: True):
    return score([(r["sp"], v, [x for x in r["steps"] if x["id"] % 3 != 1], r["steps"]) for v, r in R.items() if keep(v)])
def mis(R, keep=lambda v: True):
    return ME.score([(v, r["n"], alerts(v, r)) for v, r in R.items() if keep(v)])
def total(o, keep=lambda v: True): return crit(psr(o["live"], keep)) + crit(psr(o["oracle"], keep))
def eligible(o, keep=lambda v: True, ref=None):
    m = mis(o["live"], keep); r = ref or mis(res[CTRL]["live"], keep)
    return m["tp"] >= r["tp"] and m["fa_h"] <= r["fa_h"] + 1e-9
CTRL = json.dumps(dict(W=0.0, CMIN=0.5, EPS=0.05, RATE=1))
print(len(res), "configs (incl. v13 control)")
T = {k: total(o) for k, o in res.items()}
for k in sorted(T, key=T.get, reverse=True)[:10]:
    o = res[k]; print(f"{T[k]:.4f} elig={eligible(o)} {k}\n   live   {H.fmt(psr(o['live']))} | {ME.fmt(mis(o['live']))[:60]}\n   oracle {H.fmt(psr(o['oracle']))} | {ME.fmt(mis(o['oracle']))[:60]}")
print("v13 control", round(T[CTRL], 4))
E = {k: t for k, t in T.items() if k != CTRL and eligible(res[k])}
best = max(E, key=E.get) if E else None
print("\nSELECTED", best, round(E[best], 4) if best else None, "beats v13:", bool(best and E[best] > T[CTRL]))
SUBS = sorted({v[:2] for v in res[CTRL]["live"]}); acc = {"live": [], "oracle": []}; accm = []
for s in SUBS:
    inn = lambda v, s=s: v[:2] != s
    Ei = {k: total(o, inn) for k, o in res.items() if k != CTRL and eligible(o, inn)}
    k = max(Ei, key=Ei.get) if Ei else CTRL
    k = k if Ei and Ei[k] > total(res[CTRL], inn) else CTRL
    print("fold", s, "->", k)
    for m in acc: acc[m] += [(res[k][m][v]["sp"], v, [x for x in res[k][m][v]["steps"] if x["id"] % 3 != 1], res[k][m][v]["steps"]) for v in res[k][m] if v[:2] == s]
    accm += [(v, r["n"], alerts(v, r)) for v, r in res[k]["live"].items() if v[:2] == s]
for m in acc: print("NESTED", m, H.fmt(score(acc[m])))
print("NESTED live mistakes", ME.fmt(ME.score(accm)))
print("v13 control live", H.fmt(psr(res[CTRL]["live"])), "| oracle", H.fmt(psr(res[CTRL]["oracle"])))
