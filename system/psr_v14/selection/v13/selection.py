"""v13 selection exactly as pre-registered: crit(oracle 52) + crit(real live val 16) + mean_seeds crit(sim live 52).
Full selection + nested 4-fold subject CV. Train+val only."""
import sys, os, glob, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v11.h as H
from analyze_errvids import score
res = {}
for f in glob.glob("v13/grid_*.pkl"): res.update(pickle.load(open(f, "rb")))
crit = lambda s: (s["all"]["POS"] + s["all"]["F1"] + s["err"]["POS"] + s["err"]["F1"]) / 2 - 0.02 * s["all"]["delay_frames"] / 10
SUBS = sorted({v[:2] for _, v in H.E.ITEMS}); FOLDS = [SUBS[i::4] for i in range(4)]
def sc(out, mode, keep=lambda v: True):
    rs = [(sp, v, y, ye) for (m, v), (sp, y, ye) in out.items() if m == mode and keep(v)]
    return score(rs) if rs else None
def total(out, keep=lambda v: True):
    l = sc(out, "live", keep)
    return crit(sc(out, "oracle", keep)) + (crit(l) if l else 0) + sum(crit(sc(out, f"sim{s}", keep)) for s in range(3)) / 3
print(len(res), "configs")
T = {k: total(o) for k, o in res.items()}
for k in sorted(T, key=T.get, reverse=True)[:8]:
    o = res[k]; print(f"{T[k]:.4f} {k}\n   oracle {H.fmt(sc(o,'oracle'))}\n   live   {H.fmt(sc(o,'live'))}\n   sim0   {H.fmt(sc(o,'sim0'))}")
best = max(T, key=T.get); print("\nSELECTED", best)
v11k = json.dumps({"tau": 0.05, "theta": 0.7, "d0": 8}); print("v11 total", round(T[v11k], 4), " rank", sorted(T, key=T.get, reverse=True).index(v11k) + 1)
acc = {m: [] for m in ("oracle", "live", "sim0", "sim1", "sim2")}
for f in FOLDS:
    inner = {k: total(o, lambda v: v[:2] not in f) for k, o in res.items()}; k = max(inner, key=inner.get); print("fold", f, "->", k)
    for m in acc: acc[m] += [(sp, v, y, ye) for (mm, v), (sp, y, ye) in res[k].items() if mm == m and v[:2] in f]
for m in acc: print("NESTED", m, H.fmt(score(acc[m])))
