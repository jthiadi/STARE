"""Simulated live AR for train+val (see EXPERIMENT_LOG v13 pre-registration). Train+val only."""
import sys, os, random, collections, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v11.h as H
from psr.ar import OracleARProvider
from psr.core import frame_paths
CACHE = "v13/oracle_obs.pkl"
def oracle_obs(sp, v):
    n = len(frame_paths(sp, v)); ar = OracleARProvider(); ar.reset(sp, v, n)
    return [[o.name for o in ar.observe(t)] for t in range(n)]
_O = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
def get_oracle(sp, v):
    if v not in _O:
        _O[v] = oracle_obs(sp, v); pickle.dump(_O, open(CACHE, "wb"))
    return _O[v]
def regions(obs):
    """primary-label segment starts -> regions [s_i, s_{i+1}); region 0 = before first action (label None)."""
    starts = [(0, None)]; prev = None
    for t, o in enumerate(obs):
        lab = o[0] if o else None
        if lab is not None and lab != prev: starts.append((t, lab))
        prev = lab
    return [(s, (starts[i + 1][0] if i + 1 < len(starts) else len(obs)), lab) for i, (s, lab) in enumerate(starts)]
_T = None
def templates():
    global _T
    if _T is None:
        _T = collections.defaultdict(list)
        for sp, v in H.items("live"):
            live = H.live_obs(v)
            for s, e, lab in regions(get_oracle(sp, v)):
                if e > s: _T[lab].append((v[:2], live[s:e]))
    return _T
def simulate(sp, v, seed):
    T = templates(); rng = random.Random(f"{v}-{seed}"); obs = get_oracle(sp, v); out = []
    for s, e, lab in regions(obs):
        L = e - s
        if L <= 0: continue
        cand = [x for sub, x in T.get(lab, []) if sub != v[:2]] or [x for k in T for sub, x in T[k] if sub != v[:2] and k is not None]
        tpl = rng.choice(cand)
        out += [tpl[min(len(tpl) - 1, int(i * len(tpl) / L))] for i in range(L)]
    assert len(out) == len(obs)
    return out
if __name__ == "__main__":
    for sp, v in H.E.ITEMS: get_oracle(sp, v)
    print("cached", len(_O))
