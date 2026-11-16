"""Learned per-component verifier in the tracker, tuned on OUT-OF-FOLD probabilities (4 subject folds over
train+val; each video's probabilities come from a verifier that never saw its subject).  Step models are
leave-one-subject-out as before.  Test untouched."""
import itertools
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import build_tables as bt
from exp_trainval import ITEMS, SUBS, samples
from psr.ar import OracleARProvider
from psr.blend import Blend
from psr.core import PROC_INFO, apply_steps, frame_paths, load_psr_gt, procedure_type
from psr.lr_model import LRGroupModel
from psr.official_eval import aggregate, evaluate
from psr.tracker import GroupModel, PSRTracker, TrackerConfig
from psr.vision import CACHE


class VerifierAdapter:
    """Interface used by the tracker: part_probs(cls) and group_evidence(feat, state, groups)."""

    def __init__(self, min_q=0.0):
        self.min_q = min_q

    def part_probs(self, p):
        return {k + 1: float(p[k]) for k in range(10)}

    def group_evidence(self, feat, state, groups):
        if feat is None or feat["q"] < self.min_q:
            return None
        p = self.part_probs(feat["cls"]); out = {}
        for g in groups:
            if g == ():
                out[g] = 0.0; continue
            after = apply_steps(state, g); ev = 0.0
            for k in range(1, 11):
                if after[k] != state[k]:
                    pk = min(max(p[k], 1e-3), 1 - 1e-3)
                    lo = np.log(pk / (1 - pk))
                    ev += lo if after[k] == 1 else -lo
            out[g] = float(ev)
        return out


def load_probs(v):
    z = np.load(CACHE / "verif_oof" / f"{v}.npz")
    return {int(t): {"cls": p.astype(np.float32), "q": float(q)} for t, p, q in zip(z["t"], z["p"], z["q"])}


PROBS = {v: load_probs(v) for _, v in ITEMS}
_obs = {}


def run(sp, v, cfg, model, vis, stats=None):
    n = len(frame_paths(sp, v))
    if (sp, v) not in _obs:
        ar = OracleARProvider(); ar.reset(sp, v, n); _obs[sp, v] = [[o.name for o in ar.observe(t)] for t in range(n)]
    tr = PSRTracker(procedure_type(v), cfg, model=model, vision=vis); P = PROBS[v]
    for t in range(n):
        f = P.get(t)
        if f is not None:
            f = dict(f, quality=f["q"])
        tr.update(t, _obs[sp, v][t], f)
    tr.finish(n - 1)
    if stats is not None:
        for l in tr.log:
            if l.get("confirm"):
                stats[l["confirm"]["status"]] = stats.get(l["confirm"]["status"], 0) + 1
    return evaluate(load_psr_gt(sp, v), tr.y_hat, PROC_INFO)


BASE = dict(w_action=0.6, w_since=0.3, min_prob=0.3, null_bias=0.0, min_wait=8, hold=True, max_hold=60)

if __name__ == "__main__":
    models = {s: Blend(GroupModel(bt.build([it for it in ITEMS if it[1][:2] != s])),
                       LRGroupModel(samples([it for it in ITEMS if it[1][:2] != s]), C=3.0), 4.0) for s in SUBS}
    res = []

    def ev(name, **kw):
        mq = kw.pop("min_q", 0.0); vis = VerifierAdapter(mq); st = {}
        cfg = TrackerConfig(**{**BASE, **kw})
        a = aggregate([run(sp, v, cfg, models[v[:2]], vis, st) for sp, v in ITEMS])
        a["cfg"] = dict(kw, min_q=mq, family=name); a["gate"] = st; res.append(a)
        print(f"{name:8} LOSO(all16) {a['POS']:.3f} {a['F1']:.3f} {float(a['delay_frames']):6.1f} FP{a['FP']:3d} FN{a['FN']:3d} | {st} | {kw} mq={mq}", flush=True)

    ev("v4")                                                       # reference (no vision)
    # A: verifier as evidence inside the decision (averaged over views since the trigger)
    for wv, mq, mh in itertools.product([0.25, 0.5, 1.0, 2.0], [0.0, 0.2], [60, 100]):
        ev("evidence", w_vision=wv, wait_margin=0.0, max_wait=0, vision_window=10**9, min_q=mq, max_hold=mh)
    # B: mandatory confirmation gate (+ optional evidence)
    for thr, rej, pol, mc, wv in itertools.product([0.0, 1.0, 2.0], [-1.0, -2.0, -99.0], ["drop", "emit"], [60, 150], [0.0, 1.0]):
        ev("gate", confirm=True, confirm_mode="parts", confirm_thr=thr, reject_thr=rej, unconfirmed=pol,
           confirm_min_views=3, max_confirm=mc, w_vision=wv, wait_margin=0.0, max_wait=0, min_q=0.0)
    res.sort(key=lambda r: -(r["POS"] + r["F1"]))
    json.dump(res[:30], open("exp_verifier.json", "w"), indent=1, default=float)
    for r in res[:5]:
        print("TOP", round(r["POS"], 3), round(r["F1"], 3), round(float(r["delay_frames"]), 1), r["FP"], r["FN"], r["cfg"])
