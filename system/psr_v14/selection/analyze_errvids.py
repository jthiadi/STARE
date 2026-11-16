"""v5 config, LOSO over train+val with out-of-fold verifier probabilities.
Reports: all / error / no-error recordings on PSR_labels.csv, and all recordings on PSR_labels_with_errors.csv,
plus what goes wrong in error recordings.  Test untouched."""
import collections
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import build_tables as bt
from exp_verifier import BASE, ITEMS, PROBS, SUBS, VerifierAdapter, samples
from psr.ar import OracleARProvider
from psr.blend import Blend
from psr.core import PROC_INFO, STEP_DESC, frame_paths, load_psr_gt, procedure_type, video_dir
from psr.lr_model import LRGroupModel
from psr.official_eval import aggregate, evaluate, psr_utils
from psr.tracker import GroupModel, PSRTracker, TrackerConfig

V5 = dict(BASE, w_vision=2.0, wait_margin=0.0, max_wait=0, confirm=True, confirm_mode="parts", confirm_thr=1.0,
          reject_thr=-99.0, unconfirmed="drop", confirm_min_views=3, max_confirm=150, confirm_recent=6)


def models():
    return {s: Blend(GroupModel(bt.build([it for it in ITEMS if it[1][:2] != s])),
                     LRGroupModel(samples([it for it in ITEMS if it[1][:2] != s]), C=3.0), 4.0) for s in SUBS}


_obs = {}


def run_video(sp, v, cfg, model, vis=None):
    n = len(frame_paths(sp, v))
    if (sp, v) not in _obs:
        ar = OracleARProvider(); ar.reset(sp, v, n); _obs[sp, v] = [[o.name for o in ar.observe(t)] for t in range(n)]
    tr = PSRTracker(procedure_type(v), cfg, model=model, vision=vis or VerifierAdapter(0.0))
    P = PROBS[v]
    for t in range(n):
        f = P.get(t)
        tr.update(t, _obs[sp, v][t], dict(f, quality=f["q"]) if f else None)
    tr.finish(n - 1)
    return tr


def score(results):
    """results: list of (sp, v, y_hat). -> dict of aggregates."""
    out = {}
    per_all, per_err, per_ok, per_we = [], [], [], []
    for r in results:
        sp, v, y = r[:3]
        ye = r[3] if len(r) > 3 else y
        gt = load_psr_gt(sp, v)
        m = evaluate(gt, y, PROC_INFO)
        per_all.append(m)
        (per_err if psr_utils.video_contains_errors(gt, PROC_INFO, video_dir(sp, v)) else per_ok).append(m)
        per_we.append(evaluate(load_psr_gt(sp, v, with_errors=True), ye, PROC_INFO))
    f = lambda a: {k: round(float(a[k]), 3) for k in ("POS", "F1", "delay_frames")} | {k: int(a[k]) for k in ("TP", "FP", "FN", "n_videos")}
    out["all"], out["err"], out["ok"], out["with_errors_labels"] = f(aggregate(per_all)), f(aggregate(per_err)), f(aggregate(per_ok)), f(aggregate(per_we))
    return out


if __name__ == "__main__":
    M = models()
    cfg = TrackerConfig(**V5)
    res = []
    C = collections.Counter(); ex = []
    for sp, v in ITEMS:
        tr = run_video(sp, v, cfg, M[v[:2]])
        res.append((sp, v, tr.y_hat))
        gt = load_psr_gt(sp, v)
        if not psr_utils.video_contains_errors(gt, PROC_INFO, video_dir(sp, v)):
            continue
        gid = collections.defaultdict(list); pid = collections.defaultdict(list)
        for e in gt: gid[e["id"]].append(e["frame"])
        for e in tr.y_hat: pid[e["id"]].append(e["frame"])
        for i in set(gid) | set(pid):
            g, p = sorted(gid[i]), sorted(pid[i])
            if not g: C["FP: step not in GT"] += len(p); ex.append((v, "FP-notGT", STEP_DESC[i], p)); continue
            if not p: C["FN: never predicted"] += len(g); ex.append((v, "FN-never", STEP_DESC[i], g)); continue
            if len(p) > len(g): C["FP: extra"] += len(p) - len(g)
            if len(g) > len(p): C["FN: too few"] += len(g) - len(p); ex.append((v, "FN-few", STEP_DESC[i], g, p))
            for gf, pf in zip(g, p):
                if pf < gf: C["FP: early"] += 1; ex.append((v, "early", STEP_DESC[i], gf - pf))
    print(json.dumps(score(res), indent=1))
    print("error-recording breakdown:", dict(C))
    for e in ex: print("  ", e)
