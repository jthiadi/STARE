"""Leave-one-SUBJECT-out over TRAIN+VAL (16 subjects). Test untouched.  Model families: hold / post20."""
import itertools, json, warnings
warnings.filterwarnings("ignore")
import build_tables as bt
from psr.ar import OracleARProvider
from psr.blend import Blend
from psr.core import PROC_INFO, frame_paths, list_videos, load_psr_gt, procedure_type
from psr.lr_model import LRGroupModel
from psr.official_eval import aggregate, evaluate
from psr.tracker import GroupModel, PSRTracker, TrackerConfig

ITEMS = [("train", v) for v in list_videos("train")] + [("val", v) for v in list_videos("val")]
SUBS = sorted({v[:2] for _, v in ITEMS})
def samples(items):
    out = []
    for sp, v in items: out += [json.loads(s) for s in bt.video_samples(sp, v)[0]]
    return out
_obs = {}
def run(sp, v, cfg, model):
    n = len(frame_paths(sp, v))
    if (sp, v) not in _obs:
        ar = OracleARProvider(); ar.reset(sp, v, n); _obs[sp, v] = [[o.name for o in ar.observe(t)] for t in range(n)]
    tr = PSRTracker(procedure_type(v), cfg, model=model)
    for t in range(n): tr.update(t, _obs[sp, v][t])
    if tr.pending: tr._commit(n - 1, forced=True)
    return evaluate(load_psr_gt(sp, v), tr.y_hat, PROC_INFO)

if __name__ == "__main__":
    nb = {s: GroupModel(bt.build([it for it in ITEMS if it[1][:2] != s])) for s in SUBS}
    res = []
    for fam, C in itertools.product(["hold", "post20", "nb_only"], [1.0, 3.0]):
        if fam == "nb_only" and C == 3.0: continue
        D = 20 if fam == "post20" else 0
        lr = {s: LRGroupModel(samples([it for it in ITEMS if it[1][:2] != s]), C=C, post_d=D) for s in SUBS} if fam != "nb_only" else None
        ws = [0.0] if fam == "nb_only" else [1.0, 2.0, 4.0]
        for w, mp, nbias, wa, wsn in itertools.product(ws, [0.3, 0.4, 0.5], [0.0, 1.0, 2.0], [0.6], [0.3]):
            extra = dict(post_window=D) if D else dict(min_wait=8, hold=True, max_hold=60)
            cfg = TrackerConfig(w_action=wa, w_since=wsn, min_prob=mp, null_bias=nbias, **extra)
            per = [run(sp, v, cfg, nb[v[:2]] if lr is None else Blend(nb[v[:2]], lr[v[:2]], w)) for sp, v in ITEMS]
            a = aggregate(per)
            a_tr = aggregate([p for p, (sp, _) in zip(per, ITEMS) if sp == "train"])
            a_va = aggregate([p for p, (sp, _) in zip(per, ITEMS) if sp == "val"])
            a["cfg"] = dict(family=fam, C=C, w_lr=w, min_prob=mp, null_bias=nbias, w_action=wa, w_since=wsn)
            a["train_part"] = [a_tr["POS"], a_tr["F1"], float(a_tr["delay_frames"])]
            a["val_part"] = [a_va["POS"], a_va["F1"], float(a_va["delay_frames"])]
            res.append(a)
        b = max([r for r in res if r["cfg"]["family"] == fam and r["cfg"]["C"] == C], key=lambda r: r["POS"] + r["F1"])
        print(f"{fam:8} C={C} best LOSO(all16) {b['POS']:.3f} {b['F1']:.3f} {float(b['delay_frames']):5.1f} FP{b['FP']} FN{b['FN']} | train-part {b['train_part'][0]:.3f} {b['train_part'][1]:.3f} | val-part {b['val_part'][0]:.3f} {b['val_part'][1]:.3f} {b['val_part'][2]:.1f} | {b['cfg']}", flush=True)
    res.sort(key=lambda r: -(r["POS"] + r["F1"]))
    json.dump(res[:30], open("exp_trainval.json", "w"), indent=1, default=float)
    print("BEST", json.dumps(res[0]["cfg"]))
