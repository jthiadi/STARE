"""Causality check (val): outputs before a cut point must not change when the video is truncated at the cut.
Compares reported steps (y_hat, y_hat_err) and mistake alerts strictly before the cut, except end-of-recording
alerts (they are emitted at the end by definition).  Reads no labels."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
import eval_v14 as E
from psr.blend import Blend
from psr.core import frame_paths, list_videos, procedure_type
from psr.filter_tracker import FilterConfig
from psr.lr_model import LRGroupModel
from psr.tracker import GroupModel, TrackerConfig
from psr.v11 import mistake_alerts
from psr.v14 import V14Tracker, asd_states
from psr.vision import CACHE
import os
ar_dir = sys.argv[1]; fz = Path(os.environ.get("PSR_MODEL", HERE / "model"))
cfg = json.loads((fz / "config.json").read_text()); sm = cfg["step_model"]
model = Blend(GroupModel(fz / "tables.json"), LRGroupModel(path=fz / sm["lr_path"]), sm["w_lr"])
tcfg = TrackerConfig(w_action=sm["w_action"], w_since=sm["w_since"], null_bias=sm["null_bias"]); fcfg = FilterConfig(**cfg["filter"])
def probs(v):
    acc, cnt, qv = {}, {}, {}
    for ver in cfg["verifiers"]:
        z = np.load(CACHE / f"{ver['tag']}_val" / f"{v}.npz")
        for t, q in zip(z["t"], z["q"]): qv[int(t)] = float(q)
        for t, p in zip(z["t"], z["p"]):
            p = p.astype(np.float32); p3 = np.stack([1 - p, p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
            acc[int(t)] = acc.get(int(t), 0) + p3; cnt[int(t)] = cnt.get(int(t), 0) + 1
    return {t: {"cls": acc[t] / cnt[t], "q": qv.get(t, 0.0)} for t in acc}
def run(v, n_run, n_total):
    ar = E.PredictedARProvider(ar_dir); ar.reset("val", v, n_total); P = probs(v)
    ac = cfg["asd"]; asd = {f: s for f, s in asd_states(np.load(CACHE / "asd_val" / f"{v}.npz")["rows"], ac["cmin"]).items() if f < n_run}
    tr = V14Tracker(procedure_type(v), fcfg, model, tcfg, d0=cfg["v11"]["d0"], asd=asd, w=ac["w"], eps=ac["eps"], rate=ac["rate"])
    for t in range(n_run): tr.update(t, [o.name for o in ar.observe(t)], P.get(t))
    return tr, mistake_alerts(v, tr, n_run)
bad = checks = 0
for v in list_videos("val"):
    n = len(frame_paths("val", v)); full, fa = run(v, n, n)
    for cut in np.linspace(n * 0.2, n * 0.9, 6).astype(int):
        tr, al = run(v, cut, n); checks += 1
        a = [e for e in full.y_hat_err if e["frame"] < cut]; b = [e for e in tr.y_hat_err if e["frame"] < cut]
        # alerts: only those before the cut minus the 10 s order grace, and not end-of-recording ones
        lim = cut / 10 - 10.5
        fa_ = [x for x in fa if x["t"] < lim and x["t"] < n / 10 - 1]; al_ = [x for x in al if x["t"] < lim and x["t"] < cut / 10 - 1]
        if a != b or fa_ != al_:
            bad += 1; print("MISMATCH", v, cut, len(a), len(b), len(fa_), len(al_))
print(f"{checks} cut points, {bad} mismatches")
