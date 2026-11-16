"""v11 dev harness (train+val only; test never touched).
ARMODE=oracle: GT AR (diagnostic).  ARMODE=live: teammate's OOF live AR on val (train videos have no live AR -> skipped).
OOF verifiers (4 subject folds), LOSO step models."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("VENS", "verif_base_r336_oof,verif_base_r336_e6u6_oof")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_filter as E
from analyze_errvids import models, score
from psr.filter_tracker import FilterConfig, FilterTracker
from psr.core import frame_paths, procedure_type
from psr.ar import ARObs
LIVE_DIR = os.environ.get("LIVE_AR_DIR", "")   # folder with <video>/ar.jsonl (live AR predictions; see README)
V9 = dict(tau=0.06, theta=0.8, stable=3, conf_n=6, conf_thr=5.0, q_ref=0.1, q_min=0.1, w_model=0.6, eps=0.0)
_live = {}
def live_rows(v):
    if v not in _live:
        _live[v] = [json.loads(l) for l in open(f"{LIVE_DIR}/{v}/ar.jsonl", encoding="utf-8")]
    return _live[v]
def live_obs(v):
    return [[] if r["top"][0][0] == "background" else [r["top"][0][0]] for r in live_rows(v)]
def items(armode):
    return [(sp, v) for sp, v in E.ITEMS if armode == "oracle" or sp == "val"]
def run(sp, v, fcfg, model, armode="oracle", tracker_cls=FilterTracker, obs_fn=None):
    n = len(frame_paths(sp, v))
    if armode == "oracle":
        if (sp, v) not in E._obs:
            E.run.__globals__  # noqa
            from psr.ar import OracleARProvider
            ar = OracleARProvider(); ar.reset(sp, v, n); E._obs[sp, v] = [[o.name for o in ar.observe(t)] for t in range(n)]
        obs = E._obs[sp, v]
    else:
        obs = (obs_fn or live_obs)(v)
    assert len(obs) == n
    tr = tracker_cls(procedure_type(v), fcfg, model, E.TCFG)
    P = E.PROBS[v]
    for t in range(n):
        tr.update(t, obs[t], P.get(t))
    tr.finish(n - 1)
    return tr
def evaluate(fcfg, M, armode="oracle", **kw):
    return [(sp, v, *(lambda t: (t.y_hat, t.y_hat_err))(run(sp, v, fcfg, M[v[:2]], armode, **kw))) for sp, v in items(armode)]
def fmt(s):
    a, e, w = s["all"], s["err"], s["with_errors_labels"]
    return f"all {a['POS']:.3f}/{a['F1']:.3f}/{a['delay_frames']/10:.1f}s FP{a['FP']} FN{a['FN']} | err {e['POS']:.3f}/{e['F1']:.3f}/{e['delay_frames']/10:.1f}s | w/err {w['POS']:.3f}/{w['F1']:.3f} (n={a['n_videos']})"
if __name__ == "__main__":
    M = models()
    for mode in ("oracle", "live"):
        r = evaluate(FilterConfig(**V9), M, mode)
        print(mode, fmt(score(r)))
        if mode == "oracle":
            print(" oracle val-only", fmt(score([x for x in r if x[0] == "val"])))
