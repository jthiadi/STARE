"""Evaluate the frozen v14 PSR system and its mistake alerts with the UNMODIFIED official IndustReal evaluator.

v14 = v13 (psr/v11.py, tau 0.05, theta 0.6, d0 12) + the official IndustReal ASD as extra whole-state evidence
(psr/v14.py; model/config.json "asd"), chosen by the pre-registered val selection (EXPERIMENT_LOG).
ASD outputs are read from cache/asd_<split>/<video>.npz (tools/run_asd.py).

    python eval_v14.py --split test --ar predicted --ar-dir <dir with <video>/ar.jsonl>   # deployable
    python eval_v14.py --split test --ar gt                                              # ORACLE diagnostic

AR sources
  predicted : causal action-recognition predictions (one ar.jsonl per video, one row per frame:
              {"f": frame, "top": [[action, prob], ...]}); at frame t the top-1 action is used unless it is
              "background".  This is the deployable setting.
  gt        : ground-truth AR labels (AR_labels.csv), revealed causally.  ORACLE: an upper bound for diagnosis,
              never deployable performance.

Inputs: verifier probabilities in cache/<tag>_<split>/<video>.npz (produced by run_pipeline.py).
Ground-truth PSR labels are read only below the "evaluation only" line, after the tracker has finished a video.
Reports POS / F1 / average delay on PSR_labels.csv (all / recordings with errors / without errors, the paper's
Table 4 layout), plus the same metrics on PSR_labels_with_errors.csv (extra; not part of the official protocol).
"""
import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

from psr.ar import ARObs, ARProvider, OracleARProvider
from psr.blend import Blend
from psr.core import PROC_INFO, frame_paths, list_videos, load_psr_gt, procedure_type, video_dir
from psr.filter_tracker import FilterConfig, FilterTracker
from psr.lr_model import LRGroupModel
from psr.official_eval import aggregate, evaluate, psr_utils
from psr.tracker import GroupModel, TrackerConfig
from psr.vision import CACHE
from psr.v11 import mistake_alerts, score_mistakes
from psr.v14 import V14Tracker, asd_states

HERE = Path(__file__).resolve().parent


class PredictedARProvider(ARProvider):
    """Causal AR predictions from <ar_dir>/<video>/ar.jsonl: top-1 action at frame t, none if background."""
    kind = "predicted"

    def __init__(self, ar_dir):
        self.ar_dir = Path(ar_dir)

    def reset(self, split, video, n_frames):
        rows = [json.loads(l) for l in open(self.ar_dir / video / "ar.jsonl", encoding="utf-8")]
        assert len(rows) == n_frames and all(r["f"] == i for i, r in enumerate(rows)), f"{video}: ar.jsonl length"
        self.top = [r["top"][0] for r in rows]

    def observe(self, t):
        name, p = self.top[t]
        return [] if name == "background" else [ARObs(name, -1, float(p))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ar", required=True, choices=["predicted", "gt"])
    ap.add_argument("--ar-dir", help="folder with <video>/ar.jsonl (required for --ar predicted)")
    ap.add_argument("--model", default=str(HERE / "model"), help="frozen model folder (config.json, tables.json, lr.pkl)")
    ap.add_argument("--out", default=str(HERE / "results" / "reproduced"))
    a = ap.parse_args()
    if a.ar == "predicted" and not a.ar_dir:
        ap.error("--ar predicted needs --ar-dir")

    fz = Path(a.model)
    cfg = json.loads((fz / "config.json").read_text())
    sm = cfg["step_model"]
    model = Blend(GroupModel(fz / "tables.json"), LRGroupModel(path=fz / sm["lr_path"]), sm["w_lr"])
    tcfg = TrackerConfig(w_action=sm["w_action"], w_since=sm["w_since"], null_bias=sm["null_bias"])
    fcfg = FilterConfig(**cfg["filter"])

    def probs(v):
        """mean of the two verifiers' P(component attached), as (absent, correct, incorrect) per component"""
        acc, cnt, qv = {}, {}, {}
        for ver in cfg["verifiers"]:
            z = np.load(CACHE / f"{ver['tag']}_{a.split}" / f"{v}.npz")
            for t, q in zip(z["t"], z["q"]):
                qv[int(t)] = float(q)
            for t, p in zip(z["t"], z["p"]):
                p = p.astype(np.float32)
                p3 = np.stack([1 - p, p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
                acc[int(t)] = acc.get(int(t), 0) + p3; cnt[int(t)] = cnt.get(int(t), 0) + 1
        return {t: {"cls": acc[t] / cnt[t], "q": qv.get(t, 0.0)} for t in acc}

    def f(m):
        return {"POS": round(m["POS"], 3), "F1": round(m["F1"], 3), "delay_frames": round(float(m["delay_frames"]), 1),
                "delay_s": round(float(m["delay_frames"]) / 10, 2),
                "TP": int(m["TP"]), "FP": int(m["FP"]), "FN": int(m["FN"]), "n_videos": m["n_videos"]}

    per, per_err, per_ok, per_we, rows, mist = [], [], [], [], [], []
    t0 = time.time()
    for v in list_videos(a.split):
        n = len(frame_paths(a.split, v))
        ar = PredictedARProvider(a.ar_dir) if a.ar == "predicted" else OracleARProvider()
        ar.reset(a.split, v, n)
        ac = cfg["asd"]; asd = asd_states(np.load(CACHE / f"asd_{a.split}" / f"{v}.npz")["rows"], ac["cmin"])
        tr = V14Tracker(procedure_type(v), fcfg, model, tcfg, d0=cfg['v11']['d0'], asd=asd, w=ac["w"], eps=ac["eps"], rate=ac["rate"])
        P = probs(v)
        for t in range(n):                                  # causal: frame t sees only AR and readings up to t
            tr.update(t, [o.name for o in ar.observe(t)], P.get(t))
        tr.finish(n - 1)
        alerts = mistake_alerts(v, tr, n)       # causal: steps so far + the state at the end
        # ---- evaluation only below this line ----
        mist.append((a.split, v, n, alerts))
        gt = load_psr_gt(a.split, v)
        m = evaluate(gt, tr.y_hat, PROC_INFO)
        per.append(m)
        err = psr_utils.video_contains_errors(gt, PROC_INFO, video_dir(a.split, v))
        (per_err if err else per_ok).append(m)
        per_we.append(evaluate(load_psr_gt(a.split, v, with_errors=True), tr.y_hat_err, PROC_INFO))
        rows.append({"video": v, "errors": bool(err), "pos": round(float(m["pos"]), 3), "f1": round(float(m["f1"]), 3),
                     "delay_frames": round(float(m["avg_delay"]), 1), "FP": int(m["system_FPs"]), "FN": int(m["system_FNs"]),
                     "predicted_steps": [(e["frame"], e["description"]) for e in tr.y_hat],
                     "mistake_alerts": alerts})
    out = {"system": "v14", "split": a.split,
           "setting": "PREDICTED AR (deployable)" if a.ar == "predicted" else "GT AR (ORACLE, diagnostic only)",
           "all": f(aggregate(per)), "with_errors": f(aggregate(per_err)) if per_err else None,
           "without_errors": f(aggregate(per_ok)) if per_ok else None,
           "with_errors_labels_all": f(aggregate(per_we)),   # extra: PSR_labels_with_errors.csv, not official
           "mistakes": score_mistakes(mist),          # teammate's rules + matching (assist/evaluation/eval.py)
           "wall_s": round(time.time() - t0, 1)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f"v14_{'predictedAR' if a.ar == 'predicted' else 'gtAR'}_{a.split}.json"
    (Path(a.out) / name).write_text(json.dumps({"summary": out, "videos": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
