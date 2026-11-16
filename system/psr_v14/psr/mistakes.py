"""Mistake detector: per-frame P(assembly shows an execution error), evaluated against the frame-level error-state
labels of Schoonbeek et al., "Supervised Representation Learning Towards Generalizable Assembly State Recognition"
(RA-L 2024; third_party/asr_error_labels/error_labels.json).

Detector: [CLS, mean patch] features of the frozen v9 verifier backbone (verif_base_r336) on the localizer crop
-> standardisation -> MLP (256 hidden) -> causal mean of the last 5 scores (frame t uses crops <= t only).

Labels (training and scoring): a crop is an error (1) if it lies within +-1 frame of a labelled error frame, normal (0)
if it is more than NEG_GAP frames from every labelled error frame and from every frame where PSR_labels_raw.csv marks
a component as incorrectly installed (-1), otherwise ignored.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import video_dir

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "third_party" / "asr_error_labels" / "error_labels.json"
NEG_GAP = 150
SMOOTH = 5
CATN = {1: "missing component", 2: "incorrect placement", 3: "incorrect orientation", 4: "part-level"}

_ERRF = None


def error_frames():
    """video -> [(frame, error category)] from error_labels.json (all recordings; the file's own 'split' field is
    NOT IndustReal's PSR split and is ignored: splits always follow IndustReal's folders)."""
    global _ERRF
    if _ERRF is None:
        d = json.load(open(LABELS))
        _ERRF = defaultdict(list)
        for im, cat in zip(d["images"], d["error_cat"]):
            p = im[:-4].split("_")
            _ERRF["_".join(p[:4])].append((int(p[4]), cat))
    return _ERRF


def _raw_bad(split, v):
    out = [int(p[0][:-4]) for p in (l.strip().split(",") for l in open(video_dir(split, v) / "PSR_labels_raw.csv"))
           if len(p) >= 12 and "-1" in p[1:12]]
    return np.array(out or [-10**9])


def labels(split, v, t):
    """y: 1 error / 0 normal / -1 ignored ; cat: category of the nearest labelled error frame for y == 1, else 0"""
    e = error_frames().get(v, [])
    f = np.array([x for x, _ in e] or [-10**9]); c = np.array([x for _, x in e] or [0])
    d = np.abs(t[:, None] - f[None]); dmin = d.min(1); cat = c[d.argmin(1)]
    dbad = np.abs(t[:, None] - _raw_bad(split, v)[None]).min(1)
    y = np.full(len(t), -1); y[(dmin > NEG_GAP) & (dbad > NEG_GAP)] = 0; y[dmin <= 1] = 1
    return y, np.where(y == 1, cat, 0)


def smooth(e, k=SMOOTH):
    """causal moving mean of the last k scores"""
    if k <= 1:
        return e
    c = np.cumsum(np.insert(e, 0, 0.0)); i = np.arange(1, len(e) + 1); lo = np.maximum(0, i - k)
    return (c[i] - c[lo]) / (i - lo)


def metrics(Y, CAT, P, seed=0):
    """frame AUC / AP overall and per category (each category vs all normal frames), plus a balanced AP
    (each error frame vs one random normal frame, chance 0.5, mean of 20 draws)"""
    out = {"frames_error": int(Y.sum()), "frames_normal": int((Y == 0).sum()),
           "auc": round(roc_auc_score(Y, P), 3), "ap": round(average_precision_score(Y, P), 3), "ap_chance": round(float(Y.mean()), 3)}
    rng = np.random.default_rng(seed); neg = P[Y == 0]
    for k, name in CATN.items():
        pos = P[CAT == k]
        if not len(pos):
            continue
        m = (Y == 0) | (CAT == k); yk = (CAT[m] == k).astype(int)
        bal = [average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(pos))],
                                       np.r_[pos, rng.choice(neg, len(pos), replace=False)]) for _ in range(20)]
        out[name] = {"frames": int(len(pos)), "auc": round(roc_auc_score(yk, P[m]), 3),
                     "ap": round(average_precision_score(yk, P[m]), 3), "ap_chance": round(float(yk.mean()), 4),
                     "balanced_ap": round(float(np.mean(bal)), 3)}
    return out
