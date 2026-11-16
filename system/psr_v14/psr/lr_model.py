"""Discriminative step-group model: multinomial logistic regression over causal trigger features.

Same interface as tracker.GroupModel.score(); trained on TRAIN samples from build_tables.video_samples.
Illegal groups (impossible from the believed state) are masked out and probabilities renormalised.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .core import code, group_is_legal


def _bucket(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def featurize(proc, state_code, trigger, window, since, ntrig, duration, elapsed, post=None, post_d=0, tsince=None, vis=None):
    f = {f"proc={proc}": 1, f"trig={trigger}": 1, f"proc_trig={proc}|{trigger}": 1,
         f"ntrig={min(ntrig, 4)}": 1,
         f"dur={_bucket(duration, [5, 10, 20, 40, 80])}": 1,
         f"elapsed={_bucket(elapsed, [50, 100, 200, 400, 800])}": 1,
         "log_dur": math.log1p(duration) / 5, "log_elapsed": math.log1p(elapsed) / 7}
    for i, b in enumerate(state_code):
        f[f"bit{i}={b}"] = 1
    f[f"state={proc}|{state_code}"] = 1
    for a in window:
        f[f"w:{a}"] = 1
    for a in since:
        f[f"s:{a}"] = 1
    for a, k in (tsince or {}).items():
        f[f"ts:{a}={min(k, 3)}"] = 1
        f[f"ts:{a}|{trigger}={min(k, 3)}"] = 1
    if vis:
        for k, (b, a) in vis.items():
            f[f"vb{k}"] = max(-1.0, min(1.0, b / 5)); f[f"va{k}"] = max(-1.0, min(1.0, a / 5))
            f[f"dv{k}"] = max(-1.5, min(1.5, (a - b) / 5))
            f[f"dv{k}|bit={state_code[k]}"] = max(-1.5, min(1.5, (a - b) / 5))
    if post_d:
        seen = [(a, o) for a, o in (post or {}).items() if o <= post_d]
        for a, o in seen:
            f[f"p:{a}"] = 1
            f[f"p:{a}@{_bucket(o, [15, 40, 80])}"] = 1
        f["p_none"] = int(not seen)
    return f


def key(g):
    return ",".join(map(str, g))


class LRGroupModel:
    def __init__(self, samples=None, C=0.5, path: Path | None = None, post_d: int = 0, use_tsince: bool = False,
                 use_vis: bool = False):
        if path is not None:
            d = pickle.loads(Path(path).read_bytes())
            self.vec, self.clf, self.groups = d["vec"], d["clf"], d["groups"]
            self.post_d = d.get("post_d", 0)
            self.use_tsince = d.get("use_tsince", False)
            self.use_vis = d.get("use_vis", False)
            return
        self.post_d = post_d
        self.use_tsince = use_tsince
        self.use_vis = use_vis
        X = [featurize(s["proc"], s["state"], s["trigger"], s["window"], s["since"], s["ntrig_raw"],
                       s["duration"], s["elapsed"], s.get("post"), post_d,
                       s.get("tsince") if use_tsince else None, s.get("vis") if use_vis else None) for s in samples]
        y = [key(s["group"]) for s in samples]
        self.vec = DictVectorizer()
        Xv = self.vec.fit_transform(X)
        self.clf = LogisticRegression(C=C, max_iter=3000)
        self.clf.fit(Xv, y)
        self.groups = [tuple(int(x) for x in k.split(",")) if k else () for k in self.clf.classes_]

    def save(self, path):
        Path(path).write_bytes(pickle.dumps({"vec": self.vec, "clf": self.clf, "groups": self.groups, "post_d": self.post_d, "use_tsince": self.use_tsince, "use_vis": self.use_vis}))

    def score(self, proc, state, trigger, window, since, cfg, ntrig=1, ctx=None):
        ctx = ctx or {}
        x = featurize(proc, code(state), trigger, window, since, ntrig, ctx.get("duration", 0), ctx.get("elapsed", 0),
                      ctx.get("post"), self.post_d, ctx.get("tsince") if self.use_tsince else None,
                      ctx.get("vis") if self.use_vis else None)
        lp = self.clf.predict_log_proba(self.vec.transform([x]))[0]
        out = {}
        for g, v in zip(self.groups, lp):
            if g == () or group_is_legal(state, g):
                out[g] = float(v) + (cfg.null_bias if g == () else 0.0)
        m = max(out.values())
        z = math.log(sum(math.exp(v - m) for v in out.values())) + m
        return {g: v - z for g, v in out.items()}
