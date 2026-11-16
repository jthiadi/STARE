"""Blend of the count model (naive Bayes) and the logistic-regression step model, and loading of a
frozen configuration directory (config.json, tables.json, lr.pkl)."""
from __future__ import annotations

import json
from pathlib import Path

from .lr_model import LRGroupModel
from .tracker import GroupModel, TrackerConfig


class Blend:
    def __init__(self, nb, lr, w):
        self.nb, self.lr, self.w = nb, lr, w

    def score(self, proc, state, trigger, window, since, cfg, ntrig=1, ctx=None):
        a = self.nb.score(proc, state, trigger, window, since, cfg, ntrig, ctx)
        b = self.lr.score(proc, state, trigger, window, since, cfg, ntrig, ctx)
        return {g: a[g] + self.w * b.get(g, -30.0) for g in a}


def load_frozen(path: Path):
    """-> (TrackerConfig, model, vision_spec dict or None)"""
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    cfg = TrackerConfig.from_file(path / "config.json")
    nb = GroupModel(path / "tables.json")
    m = raw.get("model", {"type": "nb"})
    model = nb if m["type"] == "nb" else Blend(nb, LRGroupModel(path=path / m["lr_path"]), m["w_lr"])
    return cfg, model, raw.get("vision")
