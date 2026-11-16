"""Mistake detection scored exactly like the teammate's assist/evaluation/eval.py (same GT rules, collapse, matching).
Alerts = a step stream replayed into states and checked with the same rules (as their baseline_psr)."""
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ.get("TEAMMATE_AR", ""))   # teammate repository with assist/ (not included)
from assist.common import load_config
from assist.engine.graph import load_tasks
from assist.industreal import task_of
from assist.evaluation.eval import gt_mistakes, collapse, match, TYPE_OF, TYPES
from assist.evaluation.baseline import state_changes, ALERT_TYPE
CFG = load_config(); TASKS = load_tasks(CFG["graph_path"])
# Ground-truth labels for scoring (val only is scored in selection/v14): <dataset>/validation/val_annotations/<rid>/
LABELS = Path(os.environ.get("INDUSTREAL_DATA", "data")) / "validation" / "val_annotations"
FPS = CFG["fps"]; GRACE = CFG["engine"]["order_grace_s"]; WIN = CFG["evaluation"]["match_window_s"]
_gt = {}
def gt(v, n):
    if v not in _gt:
        _gt[v] = gt_mistakes(v, TASKS[task_of(v)], CFG["components"], LABELS, FPS, n / FPS, GRACE)
    return _gt[v]
def alerts_from_steps(v, steps, n):
    task = TASKS[task_of(v)]
    ch = state_changes(sorted(steps, key=lambda s: s["frame"]), task.initial_state)
    ms = gt_mistakes(v, task, CFG["components"], LABELS, FPS, n / FPS, GRACE, changes=ch)
    return [{"t": m["t"], "type": ALERT_TYPE[m["type"]], "kind": "mistake", "step": m["step"], "component": m["component"]} for m in ms]
def score(per_video):
    """per_video: list of (v, n_frames, alerts)."""
    st = {k: {"gt": 0, "alerts": 0, "tp": 0, "d": []} for k in TYPES}; hours = 0
    for v, n, al in per_video:
        hours += n / FPS / 3600
        g = gt(v, n); m = collapse(al, WIN)
        for k in TYPES:
            st[k]["gt"] += sum(1 for x in g if x["type"] == k); st[k]["alerts"] += sum(1 for a in m if TYPE_OF[a["type"]] == k)
        for a, b in match(g, m, WIN):
            st[a["type"]]["tp"] += 1; st[a["type"]]["d"].append(b["t"] - a["t"])
    tot = {x: sum(st[k][x] for k in TYPES) for x in ("gt", "alerts", "tp")}
    tot["fa_h"] = (tot["alerts"] - tot["tp"]) / hours; tot["hours"] = hours
    tot["prec"] = tot["tp"] / tot["alerts"] if tot["alerts"] else 0; tot["rec"] = tot["tp"] / tot["gt"] if tot["gt"] else 0
    d = [x for k in TYPES for x in st[k]["d"]]; tot["delay"] = sum(d) / len(d) if d else None
    tot["by"] = {k: f"{st[k]['tp']}/{st[k]['gt']} ({st[k]['alerts']} al)" for k in TYPES}
    return tot
def fmt(t):
    return f"mistakes {t['tp']}/{t['gt']} prec {t['prec']:.2f} rec {t['rec']:.2f} FA/h {t['fa_h']:.1f} delay {t['delay'] if t['delay'] is None else round(t['delay'],1)}s {t['by']}"
