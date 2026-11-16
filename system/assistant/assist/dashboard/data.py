"""What the replay dashboard shows, read from files the pipeline already wrote. No web code, no models.

Per-recording scoring reuses ``gt_mistakes``, ``collapse`` and ``match`` from ``assist.evaluation.eval``, so
the line totals are the numbers in ``eval_test.json``. Every mistake alert gets an ``outcome``:
``caught`` (matched to a real mistake), ``false_alarm``, or ``duplicate`` (a "possible" install alert later
confirmed: counted once, as in the evaluation).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from assist.common import ROOT, load_config
from assist.engine.graph import load_tasks
from assist.evaluation.eval import RUN_NAMES, TYPES, collapse, gt_mistakes, match
from assist.evaluation.psr_eval import PUBLISHED_PSR
from assist.industreal import SPLITS, task_of

# v14-psr branch: the live run is the engine with v14's PSR steps as its state (TCN actions). The original live run
# (pred_vjepap_asd) needs V-JEPA 2 outputs that are not on this machine; it is still shown when its files exist.
LIVE_RUN = "pred_asdv14"
RUNS = {LIVE_RUN: "Live system + v14 PSR", "pred_v14": "v14 PSR alone", "pred_asd": "Live system (TCN actions + ASD)",
        "pred_vjepap_asd": "Live system (TCN + V-JEPA 2 + ASD)", "gt_gt": "Oracle"}
RUN_NOTES = {LIVE_RUN: "Causal TCN actions, with the frozen v14 PSR (part verifiers + ASD) as the state input of the engine.",
             "pred_v14": "The frozen v14 PSR on its own: its reported steps and its own mistake alerts (no engine rules).",
             "pred_asd": "Causal TCN actions and the released state detector, as it would run live.",
             "pred_vjepap_asd": "Actions from the causal TCN + V-JEPA 2, and the released state detector, as it would run live.",
             "gt_gt": "Annotated actions and states: the upper bound of the rules."}
RIDS = SPLITS["test"]
OUT = ROOT / "output/assist"
VIDEOS = OUT / "web_videos"
CFG = load_config()
TASKS = load_tasks(CFG["graph_path"])
RESULT_RUNS = {    # results page, weakest state source first
    "pred_vjepap_none": "Live actions only, no state", "gt_none": "Annotated actions only, no state",
    "baseline_psr": "Baseline: official PSR + same rules", **RUNS}


# short firm commands that carry over factory noise; {name} is the step or part, {first} what must be done before it.
# An uncertain install ("the camera did not confirm it") asks for a check and does not say stop.
SPOKEN = {"wrong_order": "Stop. Wrong order. {name}. First do: {first}.", "skipped_step": "Stop. Step skipped. {name}. Do it now.",
          "incorrect_install": "Stop. Installed wrong. {name}. Fix it now.",
          "possible_incorrect_install": "Check now. {name}. Not confirmed.", "wrong_part": "Stop. Wrong part. {name}. Remove it."}


def _without_brackets(text: str) -> str:
    """Asides read badly aloud. Innermost first, since a step's name can have brackets inside another pair."""
    while True:
        shorter = re.sub(r"\s*\([^()]*\)", "", text)
        if shorter == text:
            return text
        text = shorter


def spoken_text(alert: dict) -> str | None:
    """What the dashboard says for an alert, written for the ear; None for alerts that stay silent."""
    if alert["type"] == "cannot_verify":
        return None
    title = _without_brackets(alert["title"])
    if alert["type"] == "early_warning":
        step = title.split(": ", 1)[-1].removesuffix(" is not next")
        subject = f"Removing{step[len('remove'):]}" if step.startswith("remove ") else f"The {step}"
        return f"Wait. {subject} is not next."
    name = title.split(": ", 1)[-1].removeprefix("Check the ")
    before = re.search(r"was done before (.+?)\.", _without_brackets(alert["why"]))
    return SPOKEN[alert["type"]].format(name=name[0].upper() + name[1:], first=before.group(1) if before else "")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def available_runs() -> dict[str, str | None]:
    """run -> None if every test recording has its engine result, else the reason it cannot be shown."""
    out = {}
    for run in RUNS:
        missing = [rid for rid in RIDS if not (OUT / rid / f"engine_{run}.json").exists()]
        out[run] = None if not missing else (f"{len(missing)} recordings have no engine_{run}.json; run "
                                             f"python -m assist.run_engine first")
    return out


@lru_cache(maxsize=None)
def engine(rid: str, run: str) -> dict:
    return _read(OUT / rid / f"engine_{run}.json")


@lru_cache(maxsize=None)
def score(rid: str, run: str) -> dict:
    """Real mistakes (each with ``caught`` and the alert time), alerts with ``outcome``, and counts."""
    res = engine(rid, run)
    window = CFG["evaluation"]["match_window_s"]
    gts = gt_mistakes(rid, TASKS[task_of(rid)], CFG["components"], CFG["paths"]["labels"], CFG["fps"],
                      res["duration_s"], CFG["engine"]["order_grace_s"])
    raised = [a for a in res["alerts"] if a["kind"] == "mistake"]
    counted = collapse(raised, window)
    pairs = match(gts, counted, window)
    caught_by = {id(g): a for g, a in pairs}
    caught_ids = {id(a) for _, a in pairs}
    counted_ids = {id(a) for a in counted}
    alerts = []
    for a in res["alerts"]:
        out = dict(a)
        if a["kind"] == "mistake":
            out["outcome"] = ("caught" if id(a) in caught_ids else
                              "false_alarm" if id(a) in counted_ids else "duplicate")
        alerts.append(out)
    mistakes = [{**g, "caught": id(g) in caught_by, "alert_t": caught_by[id(g)]["t"] if id(g) in caught_by else None,
                 "step_name": TASKS[task_of(rid)].steps[g["step"]].name if g["step"] else g["component"]}
                for g in gts]
    return {"alerts": alerts, "mistakes": mistakes, "n_mistakes": len(gts), "n_alerts": len(counted),
            "caught": len(pairs), "missed": len(gts) - len(pairs), "false_alarms": len(counted) - len(pairs)}


def _outcome(s: dict) -> str:
    return "amber" if s["missed"] or s["false_alarms"] else "red" if s["caught"] else "green"


def stations(run: str) -> dict:
    rows, hours, tot = [], 0.0, {"mistakes": 0, "alerts": 0, "caught": 0, "missed": 0, "false_alarms": 0}
    for rid in RIDS:
        res, s = engine(rid, run), score(rid, run)
        hours += res["duration_s"] / 3600
        for k, src in (("mistakes", "n_mistakes"), ("alerts", "n_alerts"), ("caught", "caught"),
                       ("missed", "missed"), ("false_alarms", "false_alarms")):
            tot[k] += s[src]
        by_type = {}
        for a in s["alerts"]:
            if a["kind"] != "status":
                by_type[a["type"]] = by_type.get(a["type"], 0) + 1
        rows.append({"rid": rid, "task": res["task"], "task_title": res["task_title"],
                     "duration_s": res["duration_s"],
                     "steps_done": sum(st["final"] == "done" for st in res["steps"]), "steps_total": len(res["steps"]),
                     "alerts_by_type": by_type, "mistakes": s["n_mistakes"], "caught": s["caught"],
                     "missed": s["missed"], "false_alarms": s["false_alarms"], "outcome": _outcome(s)})
    tot["hours"] = hours
    tot["false_alarms_per_h"] = tot["false_alarms"] / hours
    return {"run": run, "label": RUNS[run], "stations": rows, "totals": tot}


def station(rid: str, run: str) -> dict:
    res, s = engine(rid, run), score(rid, run)
    return {**res, "alerts": [{**a, "spoken": spoken_text(a)} for a in s["alerts"]], "mistakes": s["mistakes"],
            "summary": {k: s[k] for k in ("n_mistakes", "n_alerts", "caught", "missed", "false_alarms")},
            "has_video": (VIDEOS / f"{rid}.mp4").exists(), "run": run, "run_label": RUNS[run]}


def analytics() -> dict:
    psr = _read(OUT / "eval/psr_test_release.json")
    psr_rows = [{**row, "published": True} for row in PUBLISHED_PSR]
    names = {f"engine_{LIVE_RUN}": "Ours: engine + v14 PSR as state (TCN actions)", "engine_pred_asd": "Ours: engine, TCN actions + ASD"}
    for key in ("official_release", "engine_pred_asd", f"engine_{LIVE_RUN}", "engine_gt_gt"):
        if key in psr:
            s = psr[key]["summary"]
            psr_rows.append({"name": names.get(key, psr[key]["name"]), "f1": s["all"]["f1"], "pos": s["all"]["pos"],
                             "delay_s": s["all"]["delay_s"], "published": False, "with_errors": s["with_errors"]})
    cmp = OUT / "v14/compare.json"          # v14's own steps, scored with this repository's port of the official code
    if cmp.exists():
        v = _read(cmp)["v14_steps_our_scoring"]
        psr_rows.insert(len(PUBLISHED_PSR) + 1, {"name": "v14 PSR alone (frozen, one test run)", "f1": v["all"]["f1"],
                        "pos": v["all"]["pos"], "delay_s": v["all"]["delay_s"], "published": False, "with_errors": v["with_errors"]})
    mistakes = []
    evals = {r["run"]: r for r in _read(OUT / "eval/eval_test.json")}
    for run, name in RESULT_RUNS.items():
        if run not in evals:
            continue
        r = evals[run]
        mistakes.append({"run": run, "name": name,
                         "description": RUN_NAMES.get(r["run"], r["run"]),
                         "rows": {k: {f: r["rows"][k][f] for f in ("gt", "alerts", "tp", "precision", "recall",
                                                                   "false_alarms_per_h")} for k in (*TYPES, "all")}})
    a101_file = OUT / "assembly101/assembly101_o.json"          # second dataset; optional on this machine
    if not a101_file.exists():
        return {"psr": psr_rows, "mistakes": mistakes, "assembly101": []}
    a = _read(a101_file)
    cols = ("precision", "recall", "f1", "precision_b", "f1_b", "avg_f1")
    a101 = [{"name": n, "steps": steps, **dict(zip(cols, vals)), "ours": False} for n, steps, *vals in a["published"]]
    a101 += [{"name": "Ours: task-graph rules", "steps": steps, **{c: a["test"][steps][c] for c in cols}, "ours": True}
             for steps in ("oracle", "MiniROAD")]
    return {"psr": psr_rows, "mistakes": mistakes, "assembly101": a101}
