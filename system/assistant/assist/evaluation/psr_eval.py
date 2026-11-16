"""Procedure step recognition (PSR) metrics, comparable with the IndustReal / STORM-PSR papers.

Scored with the official IndustReal PSR protocol: F1 of correctly completed steps, POS (procedure order
similarity, a weighted Damerau-Levenshtein score) and the average delay between a step's completion and its
recognition, averaged over videos. The scoring is ported line for line from TimSchoonbeek/IndustReal
``PSR/psr_utils.py`` (checked against STORM-PSR's saved output); ``dam_lev`` reimplements
``weighted_levenshtein.dam_lev`` with the same costs.

Methods (``assist/engine/psr.py``):
  official   the IndustReal accumulated-confidence PSR on a state detector's predictions (paper baseline)
  fused      the same plus our action stream (step finished according to the actions-only engine);
             action weight / duration / threshold chosen on the 16 val videos (``--tune``)
  engine     the step completions in our engine's log (``engine_<run>.json``)
State detectors (``--asd``): ``release`` = the released IndustReal predictions (test only),
``ours`` = our detector (``assist/extract/train_asd.py``, val + test).

    python -m assist.evaluation.psr_eval --tune --asd ours                      # choose fusion settings on val
    python -m assist.evaluation.psr_eval --asd release --runs pred_asd gt_gt    # test table
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from assist.common import ROOT, load_config, load_yaml, write_json
from assist.engine.psr import (AccumulatedConfidencePSR, FusedPSR, action_completions, make_entry,
                               process_asd_predictions, requires_by_id)
from assist.industreal import SPLITS, n_frames, psr_states, task_of

FPS = 10
ASD_DIRS = {"release": None, "ours": ROOT / "output/assist/asd_ours/predictions",
            "weights": ROOT / "output/assist/asd_released/predictions"}
INITIAL = {"assy": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "main": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]}
GRID = {"action_weight": [0.0, 0.25, 0.5, 1.0, 2.0], "action_frames": [10, 30, 60], "threshold": [4.0, 6.0, 8.0]}
# IndustReal's best baseline as printed in its Table 4, and as listed in the STORM-PSR paper's Table 1.
PUBLISHED_PSR = [
    {"name": "IndustReal B3 (WACV'24), as published", "f1": 0.883, "pos": 0.797, "delay_s": 22.4,
     "with_errors": {"f1": 0.816, "pos": 0.731, "delay_s": 20.4}},
    {"name": "IndustReal, as reported in STORM-PSR (CVIU'25)", "f1": 0.891, "pos": 0.797, "delay_s": 21.0,
     "with_errors": None},
    {"name": "STORM-PSR (CVIU'25), as published", "f1": 0.901, "pos": 0.812, "delay_s": 15.5, "with_errors": None}]


# ------------------------------------------------------------------ scoring, ported from IndustReal PSR/psr_utils.py
def dam_lev(a: str, b: str, ins=1.0, dele=1.0, sub=2.0, trans=1.0) -> float:
    """Unrestricted Damerau-Levenshtein distance (Lowrance-Wagner) with uniform weights."""
    inf = len(a) * dele + len(b) * ins + 1
    da: dict[str, int] = {}
    d = [[inf] * (len(b) + 2) for _ in range(len(a) + 2)]
    for i in range(len(a) + 1):
        d[i + 1][1] = i * dele
    for j in range(len(b) + 1):
        d[1][j + 1] = j * ins
    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            k, l_ = da.get(b[j - 1], 0), db
            cost = 0 if a[i - 1] == b[j - 1] else sub
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(d[i][j] + cost, d[i + 1][j] + ins, d[i][j + 1] + dele,
                                  d[k][l_] + (i - k - 1) * dele + trans + (j - l_ - 1) * ins)
        da[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]


def procedure_order_similarity(gt: list, pred: list) -> float:
    to_s = lambda ids: "".join(chr(i + 33) for i in ids)   # noqa: E731
    return 1 - min(dam_lev(to_s(gt), to_s(pred)) / len(gt), 1) if len(gt) else 0.0


def get_f1_score(fn: int, fp: int, tp: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 1e-6
    recall = tp / (tp + fn) if tp + fn else 1e-6
    return 2 * precision * recall / (precision + recall + 1e-6)


def contains_errors(gt: list, proc_info: list, task: str, states: list) -> bool:
    """IndustReal's ``video_contains_errors``: an expected step never completed, or a part ever installed wrongly.

    Doing every step correctly but in another order is not an error here. ``states`` is [(frame, state), ...].
    """
    done = {step["id"] for step in gt}
    missing = any(a[f"expected_in_{task}"] and a["id"] not in done for a in proc_info)
    return missing or any(v == -1 for _, state in states for v in state)


def match_indices(idxes_a, all_times_a, idxes_b, all_times_b):
    times_a = np.ones(len(all_times_a)) * 1e9
    for idx in idxes_a:
        times_a[idx] = all_times_a[idx]
    out = []
    for tb in [all_times_b[i] for i in idxes_b]:
        diff = times_a - tb                # the closest later one; with none, argmin of all-inf gives entry 0, as official
        k = int(np.argmin(np.where(diff > 0, diff, np.inf)))
        out.append(k)
        times_a[k] = 1e9
    return out


def determine_performance(gt: list, pred: list, proc_info: list) -> dict:
    gt_t = np.array([e["frame"] for e in gt], dtype=int)
    gt_o = np.array([int(e["id"]) for e in gt], dtype=int)
    pr_t = np.array([e["frame"] for e in pred], dtype=int)
    pr_o = np.array([int(e["id"]) for e in pred], dtype=int)
    pr_c = np.array([int(e["conf"]) for e in pred])
    sys_fn = sys_fp = 0
    delays = np.full(len(gt), np.nan)
    for step in proc_info:
        ig = list(np.where(gt_o == step["id"])[0])
        ip = list(np.where(pr_o == step["id"])[0])
        if len(ig) == len(ip) and len(ip) > 1:
            ip = match_indices(ip, pr_t, ig, gt_t)
        elif not ig and ip:
            sys_fp += len(ip)
            continue
        elif ig and not ip:
            sys_fn += len(ig)
            continue
        elif not ig and not ip:
            continue
        elif len(ig) > len(ip):
            sys_fn += len(ig) - len(ip)
            ig = match_indices(ig, gt_t, ip, pr_t)
        else:                              # also one prediction and one label: the official code matches those too
            sys_fp += len(ip) - len(ig)
            ip = match_indices(ip, pr_t, ig, gt_t)
        for a, b in zip(ig, ip):
            delta = pr_t[b] - gt_t[a]
            if delta < 0 and pr_c[b] == 1:
                sys_fp += 1
            if delta >= 0:
                delays[a] = delta
    tp = len(pr_o) - sys_fp
    avg = float(np.nanmean(delays)) if np.any(~np.isnan(delays)) else 100.0   # official: no delay at all counts as 100
    return {"f1": get_f1_score(sys_fn, sys_fp, tp), "pos": procedure_order_similarity(list(gt_o), list(pr_o)),
            "avg_delay": avg, "tp": tp, "fp": sys_fp, "fn": sys_fn}
# ------------------------------------------------------------------ end of the ported code


class Data:
    """Everything per video that the PSR runs need, loaded once."""

    def __init__(self, cfg: dict, asd: str):
        self.cfg, self.asd = cfg, asd
        self.labels, self.videos = cfg["paths"]["labels"], cfg["paths"]["videos"]
        self.proc_info = json.loads(Path(cfg["paths"]["procedure_info"]).read_text(encoding="utf-8"))
        self.graph = load_yaml(cfg["graph_path"])["tasks"]
        self.asd_dir = Path(ASD_DIRS[asd] or cfg["paths"]["asd_release"])
        self._cache: dict = {}

    def video(self, rid: str) -> dict:
        if rid not in self._cache:
            frames = n_frames(rid, self.labels, self.videos)
            task = task_of(rid)
            actions = {}
            pred_none = self.cfg["paths"]["out"] / rid / "engine_pred_none.json"
            if pred_none.is_file():
                actions = action_completions(json.loads(pred_none.read_text(encoding="utf-8")), self.graph[task],
                                             self.cfg["components"], FPS)
            gt = load_gt(rid, self.labels)
            self._cache[rid] = {
                "frames": frames, "task": task, "gt": gt,
                "errors": contains_errors(gt, self.proc_info, task, psr_states(rid, self.labels)),
                "asd": process_asd_predictions(self.asd_dir / f"{rid}_results_pred.csv", frames,
                                               self.cfg["asd_min_conf"][self.asd]),
                "actions": actions, "requires": requires_by_id(self.graph[task], self.cfg["components"])}
        return self._cache[rid]


def load_gt(rid: str, labels: Path) -> list[dict]:
    import csv
    with open(labels / rid / "PSR_labels.csv", encoding="utf-8") as fp:
        return [{"frame": int(r[0][:-4]), "id": int(r[1]), "description": r[2]} for r in csv.reader(fp)]


def run_fused(v: dict, data: Data, action_weight=0.0, action_frames=30, threshold=6.0) -> list[dict]:
    psr = FusedPSR(data.proc_info, v["task"], data.cfg["asd_categories"], INITIAL[v["task"]],
                   threshold=threshold, requires=v["requires"], action_weight=action_weight,
                   action_frames=action_frames)
    for f in range(v["frames"]):
        for sid, conf in v["actions"].get(f, []):
            psr.action_done(sid, conf)
        psr.update(v["asd"][f][1], f)
    return psr.y_hat


def engine_steps(res: dict, components: list[str], proc_info: list, graph_task: dict) -> list[dict]:
    """Every step completion in the engine's log as a PSR prediction; incorrectly installed steps are not
    "correctly completed" and are left out, inferred ones count as implied (conf 0)."""
    bad = {(a["step"], a["t"]) for a in res["alerts"] if a["type"] == "incorrect_install"}
    out = []
    for e in res["log"]:
        if e["status"] != "done" or (e["step"], e["t"]) in bad:
            continue
        st = graph_task["steps"][e["step"]]
        k = components.index(st["component"])
        out.append(make_entry(int(round(e["t"] * FPS)), 3 * k + (0 if st["value"] == 1 else 2), proc_info,
                              conf=0 if e["source"] == "inferred" else 1))
    return sorted(out, key=lambda e: (e["frame"], e["id"]))   # same-frame steps by id, as the official PSR


def summarise(rows: list[dict]) -> dict:
    def mean(key, sel):
        vals = [r[key] for r in rows if sel(r)]
        return sum(vals) / len(vals) if vals else None
    out = {}
    for name, sel in (("all", lambda r: True), ("no_errors", lambda r: not r["errors"]),
                      ("with_errors", lambda r: r["errors"])):
        d = mean("avg_delay", sel)
        out[name] = {"videos": sum(1 for r in rows if sel(r)), "f1": mean("f1", sel), "pos": mean("pos", sel),
                     "delay_s": None if d is None else d / FPS}
    return out


def score(rids: list[str], data: Data, predict) -> dict:
    rows = []
    for rid in rids:
        v = data.video(rid)
        m = determine_performance(v["gt"], predict(rid, v), data.proc_info)
        rows.append({"rid": rid, "errors": v["errors"], **m})
    return {"summary": summarise(rows), "per_video": rows}


def tune(data: Data) -> dict:
    """Grid on the val videos; best mean F1 + POS, ties broken by the shorter delay."""
    results = []
    for w, k, thr in itertools.product(*GRID.values()):
        if w == 0 and k != GRID["action_frames"][0]:
            continue
        s = score(SPLITS["val"], data, lambda rid, v: run_fused(v, data, w, k, thr))["summary"]["all"]
        results.append({"action_weight": w, "action_frames": k, "threshold": thr, **s})
        print(json.dumps(results[-1]), flush=True)
    best = max(results, key=lambda r: (round(r["f1"] + r["pos"], 4), -r["delay_s"]))
    return {"grid": results, "chosen": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asd", choices=list(ASD_DIRS), default="release")
    parser.add_argument("--tune", action="store_true", help="Choose the fusion settings on val (needs --asd ours).")
    parser.add_argument("--runs", nargs="*", default=["pred_asd", "gt_gt"], help="Engine runs to score too.")
    args = parser.parse_args()
    cfg = load_config()
    data = Data(cfg, args.asd)
    out = cfg["paths"]["out"] / "eval"
    settings_path = out / "psr_fusion_settings.json"
    if args.tune:
        res = tune(data)
        write_json(settings_path, res)
        print("chosen on val:", json.dumps(res["chosen"]))
        return
    chosen = json.loads(settings_path.read_text(encoding="utf-8"))["chosen"] if settings_path.is_file() else None
    rids = SPLITS["test"]
    results = {f"official_{args.asd}": {"name": f"Official IndustReal PSR on {args.asd} ASD",
                                        **score(rids, data, lambda rid, v: run_fused(v, data))}}
    if chosen:
        params = {k: chosen[k] for k in GRID}
        results[f"fused_{args.asd}"] = {
            "name": f"Ours: {args.asd} ASD + action stream (fused, settings from val)", "settings": params,
            **score(rids, data, lambda rid, v: run_fused(v, data, **params))}
    names = {"pred_asd": "Ours: earlier live system (TCN actions, ASD)",
             "pred_vjepa_asd": "Ours: engine, online V-JEPA 2 actions + ASD",
             "pred_vjepap_asd": "Ours: live system (TCN + V-JEPA 2 actions, ASD)",
             "pred_vjepap_gt": "Ours: live actions + perfect state",
             "pred_gt": "Ours: engine, predicted actions + perfect state",
             "gt_asd": "Ours: engine, perfect actions + ASD",
             "gt_gt": "Ours: engine, oracle (upper bound)"}
    for run in args.runs:
        def engine_pred(rid, v, run=run):
            res = json.loads((cfg["paths"]["out"] / rid / f"engine_{run}.json").read_text(encoding="utf-8"))
            return engine_steps(res, cfg["components"], data.proc_info, data.graph[v["task"]])
        results[f"engine_{run}"] = {"name": names.get(run, run), **score(rids, data, engine_pred)}
    write_json(out / f"psr_test_{args.asd}.json", results)
    lines = [f"# Procedure step recognition (IndustReal PSR protocol, test set, {args.asd} ASD)", "",
             "| Method | F1 | POS | Delay (s) | With errors: F1 | POS | Delay (s) | No errors: F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for p in PUBLISHED_PSR:
        e = p["with_errors"]
        with_errors = f"*{e['f1']:.3f}* | *{e['pos']:.3f}* | *{e['delay_s']:.1f}*" if e else " |  | "
        lines.append(f"| *{p['name']}* | *{p['f1']:.3f}* | *{p['pos']:.3f}* | *{p['delay_s']:.1f}* | {with_errors} | |")
    for r in results.values():
        s = r["summary"]
        lines.append(f"| {r['name']} | {s['all']['f1']:.3f} | {s['all']['pos']:.3f} | {s['all']['delay_s']:.1f} | "
                     f"{s['with_errors']['f1']:.3f} | {s['with_errors']['pos']:.3f} | "
                     f"{s['with_errors']['delay_s']:.1f} | {s['no_errors']['f1']:.3f} |")
    md = "\n".join(lines) + "\n"
    (out / f"psr_test_{args.asd}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
