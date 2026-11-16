"""The action-end label (75.07 on the paper's protocol) as extra evidence for the mistake rules of engine + v9.

The engine never saw the action-end label. Here it gets each label when the GPU has finished it (end of the action plus
the queue at 1.34 s per action, the shared-GPU speed), and ``Engine.on_final_label`` may then:

  install  check a labelled install against v9's state: expected state not seen within 30 s -> possible incorrect install
  skip     a later-subgoal step was just performed while an earlier one is not done -> skipped (after a grace period)

It only adds alerts: every run checks that no step moves, so step F1, POS and delay stay those of engine + v9. A second,
alert-only setting is also tried: a higher ``low_conf`` (actions below it only give "cannot verify"), to cut false alarms.

Rule, fixed before any result. The baseline is engine + v9 with the same live actions, on val. A setting passes if

  (a) it finds at least 2 more mistakes, with no more new false alarms than new finds, or
  (b) it finds no fewer mistakes, with at least a quarter fewer false alarms.

Of the passing settings the one with the most finds, then the fewest false alarms, is chosen. If none passes on val, test
is not scored. Val is optimistic: v9 was trained on train + val, and the action-end recipe was chosen on val's true
segments. Val has 13 mistakes, so it is a small check. Test is scored once and only test decides. This is one more look
at a test set already used for other experiments.

    python -m assist.extract.run_action_end --spans --videos <16 val videos> --plan spans_val.json
    python -m assist.extract.run_action_end --score --plan spans_val.json --part 0/2      # GPU (and --part 1/2)
    python -m assist.extract.run_action_end --write --videos <16 val videos>
    python -m scripts.mistakes.end_label_check --states       # v9's saved val steps -> <video>/asd_v9.jsonl
    python -m scripts.mistakes.end_label_check --select       # CPU: val grid + rule -> output/assist/end_label_check/select_val.json
    python -m scripts.mistakes.end_label_check --test         # CPU: once, only if val passed -> test.json
"""
from __future__ import annotations

import argparse
import itertools
import json

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.extract.run_action_end import label_delays
from monitor.settings import ROOT

OUT = ROOT / "output/assist/end_label_check"
ACTIONS, STATE = "pred_vjepap", "asdv9"
SERVICE_S = 1.3436                       # shared-GPU seconds per action (benchmark_results.json)
MIN_NEW_FINDS, FEWER_FALSE_ALARMS = 2, 0.25
RULES = [None] + [{"install": install, "skip_grace_s": grace, "min_conf": conf}
                  for install, grace, conf in itertools.product((True, False), (None, 10, 30), (0.3, 0.5, 0.7))
                  if install or grace is not None]
SETTINGS = [{"final_label": rule, "low_conf": low} for low in (0.45, 0.6) for rule in RULES]


def ready_rows(rows: list[dict], service_s: float = SERVICE_S) -> list[tuple[float, str, float]]:
    """(time the label is ready, label, confidence): actions are labelled one at a time, in the order they ended."""
    rows = sorted(rows, key=lambda r: r["ended_at"])
    delays = label_delays([r["ended_at"] for r in rows], service_s)
    return [(round(r["ended_at"] + d, 6), r["label"], r["conf"]) for r, d in zip(rows, delays)]


def passes(base: dict, row: dict) -> bool:
    new_finds = row["found"] - base["found"]
    new_false_alarms = row["false_alarms"] - base["false_alarms"]
    more = new_finds >= MIN_NEW_FINDS and new_false_alarms <= new_finds
    cleaner = new_finds >= 0 and row["false_alarms"] <= (1 - FEWER_FALSE_ALARMS) * base["false_alarms"]
    return more or cleaner


def choose(base: dict, rows: list[dict]) -> dict | None:
    passed = [r for r in rows if passes(base, r)]
    return max(passed, key=lambda r: (r["found"], -r["false_alarms"])) if passed else None


# ---------------------------------------------------------------------------------------------- runs (CPU)
def write_states(split: str = "val") -> None:
    """v9's saved steps as a state stream, exactly as ``compare_v9`` does for test."""
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data
    from assist.industreal import SPLITS, n_frames, task_of
    from scripts.mistakes.compare_v9 import RESULTS, state_rows

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    proc_info = Data(cfg, "weights").proc_info
    saved = json.loads((RESULTS / f"v9_predictedAR_{split}.json").read_text(encoding="utf-8"))
    steps = {v["video"]: v["predicted_steps"] for v in saved["videos"]}
    if set(steps) != set(SPLITS[split]):
        raise ValueError(f"v9's {split} videos are not the {len(SPLITS[split])} {split} videos")
    for rid in SPLITS[split]:
        out_dir = cfg["paths"]["out"] / rid
        write_jsonl(out_dir / "asd_v9.jsonl", state_rows(
            steps[rid], list(tasks[task_of(rid)].initial_state), proc_info,
            n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"]), read_jsonl(out_dir / "asd.jsonl")))
        print(json.dumps({"states": rid}), flush=True)


def _scored(name: str, split: str, results: dict, cfg: dict, tasks: dict, data) -> dict:
    from assist.evaluation.eval import evaluate
    from assist.evaluation.psr_eval import engine_steps, score
    from assist.industreal import SPLITS

    rids = SPLITS[split]
    for rid in rids:
        write_json(cfg["paths"]["out"] / rid / f"engine_{name}.json", results[rid])
    r = evaluate(name, rids, cfg, tasks)
    rows = r["rows"]
    psr = score(rids, data, lambda rid, v: engine_steps(results[rid], cfg["components"], data.proc_info,
                                                        data.graph[v["task"]]))["summary"]["all"]
    return {"found": rows["all"]["tp"], "mistakes": rows["all"]["gt"], "alerts": rows["all"]["alerts"],
            "false_alarms": rows["all"]["alerts"] - rows["all"]["tp"],
            "precision": None if rows["all"]["precision"] is None else round(rows["all"]["precision"], 3),
            "false_alarms_per_h": round(rows["all"]["false_alarms_per_h"], 1),
            "early_warnings_per_h": round(r["early_warnings_per_h"], 1),
            "by_type": {k: {"found": rows[k]["tp"], "of": rows[k]["gt"], "alerts": rows[k]["alerts"]}
                        for k in ("skipped", "order", "install")},
            "final_label_alerts": sum(a.get("source") == "final_label" for res in results.values() for a in res["alerts"]),
            "psr_f1": round(psr["f1"], 3), "psr_pos": round(psr["pos"], 3), "psr_delay_s": round(psr["delay_s"], 1)}


def _runs(split: str, setting: dict | None, cfg: dict, tasks: dict, plain: dict | None = None) -> dict:
    from assist.industreal import SPLITS
    from assist.run_engine import run
    from scripts.mistakes.v9_checks import same_steps

    out = {}
    for rid in SPLITS[split]:
        if setting is None:
            out[rid] = run(rid, ACTIONS, STATE, cfg, tasks)
            continue
        run_cfg = {**cfg, "engine": {**cfg["engine"], "low_conf": setting["low_conf"]}}
        rule = setting["final_label"]
        final = (ready_rows(read_jsonl(cfg["paths"]["out"] / rid / "action_end.jsonl")), rule) if rule else None
        out[rid] = run(rid, ACTIONS, STATE, run_cfg, tasks, final=final)
        if plain is not None and not same_steps(plain[rid]["log"], out[rid]["log"]):
            raise RuntimeError(f"{rid}: the check moved a step")
    return out


def select() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "weights")
    plain = _runs("val", None, cfg, tasks)
    base = _scored(f"{ACTIONS}_{STATE}_endbase", "val", plain, cfg, tasks, data)
    print(json.dumps({"baseline": base}), flush=True)
    grid = []
    for setting in SETTINGS:
        results = _runs("val", setting, cfg, tasks, plain)
        grid.append({**setting, **_scored(f"{ACTIONS}_{STATE}_endcheck", "val", results, cfg, tasks, data)})
        print(json.dumps(grid[-1]), flush=True)
    best = choose(base, grid)
    report = {"rule": "(a) at least 2 more finds, no more new false alarms than new finds, or (b) no fewer finds and at "
                      "least a quarter fewer false alarms; steps never move",
              "note": "v9 was trained on train + val and the action-end recipe was chosen on val: optimistic, chooses only",
              "baseline_val": base, "passed": best is not None, "chosen": best, "grid": grid}
    write_json(OUT / "select_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=1))


def test() -> None:
    from assist.engine.graph import load_tasks
    from assist.evaluation.psr_eval import Data

    result_path = OUT / "test.json"
    if result_path.is_file():
        raise SystemExit(f"{result_path} exists: test was already scored once")
    report = json.loads((OUT / "select_val.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise SystemExit("no setting passed on val: test is not scored")
    setting = {k: report["chosen"][k] for k in ("final_label", "low_conf")}
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    data = Data(cfg, "release")
    plain = _runs("test", None, cfg, tasks)
    base = _scored(f"{ACTIONS}_{STATE}_endbase", "test", plain, cfg, tasks, data)
    check = _scored(f"{ACTIONS}_{STATE}_endcheck", "test", _runs("test", setting, cfg, tasks, plain), cfg, tasks, data)
    result = {"chosen_on_val": setting, "baseline_test": base, "endcheck_test": check, "passed": passes(base, check)}
    write_json(result_path, result)
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("states", "select", "test"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.states:
        write_states()
    if args.select:
        select()
    if args.test:
        test()


if __name__ == "__main__":
    main()
