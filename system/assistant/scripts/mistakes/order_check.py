"""Which IndustReal mistakes can a procedure-order check catch, given a perfect PSR state?

``PSR_labels_raw.csv`` gives the assembly state over time: one column per component, 1 = installed,
0 = absent, -1 = installed incorrectly. Each state change is a step. From the error-free training videos we
learn, per component and direction (install / remove), which components are *always* present (install) or
*always* absent (remove) at that moment: the step's preconditions. A step whose preconditions are not met in
the current state is flagged as an order violation.

Reported on val and test: how many real mistakes (state -1) are flagged, how many order violations the
ground truth contains, and how many correct steps raise a false alarm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from monitor.settings import ROOT
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

LABELS = ROOT / "data/evaluation/official"
OUT = ROOT / "output/mistakes"


def states(vid: str) -> list[tuple[int, tuple[int, ...]]]:
    with (LABELS / vid / "PSR_labels_raw.csv").open(newline="") as handle:
        return [(int(Path(r[0]).stem), tuple(int(x) for x in r[1:])) for r in csv.reader(handle) if r]


def names(vid: str) -> dict[int, str]:
    with (LABELS / vid / "PSR_labels_with_errors.csv").open(newline="") as handle:
        return {int(Path(r[0]).stem): r[2] for r in csv.reader(handle) if r}


def steps(vid: str) -> list[dict]:
    """State changes as steps: component, direction, the state before the change, and the mistake flag."""
    seq, out = states(vid), []
    for (_, before), (frame, after) in zip(seq, seq[1:]):
        for comp, (a, b) in enumerate(zip(before, after)):
            if a == b:
                continue
            kind = "install" if b != 0 else "remove"
            out.append({"vid": vid, "frame": frame, "comp": comp, "kind": kind, "mistake": b == -1,
                        "before": before})
    return out


def has_mistake(vid: str) -> bool:
    return any(s["mistake"] for s in steps(vid))


def learn(vids: list[str]) -> dict[tuple[int, str], dict[str, frozenset]]:
    """Preconditions per (component, direction): components always present / always absent beforehand."""
    rules: dict[tuple[int, str], dict[str, frozenset]] = {}
    for vid in vids:
        for s in steps(vid):
            present = frozenset(i for i, x in enumerate(s["before"]) if x != 0 and i != s["comp"])
            absent = frozenset(i for i, x in enumerate(s["before"]) if x == 0 and i != s["comp"])
            key = (s["comp"], s["kind"])
            if key not in rules:
                rules[key] = {"present": present, "absent": absent}
            else:
                rules[key] = {"present": rules[key]["present"] & present, "absent": rules[key]["absent"] & absent}
    return rules


def violation(rules, s) -> str | None:
    rule = rules.get((s["comp"], s["kind"]))
    if rule is None:
        return "unseen step"
    missing = sorted(i for i in rule["present"] if s["before"][i] == 0)
    blocking = sorted(i for i in rule["absent"] if s["before"][i] != 0)
    if missing or blocking:
        return f"needs {missing} present, {blocking} absent"
    return None


def evaluate(rules, vids: list[str]) -> dict:
    mistakes = flagged_mistakes = correct = false_alarms = gt_violations = 0
    details = []
    for vid in vids:
        label = names(vid)
        for s in steps(vid):
            why = violation(rules, s)
            if s["mistake"]:
                mistakes += 1
                flagged_mistakes += why is not None
            else:
                correct += 1
                false_alarms += why is not None
                gt_violations += why is not None
            if s["mistake"] or why:
                details.append({"vid": vid, "frame": s["frame"], "step": label.get(s["frame"], f"comp {s['comp']}"),
                                "component": s["comp"], "kind": s["kind"], "mistake": s["mistake"],
                                "flagged": why})
    return {"videos": len(vids), "videos_with_mistakes": sum(has_mistake(v) for v in vids),
            "mistakes": mistakes, "mistakes_flagged_by_order": flagged_mistakes,
            "correct_steps": correct, "correct_steps_flagged": false_alarms,
            "false_alarms_per_video": round(false_alarms / max(len(vids), 1), 2), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learn-from", choices=["train", "train_all"], default="train",
                        help="train: error-free training videos only; train_all: all training videos")
    args = parser.parse_args()
    learn_vids = TRAIN if args.learn_from == "train_all" else [v for v in TRAIN if not has_mistake(v)]
    rules = learn(learn_vids)
    report = {"learned_from": len(learn_vids), "rules": len(rules),
              "train_mistakes": sum(s["mistake"] for v in TRAIN for s in steps(v)),
              "val": evaluate(rules, VAL), "test": evaluate(rules, TEST)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"order_check_{args.learn_from}.json").write_text(json.dumps(report, indent=1, default=list))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "details"} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
