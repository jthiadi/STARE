"""IndustReal assembly steps from PSR state changes (install / remove), with mistake labels.

A step is a component whose PSR value changes. ``mistake=True`` means the new value is -1
(installed incorrectly). Most IndustReal mistakes are named \"Incorrectly installed …\" in
``PSR_labels_with_errors.csv``; a few are normal install names with state -1 (order / wrong timing).
"""
from __future__ import annotations

from pathlib import Path

from monitor.settings import ROOT
from scripts.data.ar_labels import load_ar_intervals
from scripts.mistakes.order_check import has_mistake, names, steps
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

LABELS = ROOT / "data/evaluation/official"
FPS = 10.0


def install_events(vids: list[str] | None = None) -> list[dict]:
    """One dict per install step: component, frame, mistake, PSR error name, nearby AR actions."""
    vids = vids if vids is not None else TRAIN + VAL + TEST
    out = []
    for vid in vids:
        label = names(vid)
        ar = load_ar_intervals(LABELS / vid / "AR_labels.csv", source_fps=FPS)
        for s in steps(vid):
            if s["kind"] != "install":
                continue
            t = s["frame"] / FPS
            active = [r["action"] for r in ar if r["start_s"] - 1e-6 <= t <= r["end_s"] + 1e-6]
            before = [r["action"] for r in ar if r["end_s"] <= t][-3:]
            after = [r["action"] for r in ar if r["start_s"] >= t][:3]
            name = label.get(s["frame"], f"comp {s['comp']}")
            out.append({
                "vid": vid,
                "frame": s["frame"],
                "comp": s["comp"],
                "mistake": s["mistake"],
                "name": name,
                "incorrect_named": name.lower().startswith("incorrect"),
                "ar_active": active,
                "ar_before": before,
                "ar_after": after,
                "before_state": s["before"],
            })
    return out


def split_of(vid: str) -> str:
    if vid in TRAIN:
        return "train"
    if vid in VAL:
        return "val"
    return "test"


def taxonomy(events: list[dict]) -> dict:
    """Counts by split: incorrect-named vs other state=-1 installs."""
    report = {}
    for split in ("train", "val", "test"):
        rows = [e for e in events if split_of(e["vid"]) == split]
        mist = [e for e in rows if e["mistake"]]
        report[split] = {
            "installs": len(rows),
            "mistakes": len(mist),
            "incorrect_named": sum(e["incorrect_named"] for e in mist),
            "other_state_neg1": sum(not e["incorrect_named"] for e in mist),
            "videos_with_mistakes": len({e["vid"] for e in mist}),
            "ar_at_mistake": _count_tuples(tuple(e["ar_active"]) or ("<none>",) for e in mist),
        }
    report["note"] = (
        "Order precondition checks only cover a few mistakes; most are incorrect installations "
        "logged while the worker is tightening (often tighten_nut / tighten_acorn_nut)."
    )
    report["train_videos_error_free"] = sum(not has_mistake(v) for v in TRAIN)
    return report


def _count_tuples(items) -> dict[str, int]:
    from collections import Counter
    return {str(k): int(v) for k, v in Counter(items).most_common()}
