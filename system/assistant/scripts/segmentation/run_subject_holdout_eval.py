"""Leave-subjects-out evaluation of the fine-tuned DINOv2 + TCN pipeline on every recording.

Subjects are split into groups; for each group the backbones (cross-fitted by
subject) and the TCN are trained only on the other subjects, then the group's
recordings are scored against their AR GT. Resumable: finished steps are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from monitor.settings import ROOT

GROUPS = [["03", "08"], ["09", "10"], ["12", "13"], ["17", "18"], ["19", "23"]]


MIN_AVAILABLE_RAM_GB = 0.35
MIN_AVAILABLE_COMMIT_GB = 1.0
RESUME_AVAILABLE_RAM_GB = 1.5
RESUME_AVAILABLE_COMMIT_GB = 3.5
MAX_RETRIES = 20


def _memory_gb() -> tuple[float, float]:
    """(available physical RAM, available commit = RAM + page file headroom) in GB."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30, status.ullAvailPageFile / 2**30


def _memory_low() -> bool:
    ram, commit = _memory_gb()
    return ram < MIN_AVAILABLE_RAM_GB or commit < MIN_AVAILABLE_COMMIT_GB


class LowMemory(RuntimeError):
    pass


def _run(cmd: list[str], log: Path) -> None:
    """Run a step; after a low-memory stop, wait for RAM to recover and retry (steps resume)."""
    import psutil

    for attempt in range(MAX_RETRIES):
        try:
            return _run_once(cmd, log)
        except LowMemory as exc:
            print(json.dumps({"low_memory_stop": cmd[3], "attempt": attempt + 1, "detail": str(exc)}), flush=True)
            while _memory_gb()[0] < RESUME_AVAILABLE_RAM_GB or _memory_gb()[1] < RESUME_AVAILABLE_COMMIT_GB:
                time.sleep(30)
            time.sleep(30)
    raise SystemExit(f"Gave up on {cmd[3]} after {MAX_RETRIES} low-memory stops")


def _run_once(cmd: list[str], log: Path) -> None:
    """Run a step; kill it if system RAM gets dangerously low (keeps the laptop stable)."""
    import psutil

    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if _memory_low():
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                ram, commit = _memory_gb()
                raise LowMemory(f"Stopped {cmd[3]}: RAM {ram:.2f} GB, commit {commit:.2f} GB available")
            time.sleep(5)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/subject_holdout_eval")
    args = parser.parse_args()
    py = sys.executable
    pool = {s for g in GROUPS for s in g}
    recs = sorted(p.name for p in (ROOT / "data/evaluation/official").iterdir()
                  if (p / "AR_labels.csv").is_file() and p.name.split("_", 1)[0] in pool)
    args.output.mkdir(parents=True, exist_ok=True)
    for g, subjects in enumerate(GROUPS):
        tests = [r for r in recs if r.split("_", 1)[0] in subjects]
        gdir = args.output / f"group{g}_{'_'.join(subjects)}"
        gdir.mkdir(parents=True, exist_ok=True)
        log = gdir / "log.txt"
        ft = gdir / "finetune"
        for fold in (0, 1):
            if (ft / f"fold{fold}.extracted").is_file():
                continue
            resume = ["--extract-only"] if (ft / f"fold{fold}.trained").is_file() else []
            _run([py, "-u", "-m", "scripts.features.finetune_dino_ar", "--fold", str(fold),
                  "--test-recordings", *tests, "--subject-split", "--extract-needed",
                  "--batch", "32", "--lr", "4e-5", "--workers", "0", "--output", str(ft), *resume], log)
        tcn = gdir / "tcn"
        if not (tcn / "ar_object_paper_metrics.json").is_file():
            _run([py, "-u", "-m", "scripts.segmentation.train_ar_object_tcn", "--test-recordings", *tests,
                  "--train-recordings", *[r for r in recs if r not in tests],
                  "--extra-features", str(ROOT / "output/ar_frame_features_hires"),
                  "--crossfit", str(ft / "feats_fold0"), str(ft / "feats_fold1"),
                  "--crossfit-meta", str(ft / "meta.json"),
                  "--verb-aux", "--progress", "--cosine", "--warp", "--epochs", "80", "--seeds", "3",
                  "--hidden", "192", "--layers", "10", "--stages", "4",
                  "--folds", "4", "--subject-folds", "--output", str(tcn)], log)
        print(json.dumps({"group": g, "subjects": subjects, "done": True}), flush=True)
        summarize(args.output)


def summarize(root: Path) -> None:
    rows = []
    for metrics in sorted(root.glob("group*/tcn/ar_object_paper_metrics.json")):
        report = json.loads(metrics.read_text(encoding="utf-8"))
        for rid, res in report["per_test"].items():
            rows.append({
                "recording": rid,
                "group": metrics.parent.parent.name,
                "all_labeled": round(res["always_predict"]["all_labeled"], 2),
                "covered_gated": round(res["background_gated"]["covered"], 2),
                "segment_top1": round(res["always_predict"]["segment_top1"], 2),
                "labeled_ok": res["labeled_ok"],
                "labeled_total": res["labeled_total"],
                "group_oof_acc": report["oof_acc"],
            })
    if not rows:
        return
    ok = sum(r["labeled_ok"] for r in rows)
    tot = sum(r["labeled_total"] for r in rows)
    summary = {
        "videos": len(rows),
        "pooled_all_labeled_pct": round(100 * ok / tot, 2),
        "mean_video_all_labeled_pct": round(sum(r["all_labeled"] for r in rows) / len(rows), 2),
        "mean_video_segment_top1_pct": round(sum(r["segment_top1"] for r in rows) / len(rows), 2),
        "min_video": min(rows, key=lambda r: r["all_labeled"]),
        "max_video": max(rows, key=lambda r: r["all_labeled"]),
        "per_video": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}), flush=True)


if __name__ == "__main__":
    main()
