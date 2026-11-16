"""Recompute every number in the paper's result tables (Tables 1-8) on a CPU and check each against the paper.

Needs the IndustReal annotations (and its .mp4 videos or JPEG frames, for frame counts), the MECCANO annotations, the saved
model outputs (README.md, "Reproduce the paper's numbers") and Python with system/assistant/requirements.txt. No GPU and no
model weights: every model output is read from system/assistant/output/, metrics/*/work/ and third_party/PROVIA/runs/.

    python metrics/reproduce_all.py --data <IndustReal root: train/, validation/, test/> [--videos <folder with <rid>.mp4>]

Writes metrics/reproduced/ and prints OK / MISMATCH per number; exits 1 on any mismatch.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "system" / "assistant"
OUT = ROOT / "metrics" / "reproduced"
PY = sys.executable

# POS / F1 / delay (all recordings), the same on recordings with errors, typed mistakes found / 32, false alarms per h
EXPECTED = {
    "T1 AR official / causal TCN": 55.74,
    "T1 AR official / fine-tuned V-JEPA 2 (hand crop, single run)": 70.20,
    "T1 AR official / final label (ensemble)": 75.07,
    "T1 AR official / final label, val (fixed before test)": 73.81,
    "T1 AR live / causal TCN": 53.72,
    "T1 AR live / final label (ensemble)": 55.52,
    "T2 final label parts (VideoMAE, 3 V-JEPA 2, +VideoMAE, +TCN, +prior)": (62.18, 72.76, 73.25, 73.74, 75.07),
    "T3 MECCANO AR / final label (val, test)": (66.12, 62.45),
    "T4/T6/T7 PSR end to end / STARE (causal TCN)": (0.819, 0.907, 6.53, 0.747, 0.839, 7.54, 9, 3.6),
    "T5 MECCANO PSR / STARE": (0.615, 0.609, 26.6),
    "T6 PSR end to end / w/o AR (vision only)": (0.772, 0.893, 7.89, 0.689, 0.815, 7.59, 8, 3.2),
    "T6 PSR end to end / w/o state detector": (0.809, 0.903, 7.74, 0.747, 0.839, 9.43, 7, 3.2),
    "T7 PSR end to end / final label (ensemble)": (0.764, 0.879, 7.18, 0.661, 0.797, 8.01, 9, 5.19),
    "T7 PSR end to end / oracle AR": (0.814, 0.915, 4.98, 0.728, 0.858, 5.02, 7, 2.8),
    "T7 AR delay in frames (causal TCN, ensemble, oracle)": (8, 25, 1),
    "T8 untyped mistakes / STARE": (8, 0.667, 1.6),
    "T8 untyped mistakes / PROVIA mean": (3.333, 0.194, 5.46),
    "T8 untyped mistakes / Causal-TCN mean": (3.0, 0.228, 4.0),
    "T8 untyped mistakes / Controlled PREGO mean": (1.667, 0.155, 3.6),
}
got, bad = {}, []


def check(name, value):
    got[name] = value
    exp = EXPECTED[name]
    ok = value == exp
    print(f"{'OK      ' if ok else 'MISMATCH'} {name}: {value}" + ("" if ok else f"  (expected {exp})"), flush=True)
    if not ok:
        bad.append(name)


def missing(what):
    print(f"MISSING  {what}", flush=True)
    bad.append(what)


def run(cmd, cwd, env):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f"failed: {' '.join(map(str, cmd))}")
    return r.stdout


def grid(script, args, n, env):
    """Run a validation grid split over n CPU processes."""
    procs = [subprocess.Popen([PY, script, *args, str(i), str(n)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True) for i in range(n)]
    for p in procs:
        err = p.communicate()[1]
        if p.returncode:
            print(err[-3000:]); raise SystemExit(f"grid failed: {script} {args}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--videos", default=None)
    a = ap.parse_args()
    data = Path(a.data).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    # the assistant reads annotations from data/evaluation/official/<rid>/ (its layout): copy them from the dataset
    lab = A / "data" / "evaluation" / "official"
    for split, sub in (("train", "train_annotations"), ("validation", "val_annotations"), ("test", "test_annotations")):
        for d in (data / split / sub).iterdir():
            (lab / d.name).mkdir(parents=True, exist_ok=True)
            for f in ("AR_labels.csv", "PSR_labels.csv", "PSR_labels_raw.csv", "PSR_labels_with_errors.csv"):
                if (d / f).exists():
                    shutil.copyfile(d / f, lab / d.name / f)
    base = os.environ.get("PYTHONPATH", "")                       # keep the caller's package paths (e.g. a venv)
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "INDUSTREAL_DATA": str(data), "TEAMMATE_AR": str(A),
           "PYTHONPATH": os.pathsep.join(x for x in (str(A), base) if x)}
    if a.videos:
        env["INDUSTREAL_VIDEOS"] = str(Path(a.videos).resolve())
    ar = ROOT / "metrics/ar"

    # ------------------------------------------------ Table 1: action recognition on IndustReal (official and live)
    run([PY, ar / "score_live_models_paper_protocol.py"], A, env)
    r = json.loads((ar / "results/live_models_paper_protocol.json").read_text())
    check("T1 AR official / causal TCN", r["test"]["TCN alone"])
    # "<run>  test segs <n>  val <v>  test <t>  recorded {...}", one line per saved V-JEPA 2 run
    lines = run([PY, ar / "score_vjepa_paper_protocol.py"], A, env).splitlines()
    single = [l.split() for l in lines if l.split()[:1] == ["vjft_hand_k12"]]
    check("T1 AR official / fine-tuned V-JEPA 2 (hand crop, single run)",
          float(single[0][single[0].index("test", 2) + 1]) if single else None)
    run([PY, ar / "score_final_labels_paper_protocol.py"], A, env)
    r = json.loads((ar / "results/final_labels_paper_protocol.json").read_text())
    check("T1 AR official / final label (ensemble)", r["test"]["top1"])
    check("T1 AR official / final label, val (fixed before test)", r["val"]["top1"])
    run([PY, ar / "score_live_condition.py"], A, env)
    r = json.loads((ar / "results/live_condition_test.json").read_text())["true_actions_labelled_right_pct"]
    check("T1 AR live / causal TCN", r["TCN"])
    check("T1 AR live / final label (ensemble)", r["Final labels (V-JEPA 2 ensemble + VideoMAE + TCN)"])

    # ------------------------------------------------ Table 2: what each part of the final label adds
    run([PY, ar / "score_final_label_breakdown.py"], A, env)
    r = json.loads((ar / "results/final_label_breakdown.json").read_text())["test"]
    check("T2 final label parts (VideoMAE, 3 V-JEPA 2, +VideoMAE, +TCN, +prior)",
          (r["VideoMAE alone"], r["Mean of 3 V-JEPA 2 runs"], r["+ VideoMAE (mean of 4)"], r["+ TCN (w = 0.95)"],
           r["+ order prior (final label)"]))

    # ------------------------------------------------ Table 3: action recognition on MECCANO
    if (A / "output/meccano_action_end/vmae_probs.npz").exists():
        run([PY, ar / "score_meccano_final_labels.py"], A, env)
        r = json.loads((ar / "results/meccano_final_labels.json").read_text())
        check("T3 MECCANO AR / final label (val, test)", (r["val"], r["test"]))
    else:
        missing("MECCANO action recognition outputs (system/assistant/output/meccano_action_end)")

    # ------------------------------------------------ Tables 4, 6, 7: step recognition end to end (metrics/e2e)
    # Each action label reaches the tracker only after its measured compute (results/latency*.json); the tracker's own
    # settings are re-tuned on validation per input, then test is scored once. Written under metrics/reproduced/e2e.
    e2e, ed = ROOT / "metrics/e2e", OUT / "e2e"
    shutil.rmtree(ed, ignore_errors=True)
    eenv = {**env, "E2E_WORK": str(ed / "work"), "E2E_RESULTS": str(ed / "results"), "INDUSTREAL_DATA": str(data)}
    run([PY, e2e / "streams.py", "tcn", "ens", "v13"], ROOT, eenv)
    ns = max(1, min(10, (os.cpu_count() or 2) - 1))
    for name in ("tcn", "ens", "oracle"):
        grid(e2e / "retune.py", ["grid", name], ns, eenv)
        run([PY, e2e / "retune.py", "select", name], ROOT, eenv)
    run([PY, e2e / "test_once.py", "check-oracle"], ROOT, eenv)     # implementation check: shift 0 = the unshifted oracle
    run([PY, e2e / "test_once.py", "tcn", "ens", "oracle", "v13"], ROOT, eenv)
    grid(e2e / "noar.py", ["grid"], 6, eenv)                       # w/o AR (PREREGISTRATION_noar.md)
    run([PY, e2e / "noar.py", "select"], ROOT, eenv)
    run([PY, e2e / "noar.py", "test"], ROOT, eenv)
    s = json.loads((ed / "results/e2e_summary.json").read_text())
    row = lambda r: (*r["all"], *r["with_errors"], r["mistakes_found"], r["false_alarms_per_h"])
    check("T4/T6/T7 PSR end to end / STARE (causal TCN)", row(s["tcn"]))
    check("T6 PSR end to end / w/o AR (vision only)", row(s["noar"]))
    check("T6 PSR end to end / w/o state detector", row(s["v13"]))
    check("T7 PSR end to end / final label (ensemble)", row(s["ens"]))
    check("T7 PSR end to end / oracle AR", row(s["oracle"]))
    check("T7 AR delay in frames (causal TCN, ensemble, oracle)",
          (s["latency"]["D_tcn_frames"], s["latency"]["D_ens_frames"], s["oracle"]["D_frames"]))

    # ------------------------------------------------ Table 5: step recognition on MECCANO (metrics/meccano)
    # The tracker re-run on the saved model outputs (work/ar, work/vprobs, work/steps), scored with STORM-PSR's evaluator.
    mec = ROOT / "metrics/meccano"
    if (mec / "work/vprobs").is_dir():
        md = OUT / "meccano"                       # settings selected on MECCANO validation (results/retune.json)
        run([PY, mec / "stare_lite.py", "run"], ROOT, {**env, "MECCANO_RESULTS": str(md)})
        r = json.loads((md / "meccano_psr.json").read_text())
        check("T5 MECCANO PSR / STARE", (r["test"]["POS"], r["test"]["F1"], r["test"]["delay_s"]))
    else:
        missing("MECCANO step recognition outputs (metrics/meccano/work)")

    # ------------------------------------------------ Table 8: mistake detection without the type, vs PROVIA
    if (ROOT / "third_party/PROVIA/runs/industreal/model_seed2/test_candidates.npz").exists():
        run([PY, ROOT / "metrics/provia/untyped.py"], ROOT, {**env, "PROVIA_RESULTS": str(OUT / "provia")})
        r = json.loads((OUT / "provia/untyped.json").read_text())
        st = r["STARE"]["test"]
        check("T8 untyped mistakes / STARE", (st["found"], st["precision"], st["false_alarms_per_h"]))
        for m in ("PROVIA", "Causal-TCN", "Controlled PREGO"):
            check(f"T8 untyped mistakes / {m} mean",
                  tuple(r[m][f"test_{k}_mean_sd"][0] for k in ("found", "precision", "false_alarms_per_h")))
    else:
        missing("PROVIA outputs (third_party/PROVIA/runs)")

    (OUT / "summary.json").write_text(json.dumps({"got": got, "mismatches": bad}, indent=1))
    print("\nALL NUMBERS REPRODUCED" if not bad else f"\n{len(bad)} MISMATCH(ES): {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
