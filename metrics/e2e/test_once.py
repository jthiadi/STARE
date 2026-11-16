"""Score each re-tuned v14 once on test with the official evaluator, on its shifted stream (PREREGISTRATION.md, "Test").
Needs work/streams (streams.py) and work/models/<name> (retune.py select). Writes results/test_<name>/ and
results/e2e_summary.json.   python metrics/e2e/test_once.py [tcn] [ens] [oracle] [v9]
Ablations (PREREGISTRATION_ablations.md): "oracle" = annotated actions shifted by v14's p95 (work/models/oracle from
retune.py); "v9" = v9's frozen settings (metrics/psr/model_v9_settings) on work/streams/v9_test, no re-tune.
    python metrics/e2e/test_once.py check-oracle    # D = 0 must reproduce Table 3's oracle row (already reported)"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("E2E_WORK", HERE / "work"))           # outputs (reproduce_all.py points these elsewhere)
RES = Path(os.environ.get("E2E_RESULTS", HERE / "results"))
ROOT = HERE.parents[1]
V14, A, FR = ROOT / "system/psr_v14", ROOT / "system/assistant", ROOT / "system/v14_demo"


M9 = ROOT / "metrics/psr/model_v9_settings"
M13 = ROOT / "metrics/psr/model_v13_settings"


def evaluate(name: str, D_oracle: int | None = None, model: Path | None = None, out: Path | None = None) -> dict:
    out = out or RES / f"test_{name}"
    if name == "oracle":
        args = ["--split", "test", "--ar", "gt"]
    else:
        args = ["--split", "test", "--ar", "predicted", "--ar-dir", str(WORK / "streams" / f"{name}_test")]
    model = model or {"v9": M9, "v13": M13}.get(name, WORK / "models" / name)
    args += ["--model", str(model), "--out", str(out)]
    shift = ""
    if name == "oracle":                                  # OracleARProvider reveals frame t - D at frame t
        shift = (f"sys.path.insert(0, r'{HERE}'); import oracle_shift, psr.ar; "
                 f"oracle_shift.patch(psr.ar.OracleARProvider, {int(D_oracle)}); ")
    code = (f"import sys,runpy; sys.path[:0]=[r'{V14}', r'{FR}']; import frames_compat; {shift}"
            f"sys.argv=[r'eval_v14.py']+{args!r}; runpy.run_path(r'{V14 / 'eval_v14.py'}', run_name='__main__')")
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "TEAMMATE_AR": str(A),
           "INDUSTREAL_DATA": os.environ.get("INDUSTREAL_DATA", str(ROOT / "system/dataset")),
           "PYTHONPATH": os.pathsep.join(x for x in (str(V14), str(A), os.environ.get("PYTHONPATH", "")) if x)}
    r = subprocess.run([sys.executable, "-c", code], cwd=V14, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f"eval failed: {name}")
    s = json.loads((out / f"v14_{'gtAR' if name == 'oracle' else 'predictedAR'}_test.json").read_text())["summary"]
    a, e, m = s["all"], s["with_errors"], s["mistakes"]["all"]
    return {"all": [a["POS"], a["F1"], a["delay_s"]], "with_errors": [e["POS"], e["F1"], e["delay_s"]],
            "mistakes_found": m["tp"], "false_alarms_per_h": m["false_alarms_per_h"]}


def check_oracle() -> None:
    """Implementation check (PREREGISTRATION_ablations.md): the shifted oracle with D = 0 and the frozen v14 must give
    Table 3's oracle row exactly. Written to work/, never to the summary."""
    r = evaluate("oracle", 0, V14 / "model", WORK / "check_oracle_D0")
    exp = {"all": [0.803, 0.913, 5.0], "with_errors": [0.728, 0.858, 5.02], "mistakes_found": 7, "false_alarms_per_h": 2.8}
    ok = all(r[k] == v for k, v in exp.items())
    print(("OK" if ok else "MISMATCH"), json.dumps(r), flush=True)
    if not ok:
        raise SystemExit("shifted oracle with D = 0 does not reproduce Table 3")


def main() -> None:
    if sys.argv[1:] == ["check-oracle"]:
        return check_oracle()
    lat = json.loads((HERE / "results/latency.json").read_text())
    res = {"latency": {k: lat[k] for k in ("gpu", "c_ar_ms_per_frame", "v14_mean_ms", "v14_p95_ms", "ensemble_pass_ms",
                                           "realtime_one_gpu", "L_s", "D_tcn_frames", "D_ens_frames")}}
    summ = RES / "e2e_summary.json"
    old = json.loads(summ.read_text()) if summ.exists() else {}
    for name in (sys.argv[1:] or ["tcn", "ens"]):
        assert name not in old, f"{name} was already scored on test (one look only)"
        if name in ("v9", "v13"):
            l9 = json.loads((HERE / "results/latency_v9.json").read_text())
            res[name] = {"chosen": f"{name} frozen settings (no re-tune)", "D_frames": l9["D_v9_frames"],
                         "v9_p95_ms": l9["v9_p95_ms"], "L_s": l9["L_v9_s"], **evaluate(name)}
        elif name == "oracle":
            sys.path.insert(0, str(HERE))
            from oracle_shift import d_oracle
            D = d_oracle()
            res[name] = {"chosen": json.loads((RES / f"retune_{name}.json").read_text())["chosen"], "D_frames": D,
                         **evaluate(name, D)}
        else:
            res[name] = {"chosen": json.loads((RES / f"retune_{name}.json").read_text())["chosen"], **evaluate(name)}
        print(name, json.dumps(res[name]), flush=True)
    summ.write_text(json.dumps({**old, **res}, indent=1))


if __name__ == "__main__":
    main()
