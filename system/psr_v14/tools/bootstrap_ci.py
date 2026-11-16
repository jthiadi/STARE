"""95 % bootstrap intervals over recordings for POS / F1 / delay, from an eval_v13.py output file.

The official metrics are means over recordings, so recordings are resampled with replacement (10,000 resamples,
percentile interval).  Reads only the per-recording metrics stored in the result file.

    python tools/bootstrap_ci.py results/v13_predictedAR_test.json
"""
import json
import sys

import numpy as np


def ci(x, reps=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows = json.load(open(sys.argv[1]))["videos"]
    out = {}
    for name, keep in [("all", lambda r: True), ("with_errors", lambda r: r["errors"]),
                       ("without_errors", lambda r: not r["errors"])]:
        rs = [r for r in rows if keep(r)]
        out[name] = {"n": len(rs)}
        for k, f in [("POS", "pos"), ("F1", "f1"), ("delay_s", "delay_frames")]:
            vals = [r[f] / 10 if k == "delay_s" else r[f] for r in rs]
            m, lo, hi = ci(vals)
            out[name][k] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
