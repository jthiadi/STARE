"""Step 4 of the on-demand pipeline: frozen v14 run live on one video (GPU), fed by the newest live actions.
Per frame, as demo_v14 / eval_v14: AR top-1 from output/assist/<rid>/ar_vjepa_prefix.jsonl (step 2: TCN + V-JEPA 2),
even frames: localizer + both DINOv2 verifiers, odd frames: official IndustReal ASD -> V14Tracker -> v14 mistake alerts.
Writes output/assist/<rid>/v14_live.json (steps, alerts, timing). The frozen package (PSR_v14/) is only read.
    python -m live.step_v14 <rid>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SYSTEM = Path(__file__).resolve().parents[2]          # system/
DEMO = SYSTEM / "v14_demo"
sys.path.insert(0, str(DEMO))


def main(rid: str, split: str | None = None) -> dict:
    import os
    if split is None:                                     # the recording's own split (test / val / train)
        from assist.industreal import SPLITS
        split = next(s for s in ("test", "val", "train") if rid in SPLITS[s])
    os.environ.setdefault("INDUSTREAL_DATA", str(SYSTEM / "dataset"))
    os.environ.setdefault("TEAMMATE_AR", str(SYSTEM / "assistant"))
    import cv2
    import demo_v14 as D                                  # frozen v14 through the demo's own classes
    from psr.core import frame_paths, procedure_type
    from psr.v11 import mistake_alerts

    out = Path("output/assist") / rid
    rows = [json.loads(l) for l in open(out / "ar_vjepa_prefix.jsonl", encoding="utf-8")]
    paths = frame_paths(split, rid)
    n = len(paths)
    assert len(rows) == n, f"{rid}: {len(rows)} AR rows vs {n} frames"
    vision = D.LiveVision(); vision.reset()
    asd = {}
    tr = D.make_tracker(rid, asd)
    lat, t0 = [], time.time()
    for t in range(n):
        a = time.perf_counter()
        bgr = cv2.imread(str(paths[t]))
        name = rows[t]["top"][0][0]
        names = [] if name == "background" else [name]
        feat = None
        if t % 2 == 0:
            feat, _ = vision(bgr)
        elif D.AC["rate"] == 1 or t % 2 == 1:
            st, _ = vision.asd_state(bgr)
            if st is not None:
                asd[t] = st
        tr.update(t, names, feat)
        lat.append((time.perf_counter() - a) * 1000)
    tr.finish(n - 1)
    alerts = mistake_alerts(rid, tr, n)
    res = {"video": rid, "frames": n, "actions": "ar_vjepa_prefix.jsonl (TCN + V-JEPA 2, live)",
           "predicted_steps": [[e["frame"], e["description"]] for e in tr.y_hat],
           "predicted_steps_err": [[e["frame"], e["description"]] for e in tr.y_hat_err],
           "mistake_alerts": alerts, "asd_states": {str(k): v for k, v in sorted(asd.items())},
           "latency_ms": {"median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                          "mean": round(float(np.mean(lat)), 1)},
           "seconds": round(time.time() - t0, 1)}
    (out / "v14_live.json").write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("video", "frames", "latency_ms", "seconds")} |
                     {"steps": len(res["predicted_steps"]), "alerts": len(alerts)}), flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1])
