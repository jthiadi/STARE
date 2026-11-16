"""Measured per-frame compute of v9 (PREREGISTRATION_ablations.md): the v14 live step (live/step_v14.py) with the ASD
branch removed and v9's settings (metrics/psr/model_v9_settings): localizer + both verifiers on even frames, the state
filter every frame. Same GPU, recordings and AR rows as latency.py. Writes results/latency_v9.json.
Run from STARE with env.sh sourced:  CUDA_VISIBLE_DEVICES=0 python metrics/e2e/latency_v9.py"""
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A, SYS = ROOT / "system/assistant", ROOT / "system"
M9 = ROOT / "metrics/psr/model_v9_settings"
os.chdir(A)
sys.path[:0] = [str(A), str(SYS / "v14_demo")]
os.environ.setdefault("INDUSTREAL_DATA", str(SYS / "dataset"))
os.environ.setdefault("TEAMMATE_AR", str(A))
FPS = 10.0


def v9(rid: str, D) -> dict:
    import cv2
    from psr.blend import Blend
    from psr.core import frame_paths, procedure_type
    from psr.filter_tracker import FilterConfig
    from psr.lr_model import LRGroupModel
    from psr.tracker import GroupModel, TrackerConfig
    from psr.v14 import V14Tracker
    cfg = json.loads((M9 / "config.json").read_text()); sm = cfg["step_model"]
    model = Blend(GroupModel(M9 / "tables.json"), LRGroupModel(path=M9 / sm["lr_path"]), sm["w_lr"])
    tr = V14Tracker(procedure_type(rid), FilterConfig(**cfg["filter"]), model,
                    TrackerConfig(w_action=sm["w_action"], w_since=sm["w_since"], null_bias=sm["null_bias"]),
                    d0=cfg["v11"]["d0"], asd={}, w=0.0, eps=cfg["asd"]["eps"], rate=cfg["asd"]["rate"])
    rows = [json.loads(l) for l in open(A / "output/assist" / rid / "ar_vjepa_prefix.jsonl", encoding="utf-8")]
    paths = frame_paths("test", rid)
    n = len(paths); assert len(rows) == n
    vision = D.LiveVision(); vision.reset()
    lat, t0 = [], time.time()
    for t in range(n):                                   # as live/step_v14.py, without the odd-frame ASD
        a = time.perf_counter()
        bgr = cv2.imread(str(paths[t]))
        name = rows[t]["top"][0][0]
        feat = vision(bgr)[0] if t % 2 == 0 else None
        tr.update(t, [] if name == "background" else [name], feat)
        lat.append((time.perf_counter() - a) * 1000)
    tr.finish(n - 1)
    del vision
    return {"frames": n, "latency_ms": {"median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                                        "mean": round(float(np.mean(lat)), 1)}, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    import torch
    import demo_v14 as D
    lat = json.loads((HERE / "results/latency.json").read_text())
    rids = lat["recordings"]
    vv = {r: v9(r, D) for r in rids}
    for r in rids:
        print(r, vv[r], flush=True)
    c_ar = lat["c_ar_ms_per_frame"] / 1000
    v_mean = sum(vv[r]["latency_ms"]["mean"] * vv[r]["frames"] for r in rids) / sum(vv[r]["frames"] for r in rids) / 1000
    v_p95 = max(vv[r]["latency_ms"]["p95"] for r in rids) / 1000
    L = (lat["chunk_frames"] - 1) / FPS + lat["chunk_frames"] * c_ar + v_p95
    L_or = lat["v14_p95_ms"] / 1000                      # oracle: no AR model, only v14's own compute
    res = {"gpu": torch.cuda.get_device_name(0), "recordings": rids, "v9": vv,
           "c_ar_ms_per_frame": lat["c_ar_ms_per_frame"], "v9_mean_ms": round(1000 * v_mean, 2),
           "v9_p95_ms": round(1000 * v_p95, 2), "realtime_one_gpu": c_ar + v_mean <= 1 / FPS,
           "L_v9_s": round(L, 3), "D_v9_frames": math.ceil(round(FPS * L, 6)),
           "L_oracle_s": round(L_or, 3), "D_oracle_frames": math.ceil(round(FPS * L_or, 6))}
    (HERE / "results/latency_v9.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "v9"}, indent=1))


if __name__ == "__main__":
    main()
