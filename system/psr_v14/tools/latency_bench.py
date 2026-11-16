"""End-to-end per-frame latency of frozen v14, same pipeline as evaluated (val, first 3 videos x 1200 frames).

Per frame: decode JPEG -> AR lookup -> (every 2nd frame) DINOv2-small localizer + 160-px view-quality crop +
336-px crop (JPEG round trip, as in the evaluated cache) -> both DINOv2-base verifiers; every frame the official
ASD (YOLOv8m, 640 px) -> v14 state filter.
At the end of each video the mistake check runs once.  The AR model's own compute is NOT included: AR labels
are read from <ar-dir>/<video>/ar.jsonl.

    python tools/latency_bench.py --ar-dir <dir with <video>/ar.jsonl> [--weights weights]

Ported from the development script that produced results/latency_v13.json (same steps and frame selection;
it used the development copy of the tracker, numerically identical to psr/v11.py's V11Tracker).
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
os.environ.setdefault("VBACK", "base"); os.environ.setdefault("VRES", "336")

import cv2
import numpy as np
import torch

import train_verifier as TV
from psr.blend import Blend
from psr.core import frame_paths, list_videos, procedure_type
from psr.filter_tracker import FilterConfig
from psr.lr_model import LRGroupModel
from psr.tracker import GroupModel, TrackerConfig
from psr.v11 import mistake_alerts
from psr.v14 import V14Tracker, ASD_CLASSES
from psr.vision import VisionVerifier, crop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("--asd-weights", default=str(HERE / "weights" / "asd_best_IndustRealandSynthetic.pt"))
    ap.add_argument("--model", default=str(HERE / "model"))
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--out", default=str(HERE / "results" / "reproduced" / "latency_v14.json"))
    a = ap.parse_args()

    fz = Path(a.model)
    cfg = json.loads((fz / "config.json").read_text()); sm = cfg["step_model"]
    model = Blend(GroupModel(fz / "tables.json"), LRGroupModel(path=fz / sm["lr_path"]), sm["w_lr"])
    tcfg = TrackerConfig(w_action=sm["w_action"], w_since=sm["w_since"], null_bias=sm["null_bias"])
    fcfg = FilterConfig(**cfg["filter"])
    vers = []
    for v in cfg["verifiers"]:
        TV.UNFREEZE = v.get("unfreeze", 4)
        m = TV.Verifier().cuda()
        m.load_state_dict(torch.load(Path(a.weights) / f"{v['tag']}.pt", map_location="cuda"))
        vers.append(m.eval().half())
    vv = VisionVerifier(bank="exemplar")
    from ultralytics import YOLO
    asd_m = YOLO(a.asd_weights); ac = cfg["asd"]

    lat = {k: [] for k in ["decode", "ar", "asd", "localize", "verifiers", "filter", "end_to_end"]}; mist_ms = []
    torch.cuda.reset_peak_memory_stats()
    for vid in list_videos("val")[:a.videos]:
        P = frame_paths("val", vid)
        rows = [json.loads(l) for l in open(Path(a.ar_dir) / vid / "ar.jsonl", encoding="utf-8")]
        live = [[] if r["top"][0][0] == "background" else [r["top"][0][0]] for r in rows]
        tr = V14Tracker(procedure_type(vid), fcfg, model, tcfg, d0=cfg["v11"]["d0"], asd={}, w=ac["w"], eps=ac["eps"], rate=ac["rate"]); vv.prev_box = None
        n = min(len(P), a.frames)
        for t in range(n):
            t_a = time.perf_counter(); bgr = cv2.imread(str(P[t])); t_b = time.perf_counter()
            names = live[t]; t_g = time.perf_counter()
            if ac["rate"] == 1 or t % 2 == 1:
                res = asd_m.predict(bgr, conf=0.05, imgsz=640, verbose=False)[0]
                if len(res.boxes):
                    i = int(res.boxes.conf.argmax()); st = ASD_CLASSES[int(res.boxes.cls[i])]
                    if float(res.boxes.conf[i]) >= ac["cmin"] and st not in ("background", "error_state"): tr.asd[t] = st
            t_c = time.perf_counter(); feat = None; l1 = l2 = 0.0
            if t % 2 == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); _, pt = vv.enc.encode(rgb, vv.FRAME_SIZE)
                box, _ = vv.localize(pt)
                t_d = time.perf_counter(); l1 = t_d - t_c
                if box is not None:
                    roi = cv2.resize(crop(rgb, box, pad=0.1), (160, 160), interpolation=cv2.INTER_AREA)
                    q, _ = vv.view_quality(roi)
                    hr = cv2.resize(crop(bgr, box, pad=0.1), (336, 336), interpolation=cv2.INTER_AREA)
                    jpg = cv2.imencode(".jpg", hr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
                    x = TV.to_tensor([jpg]).cuda().half()
                    with torch.no_grad():
                        ps = [torch.sigmoid(m(x).float())[0].cpu().numpy() for m in vers]
                    p = np.mean([np.stack([1 - s, s, np.full(10, 1e-3, np.float32)], 1).reshape(-1) for s in ps], 0)
                    feat = {"cls": p, "q": q}
                torch.cuda.synchronize(); l2 = time.perf_counter() - t_d
            t_e = time.perf_counter(); tr.update(t, names, feat); t_f = time.perf_counter()
            for k, x in zip(lat, [t_b - t_a, t_g - t_b, t_c - t_g, l1, l2, t_f - t_e, t_f - t_a]):
                lat[k].append(x)
        tr.finish(n - 1)
        g = time.perf_counter(); mistake_alerts(vid, tr, n); mist_ms.append(1000 * (time.perf_counter() - g))

    out = {k: {"median_ms": round(1000 * float(np.median(x)), 1), "p95_ms": round(1000 * float(np.percentile(x, 95)), 1),
               "max_ms": round(1000 * float(np.max(x)), 1)} for k, x in lat.items()}
    out["mistake_check_at_end_ms"] = [round(x, 1) for x in mist_ms]
    out["peak_gpu_mb"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
    out["gpu"] = torch.cuda.get_device_name(0)
    out["note"] = "AR read from ar.jsonl (AR model compute NOT included); first frames include warm-up"
    out["mean_end_to_end_ms"] = round(1000 * float(np.mean(lat["end_to_end"])), 1); out["frames"] = len(lat["end_to_end"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
