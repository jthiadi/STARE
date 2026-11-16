"""Imports the UNMODIFIED official IndustReal evaluator (PSR/psr_utils.py).

psr_utils imports `bounding_box` only for drawing; we stub that one module so the metric code
(determine_performance, procedure_order_similarity, get_f1_score) runs untouched.
"""
import sys
import types

from .core import OFFICIAL_PSR

if "bounding_box" not in sys.modules:
    stub = types.ModuleType("bounding_box")
    stub.bounding_box = types.SimpleNamespace(add=lambda *a, **k: None)
    sys.modules["bounding_box"] = stub
if str(OFFICIAL_PSR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_PSR))

import psr_utils  # noqa: E402

FPS = psr_utils.FPS


def evaluate(gt, pred, proc_info):
    return psr_utils.determine_performance(gt, pred, proc_info, verbose=False)


def aggregate(per_video: list[dict]) -> dict:
    m = psr_utils.initiate_metrics()
    for x in per_video:
        psr_utils.update_metrics(m, x)
    n = len(per_video)
    return {
        "POS": sum(m["pos"]) / n, "F1": sum(m["f1"]) / n,
        "delay_frames": sum(m["avg_delay"]) / n,
        "TP": m["system_TPs"], "FP": m["system_FPs"], "FN": m["system_FNs"], "n_videos": n,
    }
