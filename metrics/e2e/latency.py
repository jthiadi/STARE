"""Measured compute of the live system on this GPU (see PREREGISTRATION.md). Writes results/latency.json.
Run from STARE with env.sh sourced:  CUDA_VISIBLE_DEVICES=0 python metrics/e2e/latency.py"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "system" / "assistant"
os.chdir(A)
sys.path.insert(0, str(A))

import numpy as np
import torch

from assist.common import load_config
from assist.extract import causal_pipeline as CP
from assist.industreal import SPLITS

CHUNK, FPS = 4, 10.0
RIDS = [SPLITS["test"][i] for i in (0, 11, 22)]          # fixed before measuring
WORK = HERE / "work" / "latency"


def ar_path(rid: str) -> dict:
    """The causal AR pipeline of causal_pipeline.run_pipeline, every component with batch CHUNK, features kept in WORK."""
    from transformers import AutoModel

    from monitor.settings import DINO_V2_DIR
    from scripts.data.tas_labels import load_mapping
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as ssv2_x
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features.finetune_dino_ar import VERBS

    cfg = load_config()
    video = cfg["paths"]["videos"] / f"{rid}.mp4"
    F = WORK / "_features"
    dirs = {k: F / k for k in ("dino", "hires", "ssv2", "ftdino0", "ftdino1", "vmae0", "vmae1")}
    t: dict = {}
    dino = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
    with CP.Timer("dinov2_3_views", t):
        CP._save(dirs["dino"] / f"{rid}.npz", feats=dino_x.extract(dino, video, CHUNK),
                 views=np.array([v[0] for v in dino_x.VIEWS]))
    with CP.Timer("dinov2_hires_hands", t):
        CP._save(dirs["hires"] / f"{rid}.npz", feats=dino_x.extract(dino, video, CHUNK, views=dino_x.HIRES_VIEWS, grid=True),
                 views=np.array([v[0] for v in dino_x.HIRES_VIEWS]))
    del dino; torch.cuda.empty_cache()
    meta = json.loads((CP.EVAL / "finetune/meta.json").read_text(encoding="utf-8"))
    for k in (0, 1):
        with CP.Timer(f"finetuned_dinov2_fold{k}", t):
            CP._save(dirs[f"ftdino{k}"] / f"{rid}.npz", **CP.finetuned_dino(
                video, CP.EVAL / f"finetune/fold{k}.pt", len(meta["objects"]), len(VERBS), batch=CHUNK))
    ssv2 = ssv2_x.load_model()
    with CP.Timer("videomae_ssv2_causal", t):
        feats, logits = ssv2_x.extract(ssv2, video, 4, 1, causal=True)
        CP._save(dirs["ssv2"] / f"{rid}.npz", feats=feats[:, None, :], video_logits=logits)
    del ssv2; torch.cuda.empty_cache()
    actions = load_mapping()
    cache = WORK / "frame_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for k in (0, 1):
        with CP.Timer(f"frame_cache_fold{k}", t):
            vft.build_cache(rid, video, cache)
        model = vft.load_model(len(actions), CP.EVAL / f"vmae_finetune/fold{k}/best.pt")
        with CP.Timer(f"finetuned_videomae_fold{k}_causal", t):
            feats, logits = vft.extract(model, rid, 4, 1, cache=cache, causal=True)
            CP._save(dirs[f"vmae{k}"] / f"{rid}.npz", feats=feats[:, None, :], video_logits=logits)
        del model; torch.cuda.empty_cache()
        shutil.rmtree(cache / rid, ignore_errors=True)
    with CP.Timer("tcn_verb_noun", t):
        pv, _ = CP.tcn_probs("verb", rid, dirs)
        CP.tcn_probs("noun", rid, dirs)
    n = len(pv)
    sec = sum(v["seconds"] for v in t.values())
    return {"frames": n, "components": t, "seconds": round(sec, 2), "ms_per_frame": round(1000 * sec / n, 2)}


def v14(rid: str) -> dict:
    """The frozen v14 live step (live/step_v14.py) in a work folder; its own per-frame timing."""
    import live.step_v14 as S
    w = WORK / "v14"
    (w / "output/assist" / rid).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(A / "output/assist" / rid / "ar_vjepa_prefix.jsonl", w / "output/assist" / rid / "ar_vjepa_prefix.jsonl")
    cwd = os.getcwd(); os.chdir(w)
    try:
        r = S.main(rid)
    finally:
        os.chdir(cwd)
    return {"frames": r["frames"], "latency_ms": r["latency_ms"]}


def ensemble_pass() -> dict:
    from assist.extract import run_action_end as R
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.benchmark()
    return json.loads(buf.getvalue())


def main() -> None:
    gpu = torch.cuda.get_device_name(0)
    ar = {rid: ar_path(rid) for rid in RIDS}
    vv = {rid: v14(rid) for rid in RIDS}
    ens = ensemble_pass()
    frames = sum(r["frames"] for r in ar.values())
    c_ar = sum(r["seconds"] for r in ar.values()) / frames                      # s per frame, all components
    v_mean = sum(vv[r]["latency_ms"]["mean"] * vv[r]["frames"] for r in RIDS) / sum(vv[r]["frames"] for r in RIDS) / 1000
    v_p95 = max(vv[r]["latency_ms"]["p95"] for r in RIDS) / 1000              # worst recording's p95
    P = ens["ms_per_action"] / 1000
    L = (CHUNK - 1) / FPS + CHUNK * c_ar + v_p95
    res = {"gpu": gpu, "recordings": RIDS, "chunk_frames": CHUNK,
           "ar_path": ar, "v14": vv, "ensemble_pass": ens,
           "c_ar_ms_per_frame": round(1000 * c_ar, 2), "v14_mean_ms": round(1000 * v_mean, 2),
           "v14_p95_ms": round(1000 * v_p95, 2), "ensemble_pass_ms": round(1000 * P, 1),
           "realtime_one_gpu": c_ar + v_mean <= 1 / FPS,
           "L_s": round(L, 3), "D_tcn_frames": math.ceil(round(FPS * L, 6)),
           "D_ens_frames": math.ceil(round(FPS * (0.5 + L + P), 6)), "ens_ready_after_end_s": round(L + P, 3)}
    out = HERE / "results" / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("ar_path", "v14", "ensemble_pass")}, indent=1))


if __name__ == "__main__":
    main()
