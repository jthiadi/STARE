"""The causal AR pipeline from a video file, on the GPU (used by ``run_ar --compute``).

Same features and networks as the causal TCN was trained on (see scripts/benchmark/pipeline_cost.py):
DINOv2-small (3 views + hi-res hands), 2 fine-tuned DINOv2 hand-crop backbones, VideoMAE-SSv2 with clips
ending at each frame, 2 IndustReal-fine-tuned VideoMAEs (causal clips), then the verb and noun causal TCN
ensembles averaged over both cross-fitted feature variants. One model is on the GPU at a time (< 2 GB).
Intermediate features go to ``output/assist/_features/<name>/<rid>.npz`` and are reused when present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from assist.common import ROOT

EVAL = ROOT / "output/official_split_eval"
FEATS = ROOT / "output/assist/_features"


class Timer:
    def __init__(self, name: str, timing: dict):
        self.name, self.timing = name, timing

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.timing[self.name] = {"seconds": round(time.perf_counter() - self.start, 2),
                                  "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2**20)}
        print(json.dumps({"component": self.name, **self.timing[self.name]}), flush=True)


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def finetuned_dino(video: Path, checkpoint: Path, n_obj: int, n_verb: int, batch: int = 32) -> dict:
    """Hands crop -> JPEG (as dump_ar_crops) -> fine-tuned DINOv2 (as finetune_dino_ar.extract)."""
    from scripts.features.dump_ar_crops import CROP_SIZE, HANDS_BOX
    from scripts.features.finetune_dino_ar import INPUT_HW, Net, _to_tensor, embed_batch

    model = Net(n_obj, n_verb).to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval()
    cap, parts, buf = cv2.VideoCapture(str(video)), [], []
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                crop = frame[int(HANDS_BOX[1] * h):int(HANDS_BOX[3] * h), int(HANDS_BOX[0] * w):int(HANDS_BOX[2] * w)]
                crop = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_AREA)
                crop = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)
                crop = cv2.resize(crop, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
                buf.append(_to_tensor(crop))
            if buf and (len(buf) >= batch or not ok):
                parts.append(embed_batch(model, torch.from_numpy(np.stack(buf)).to("cuda")))
                buf = []
            if not ok:
                break
    cap.release()
    del model
    torch.cuda.empty_cache()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def tcn_probs(target: str, rid: str, dirs: dict[str, Path]) -> tuple[np.ndarray, list[str]]:
    from scripts.segmentation.train_ar_object_tcn import TCN, load_features, predict_proba

    run = EVAL / f"tcn_{target}_causal"
    bundle = torch.load(run / "model.pt", map_location="cuda", weights_only=False)
    variants = [load_features(rid, dirs["dino"], False, [dirs["hires"], dirs["ssv2"], dirs[f"ftdino{i}"],
                                                         dirs[f"vmae{i}"]]) for i in (0, 1)]
    probs = []
    for sd in bundle["models"]:
        m = TCN(bundle["dim"], len(bundle["classes"]), hidden=bundle["hidden"], layers=bundle["layers"],
                stages=bundle["stages"], aux=bundle["aux"], causal=True).to("cuda")
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        with torch.no_grad():
            probs += [predict_proba(m.eval(), v, "cuda") for v in variants]
        del m
    torch.cuda.empty_cache()
    return np.mean(probs, axis=0), list(bundle["classes"])


def run_pipeline(rid: str, cfg: dict):
    from transformers import AutoModel

    from monitor.settings import DINO_V2_DIR
    from scripts.data.tas_labels import load_mapping
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as ssv2_x
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features.finetune_dino_ar import VERBS

    video = cfg["paths"]["videos"] / f"{rid}.mp4"
    dirs = {"dino": FEATS / "dino", "hires": FEATS / "dino_hires", "ssv2": FEATS / "ssv2_causal",
            "ftdino0": FEATS / "ftdino0", "ftdino1": FEATS / "ftdino1",
            "vmae0": FEATS / "vmae0_causal", "vmae1": FEATS / "vmae1_causal"}
    timing: dict = {"gpu": torch.cuda.get_device_name(0)}

    def todo(name: str) -> bool:
        return not (dirs[name] / f"{rid}.npz").is_file()

    if todo("dino") or todo("hires"):
        dino = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
        with Timer("dinov2_3_views", timing):
            _save(dirs["dino"] / f"{rid}.npz", feats=dino_x.extract(dino, video, 16),
                  views=np.array([v[0] for v in dino_x.VIEWS]))
        with Timer("dinov2_hires_hands", timing):
            _save(dirs["hires"] / f"{rid}.npz", feats=dino_x.extract(dino, video, 16, views=dino_x.HIRES_VIEWS, grid=True),
                  views=np.array([v[0] for v in dino_x.HIRES_VIEWS]))
        del dino
        torch.cuda.empty_cache()
    meta = json.loads((EVAL / "finetune/meta.json").read_text(encoding="utf-8"))
    for k in (0, 1):
        if todo(f"ftdino{k}"):
            with Timer(f"finetuned_dinov2_fold{k}", timing):
                _save(dirs[f"ftdino{k}"] / f"{rid}.npz",
                      **finetuned_dino(video, EVAL / f"finetune/fold{k}.pt", len(meta["objects"]), len(VERBS)))
    if todo("ssv2"):
        ssv2 = ssv2_x.load_model()
        with Timer("videomae_ssv2_causal", timing):
            feats, logits = ssv2_x.extract(ssv2, video, 4, 4, causal=True)
            _save(dirs["ssv2"] / f"{rid}.npz", feats=feats[:, None, :], video_logits=logits)
        del ssv2
        torch.cuda.empty_cache()
    actions = load_mapping()
    cache = FEATS / "frame_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for k in (0, 1):
        if todo(f"vmae{k}"):
            with Timer(f"frame_cache_fold{k}", timing):
                vft.build_cache(rid, video, cache)
            model = vft.load_model(len(actions), EVAL / f"vmae_finetune/fold{k}/best.pt")
            with Timer(f"finetuned_videomae_fold{k}_causal", timing):
                feats, logits = vft.extract(model, rid, 4, 8, cache=cache, causal=True)
                _save(dirs[f"vmae{k}"] / f"{rid}.npz", feats=feats[:, None, :], video_logits=logits)
            del model
            torch.cuda.empty_cache()
    with Timer("tcn_verb_noun", timing):
        pv, verbs = tcn_probs("verb", rid, dirs)
        pn, nouns = tcn_probs("noun", rid, dirs)
    return pv, pn, verbs, nouns, timing
