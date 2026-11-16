"""Frozen per-frame features for the causal action model on MECCANO (STARE-lite, see PREREGISTRATION.md).
Same extractors as STARE's action recognition (teammate's code, unchanged): DINOv2-small on 3 views of the frame, and
VideoMAE-SSv2 with each 16-frame clip ENDING at its anchor (causal, one clip every 4 frames).
Writes work/feats/{dino,ssv2}/<vid>.npz.   CUDA_VISIBLE_DEVICES=0 python metrics/meccano/features.py 0/2"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "system" / "assistant"
os.chdir(A)
sys.path.insert(0, str(A))

import numpy as np
import torch

VIDEOS = Path(os.environ.get("MECCANO_VIDEOS", Path(__file__).resolve().parents[2] / "data/MECCANO/MECCANO_RGB_Videos"))
OUT = HERE / "work" / "feats"


def main() -> None:
    from transformers import AutoModel
    from monitor.settings import DINO_V2_DIR
    from scripts.features import extract_ar_frame_features as dino_x
    from scripts.features import extract_videomae_features as ssv2_x

    k, n = (int(x) for x in sys.argv[1].split("/"))
    vids = sorted(p.stem for p in VIDEOS.glob("*.mp4"))[k::n]
    (OUT / "dino").mkdir(parents=True, exist_ok=True); (OUT / "ssv2").mkdir(parents=True, exist_ok=True)
    dino = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True).to("cuda").half().eval()
    for v in vids:
        f = OUT / "dino" / f"{v}.npz"
        if not f.exists():
            t0 = time.time()
            feats = dino_x.extract(dino, VIDEOS / f"{v}.mp4", 16)
            np.savez(f, feats=feats, views=np.array([x[0] for x in dino_x.VIEWS]))
            print(json.dumps({"dino": v, "frames": len(feats), "s": round(time.time() - t0, 1)}), flush=True)
    del dino; torch.cuda.empty_cache()
    ssv2 = ssv2_x.load_model()
    for v in vids:
        f = OUT / "ssv2" / f"{v}.npz"
        if not f.exists():
            t0 = time.time()
            feats, logits = ssv2_x.extract(ssv2, VIDEOS / f"{v}.mp4", 4, 8, causal=True)
            np.savez(f, feats=feats[:, None, :], video_logits=logits)
            print(json.dumps({"ssv2": v, "frames": len(feats), "s": round(time.time() - t0, 1)}), flush=True)
    print("FEATURES DONE", k, n, flush=True)


if __name__ == "__main__":
    main()
