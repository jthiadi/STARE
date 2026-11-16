"""The deployed final-label model (action-end recipe: 3 V-JEPA 2 runs + VideoMAE, weight 0.95 vs the TCN, order prior) on the
IndustReal paper's AR protocol: 3,678 annotated test segments (true start / end), 74 actions, top-1. Paper: MViTv2 65.3.

Runs the teammate's own benchmark (scripts/recognition/action_end_benchmark.py) unchanged, with two inputs recomputed here
because they are not on the server: (1) the fine-tuned VideoMAE's scores on the annotated segments (3 temporal samplings, as
the recipe), (2) the causal TCN's per-frame probabilities, taken from our run of their TCN pipeline (live_probs.npz, step 1
of system/assistant/live) instead of their cache. Their guard is kept: val must give the locked 73.81 or test is not scored.
Run from system/assistant on the GPU machine (needs live_probs.npz for the 16 val + 32 test recordings):
    python ../../metrics/ar/score_final_labels_paper_protocol.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from assist.common import load_config
from assist.extract.action_end import SAMPLINGS
from monitor.settings import ROOT
from scripts.recognition import action_end_benchmark as B

OWN = ROOT / "output/assist/action_end_paper"           # our own folder (the teammate's files are not written)
VMAE_OWN = OWN / "vmae_full_clip_probs.npz"


def vmae_segment_probs() -> None:
    if VMAE_OWN.exists():
        return
    import torch
    from scripts.data.tas_labels import load_mapping
    from scripts.features import finetune_videomae_ar as vft
    ref = np.load(B.VJEPA / "clip_probs_vjft_hand_k12.npz")                # the annotated segments, in the saved order
    model = vft.load_model(len(load_mapping()), ROOT / "release_vjepa/vmae_full.pt")
    arrays = {}
    for split in ("val", "test"):
        keys = [str(k) for k in ref[f"keys/{split}"]]
        segs = [(k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2]), 1) for k in keys]
        arrays.update({f"keys/{split}": ref[f"keys/{split}"], f"labels/{split}": ref[f"labels/{split}"],
                       f"full/{split}": vft.tta_probs(model, segs, 8, SAMPLINGS)})
        print(json.dumps({"vmae_segments": split, "n": len(segs)}), flush=True)
    del model; torch.cuda.empty_cache()
    OWN.mkdir(parents=True, exist_ok=True)
    np.savez(VMAE_OWN, **arrays)


def main() -> None:
    import tempfile
    vmae_segment_probs()
    out = load_config()["paths"]["out"]
    B.VMAE = VMAE_OWN
    B.OUT = Path(tempfile.mkdtemp(prefix="action_end_rerun_"))        # their one-look guard applies per run folder
    B.from_cache = lambda rid, cfg, actions: np.load(out / rid / "live_probs.npz")["probs"]
    B.main()                                                              # their guard, their scoring
    res = json.loads((B.OUT / "paper_test.json").read_text())
    dest = Path(__file__).resolve().parent / "results" / "final_labels_paper_protocol.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
