"""The live action models on the IndustReal paper's AR protocol: each annotated test segment cut at its TRUE start and end,
classified among the 74 actions (background excluded), top-1; the label is fixed at the segment's end and the order prior uses
earlier actions only. Paper: SlowFast 60.4, MViTv2 65.3 (RGB).

The teammate's scripts/recognition/pipeline_on_segments.py, unchanged except for where the TCN's per-frame probabilities come
from (our run of their causal TCN pipeline, live_probs.npz, instead of their cache):
  TCN on a segment   mean of its frame probabilities over the segment's own frames (their tcn_on_segment)
  V-JEPA 2           the action-beginning run (the one in the live label) on the segment's own frames, 3 samplings
                     (their --score, output/assist/pipeline_segments/probs)
  blend              (1 - w) log p_tcn + w log p_vjepa, + order prior over earlier actions (their fused_scores, decode_filter)

No setting is chosen here (the recomputed TCN has seen the val videos, so val cannot choose fairly). Fixed settings only:
  TCN alone (w 0) | V-JEPA 2 alone (w 1) | the live system's label (w 0.1, prior 0.15, smoothing 0.01) |
  the teammate's val-chosen blend (w 0.7, prior 0.2, smoothing 0.01; chosen by them with out-of-fold val outputs)
Run from system/assistant:  python ../../metrics/ar/score_live_models_paper_protocol.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from assist.common import load_config
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping
from scripts.recognition import pipeline_on_segments as P
from scripts.recognition.fuse_recognition import fused_scores
from scripts.recognition.online_vjepa_end import decode_filter
from scripts.recognition.sequence_recognition import transition_counts, videos
from scripts.segmentation.run_official_split_eval import TRAIN

SETTINGS = {"TCN alone": (0.0, 0.0, 0.01), "V-JEPA 2 alone (action-beginning run)": (1.0, 0.0, 0.01),
            "Live label: TCN + V-JEPA 2, w 0.1 (as deployed)": (P.LIVE_WEIGHT, round(P.LIVE_WEIGHT * P.LIVE_STRENGTH, 3), P.LIVE_SMOOTHING),
            "TCN + V-JEPA 2, w 0.7 (teammate's val-chosen blend)": (0.7, 0.2, 0.01)}


def main() -> None:
    cfg, actions = load_config(), load_mapping()
    out = cfg["paths"]["out"]
    import assist.extract.run_ar as RA
    RA.from_cache = lambda rid, cfg, actions: np.load(out / rid / "live_probs.npz")["probs"]
    first, pairs = transition_counts(TRAIN, actions)
    res = {"protocol": "IndustReal AR: true segments, 74 actions, top-1; label fixed at segment end; prior over earlier actions",
           "paper": {"SlowFast": 60.4, "MViTv2": 65.3}}
    for split in ("test", "val"):
        data = P._split(SPLITS[split], cfg, actions)
        groups = videos(data["keys"])
        res[split] = {"segments": int(len(data["labels"]))}
        for name, (w, lam, alpha) in SETTINGS.items():
            pred = decode_filter(fused_scores(data, ("vjepa",), w), groups, first, pairs, lam, alpha)
            res[split][name] = round(float((pred == data["labels"]).mean() * 100), 2)
    res["note"] = ("val is not a fair check here: the deployed TCN has seen the val videos (the teammate's README val numbers "
                   "use out-of-fold TCN outputs). Test numbers are fair: no setting was chosen on test or here.")
    dest = Path(__file__).resolve().parent / "results" / "live_models_paper_protocol.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
