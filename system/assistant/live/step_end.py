"""Step 3 of the on-demand pipeline: the teammate's action-end label ("Final labels", 75.07 % on the paper's test) for one
video, with their functions and locked recipe (assist.extract.action_end: 3 V-JEPA 2 runs + VideoMAE, 3 samplings each,
weight 0.95 vs the TCN, prior strength 0.2, smoothing 0.01). Spans = the TCN's detected actions (final_spans), as live.
Input: output/assist/<rid>/live_probs.npz. Writes output/assist/<rid>/action_end.jsonl and action_end_probs.npz.
    python -m live.step_end <rid> [--compare]   # --compare: member scores vs the teammate's saved probs (if any)"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from assist.common import load_config, write_jsonl
from assist.engine.events import build_events
from assist.extract.action_end import MEMBERS, SAMPLINGS, SMOOTHING, STRENGTH, WEIGHT, end_labels, mean_clip
from assist.extract.run_action_end import RUNS, VMAE_WEIGHTS, _vjepa, end_rows, final_spans
from scripts.data.tas_labels import load_mapping
from scripts.recognition.sequence_recognition import transition_counts
from scripts.segmentation.run_official_split_eval import TRAIN

import os
SAVED = Path(os.environ.get("SAVED_ACTION_END_PROBS", "output/assist/action_end/probs"))   # teammate's saved runs (--compare only)


def main(rid: str, compare: bool = False, batch: int = 8) -> dict:
    import torch
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features import vjepa2_finetune as ft

    cfg, actions = load_config(), load_mapping()
    out = cfg["paths"]["out"] / rid
    t0 = time.time()
    live = np.load(out / "live_probs.npz")["probs"]
    e = cfg["events"]
    top = live.argmax(1)
    events = build_events([(f / cfg["fps"], actions[a], float(live[f, a])) for f, a in enumerate(top)], cfg["fps"],
                          e["min_frames"], e["min_conf"], tuple(e["ignore"]))
    spans = final_spans(events, cfg["fps"], len(live))
    probs = {}
    for name in RUNS:                                         # exactly run_action_end.score's predict, for one video
        model, args = _vjepa(name)
        p = ft.predict(model, [(rid, s, en, 1, f) for s, en in spans for f in SAMPLINGS], args, batch)
        probs[name] = p.reshape(-1, len(SAMPLINGS), p.shape[1]).mean(1)
        del model
        torch.cuda.empty_cache()
    model = vft.load_model(len(actions), VMAE_WEIGHTS)
    probs["vmae_full"] = vft.tta_probs(model, [(rid, s, en, 1) for s, en in spans], batch, SAMPLINGS)
    del model
    torch.cuda.empty_cache()
    np.savez_compressed(out / "action_end_probs.npz", spans=np.array(spans, np.int64), **probs)
    first, pairs = transition_counts(TRAIN, actions)
    clip = mean_clip(probs)
    pred = end_labels(live, spans, clip, WEIGHT, STRENGTH, SMOOTHING, first, pairs)
    write_jsonl(out / "action_end.jsonl", end_rows(events, pred.tolist(), clip, actions))
    rep = {"video": rid, "actions": len(spans), "seconds": round(time.time() - t0, 1)}
    if compare:
        diffs = {}
        for m in MEMBERS:
            f = SAVED / m / f"{rid}.npz"
            if f.exists():
                ref = np.load(f)["probs"]
                diffs[m] = (round(float(np.abs(ref - probs[m]).max()), 5) if ref.shape == probs[m].shape
                            else f"shape {ref.shape} vs {probs[m].shape}")
        rep["max_abs_diff_vs_saved"] = diffs or "no saved probs for this video"
    print(json.dumps(rep), flush=True)
    return rep


if __name__ == "__main__":
    main(sys.argv[1], "--compare" in sys.argv)
