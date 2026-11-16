"""Step 2 of the on-demand pipeline: the teammate's online V-JEPA 2 label (``--variant prefix``, the live system's) for one
video, with their functions and their chosen setting (V-JEPA 2 action-beginning run alone, w 0.1, prior strength 1.5,
smoothing 0.01: README "Action-beginning V-JEPA 2"; scripts/recognition/pipeline_on_segments.py LIVE_*).
Input: output/assist/<rid>/live_probs.npz (step 1). Writes output/assist/<rid>/ar_vjepa_prefix.jsonl (the engine's
``pred_vjepap`` actions) and vjepa_members.npz.
    python -m live.step_vjepa <rid> [--compare]   # --compare: member scores vs the saved test_members_0.npz"""
import json
import sys
import time

import numpy as np

from assist.common import load_config, write_jsonl
from assist.engine.events import build_events
from assist.extract import run_ar_vjepa as R
from assist.extract.online_vjepa import shown_probs, updates
from assist.extract.run_ar import topk_rows
from scripts.data.tas_labels import load_mapping
from scripts.recognition.predict_vjepa import member_probs
from scripts.recognition.sequence_recognition import transition_counts
from scripts.segmentation.run_official_split_eval import TRAIN

MEMBER, W, STRENGTH, SMOOTHING = "vjft_hand_k12_p10", 0.1, 1.5, 0.01


def main(rid: str, compare: bool = False) -> dict:
    R._use_variant("prefix")
    cfg, actions = load_config(), load_mapping()
    out = cfg["paths"]["out"] / rid
    t0 = time.time()
    live = np.load(out / "live_probs.npz")["probs"]
    e = cfg["events"]
    top = live.argmax(1)
    events = build_events([(f / cfg["fps"], actions[a], float(live[f, a])) for f, a in enumerate(top)], cfg["fps"],
                          e["min_frames"], e["min_conf"], tuple(e["ignore"]))
    upd = updates(events, cfg["fps"], len(live), R.EVERY)
    segments = [(rid, s, en) for _, s, en, _ in upd]
    members = member_probs(segments, R.WEIGHTS_DIR, R._caches(), fracs=(0.5,))
    np.savez_compressed(out / "vjepa_members.npz", upd=np.array(upd, np.int64), **members)
    first, pairs = transition_counts(TRAIN, actions)
    shown = shown_probs(live, events, upd, members[MEMBER], W, STRENGTH, SMOOTHING, first, pairs, R.DELAY)
    write_jsonl(out / R.LABELS, topk_rows(shown, actions, cfg["fps"], 5))
    rep = {"video": rid, "updates": len(upd), "seconds": round(time.time() - t0, 1)}
    if compare:
        try:
            p = np.load(R.OUT / "test_plan.npz"); saved = np.load(R.OUT / "test_members_0.npz")
            rids = list(saved["rids"]); plan = {r: p[r] for r in p.files}
            off = sum(len(plan[r]) for r in sorted(plan)[:sorted(plan).index(rid)] if r in rids) if rid in rids else None
            if off is not None:
                same_plan = np.array_equal(plan[rid], np.array(upd))
                ref = saved[MEMBER][off:off + len(plan[rid])]
                rep["same_update_windows_as_saved"] = bool(same_plan)
                if same_plan:
                    rep["max_abs_diff_vs_saved_member"] = float(np.abs(ref - members[MEMBER]).max())
                    rep["top1_agreement_vs_saved_member"] = float((ref.argmax(1) == members[MEMBER].argmax(1)).mean())
        except Exception as ex:                             # comparison only
            rep["compare_error"] = repr(ex)[:200]
    print(json.dumps(rep), flush=True)
    return rep


if __name__ == "__main__":
    main(sys.argv[1], "--compare" in sys.argv)
