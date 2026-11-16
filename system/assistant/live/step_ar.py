"""Step 1 of the on-demand pipeline: the teammate's causal AR pipeline on one video (their code, unchanged).
assist.extract.causal_pipeline.run_pipeline -> verb / noun TCN probabilities -> action probabilities
(assist.extract.run_ar.action_probs). Writes output/assist/<rid>/live_probs.npz (full per-frame probabilities, needed by
the online V-JEPA 2 step) and output/assist/<rid>/ar_live.jsonl (top-5 rows, same format as their ar.jsonl).
    python -m live.step_ar <rid> [--compare]      # --compare: agreement with the saved ar.jsonl"""
import json
import sys
import time

import numpy as np

from assist.common import load_config, write_jsonl
from assist.extract.causal_pipeline import run_pipeline
from assist.extract.run_ar import action_probs, topk_rows
from scripts.data.tas_labels import load_mapping


def main(rid: str, compare: bool = False) -> dict:
    cfg, actions = load_config(), load_mapping()
    out = cfg["paths"]["out"] / rid
    t0 = time.time()
    pv, pn, verbs, nouns, timing = run_pipeline(rid, cfg)
    probs = action_probs(pv, pn, verbs, nouns, actions)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "live_probs.npz", probs=probs.astype(np.float32))
    write_jsonl(out / "ar_live.jsonl", topk_rows(probs, actions, cfg["fps"], 5))
    rep = {"video": rid, "frames": len(probs), "seconds": round(time.time() - t0, 1)}
    if compare and (out / "ar.jsonl").exists():
        saved = [json.loads(l)["top"][0][0] for l in open(out / "ar.jsonl", encoding="utf-8")]
        mine = [actions[i] for i in probs.argmax(1)]
        n = min(len(saved), len(mine))
        rep["top1_agreement_with_saved_ar_jsonl"] = round(sum(a == b for a, b in zip(saved[:n], mine[:n])) / n, 4)
        rep["saved_frames"] = len(saved)
    print(json.dumps(rep), flush=True)
    return rep


if __name__ == "__main__":
    main(sys.argv[1], "--compare" in sys.argv)
