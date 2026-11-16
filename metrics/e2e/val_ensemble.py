"""Ensemble final labels on the 16 validation recordings (PREREGISTRATION.md, "Validation inputs").
TCN input = the out-of-fold validation labels v14 was selected with (output/assist/<val>/ar.jsonl, top 5 per frame;
the rest of the probability spread evenly). Then live/step_end.py unchanged (3 V-JEPA 2 runs + VideoMAE + TCN, locked
recipe) on the spans detected from those labels. Writes work/val_ensemble/<rid>/{live_probs.npz, ar_live.jsonl,
action_end.jsonl}; existing outputs elsewhere are not touched.
Run from STARE with env.sh sourced:  CUDA_VISIBLE_DEVICES=0 python metrics/e2e/val_ensemble.py 0/2   (shard k/n)"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "system" / "assistant"
os.chdir(A)
sys.path.insert(0, str(A))

import numpy as np

from assist.common import load_config
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping
import live.step_end as SE

WORK = HERE / "work" / "val_ensemble"


def oof_probs(rid: str, actions: list[str]) -> np.ndarray:
    idx = {a: i for i, a in enumerate(actions)}
    rows = [json.loads(l) for l in open(A / "output/assist" / rid / "ar.jsonl", encoding="utf-8")]
    P = np.zeros((len(rows), len(actions)), np.float32)
    for i, r in enumerate(rows):
        top = r["top"]
        P[i] = max(0.0, 1.0 - sum(p for _, p in top)) / (len(actions) - len(top))
        for a, p in top:
            P[i, idx[a]] = p
        P[i, idx[top[0][0]]] += 1e-6                   # ties at 3 decimals: keep the recorded top 1 (exact spans)
        assert r["f"] == i and P[i].argmax() == idx[top[0][0]]
    return P


def main() -> None:
    k, n = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))
    actions = load_mapping()
    cfg = load_config()
    cfg["paths"]["out"] = WORK
    SE.load_config = lambda *a, **kw: cfg
    for rid in SPLITS["val"][k::n]:
        d = WORK / rid
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "live_probs.npz").exists():
            np.savez_compressed(d / "live_probs.npz", probs=oof_probs(rid, actions))
            shutil.copyfile(A / "output/assist" / rid / "ar.jsonl", d / "ar_live.jsonl")
        if not (d / "action_end.jsonl").exists():
            SE.main(rid)
    print("VAL ENSEMBLE DONE", k, n, flush=True)


if __name__ == "__main__":
    main()
