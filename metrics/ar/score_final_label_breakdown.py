"""What each part of the final label adds on the IndustReal paper protocol (paper Table Y; diagnostic only, nothing is chosen).

Rebuilds the deployed action-end recipe one part at a time from the saved outputs: each model alone, the mean of the 3
V-JEPA 2 runs, + VideoMAE, + the TCN (w 0.95), + the action-order prior (= the final label). Same inputs as
score_final_labels_paper_protocol.py (the fine-tuned VideoMAE's segment scores, the causal TCN's live_probs.npz); it refuses to
write anything unless the full recipe gives the locked val 73.81 and test 75.07. Only test is reported: the deployed TCN has
seen the val videos (README "Note on val"), so val rows that use it would be inflated. The MECCANO column of Table Y is
system/assistant/output/meccano_action_end/ablation.json.

    cd system/assistant && python ../../metrics/ar/score_final_label_breakdown.py   # CPU, ~1 min -> results/final_label_breakdown.json
"""
import json
import sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[2] / "system" / "assistant"
sys.path.insert(0, str(A))
from assist.common import load_config  # noqa: E402
from assist.extract.action_end import MEMBERS, SMOOTHING, STRENGTH, WEIGHT, end_labels  # noqa: E402
from assist.industreal import SPLITS  # noqa: E402
from monitor.settings import ROOT  # noqa: E402
from scripts.data.tas_labels import load_mapping  # noqa: E402
from scripts.recognition import action_end_benchmark as B  # noqa: E402
from scripts.recognition.sequence_recognition import transition_counts  # noqa: E402
from scripts.segmentation.run_official_split_eval import TRAIN  # noqa: E402

RESULT = Path(__file__).resolve().parent / "results" / "final_label_breakdown.json"
VMAE = ROOT / "output/assist/action_end_paper/vmae_full_clip_probs.npz"
NAMES = {"vmae_full": "VideoMAE alone", "vjft_area_k12": "V-JEPA 2 work-area crop alone",
         "vjft_hand_k12": "V-JEPA 2 hand crop alone", "vjft_hand_k12_s1": "V-JEPA 2 hand crop, seed 1, alone"}
STEPS = (("TCN alone", 0.0, 0.0), ("+ TCN (w = 0.95)", WEIGHT, 0.0), ("+ order prior (final label)", WEIGHT, STRENGTH))
LOCKED = {"val": 73.81, "test": 75.07}


def acc(pred: np.ndarray, labels: np.ndarray) -> float:
    return round(100 * float((pred == labels).mean()), 2)


def rows(split: str, first: np.ndarray, pairs: np.ndarray) -> dict[str, float]:
    B.VMAE = VMAE
    keys, labels, probs = B.load_split(split)
    out = {NAMES[m]: acc(p.argmax(1) + 1, labels) for m, p in probs.items()}
    out["Mean of 3 V-JEPA 2 runs"] = acc((sum(probs[m] for m in MEMBERS[1:]) / 3).argmax(1) + 1, labels)
    mean = sum(probs[m] for m in MEMBERS) / len(MEMBERS)
    out["+ VideoMAE (mean of 4)"] = acc(mean.argmax(1) + 1, labels)
    out_dir = load_config()["paths"]["out"]
    for name, w, strength in STEPS:
        pred = np.zeros(len(labels), np.int64)
        for rid in SPLITS[split]:
            order = sorted((int(k.split("|")[1]), i) for i, k in enumerate(keys) if k.split("|")[0] == rid)
            idx = np.array([i for _, i in order])
            spans = [tuple(int(x) for x in keys[i].split("|")[1:]) for i in idx]
            tcn = np.load(out_dir / rid / "live_probs.npz")["probs"]
            pred[idx] = end_labels(tcn, spans, mean[idx], w, strength, SMOOTHING, first, pairs)
        out[name] = acc(pred, labels)
    return out


def main() -> None:
    first, pairs = transition_counts(TRAIN, load_mapping())
    val, test = rows("val", first, pairs), rows("test", first, pairs)
    final = "+ order prior (final label)"
    if (val[final], test[final]) != (LOCKED["val"], LOCKED["test"]):
        raise SystemExit(f"full recipe gives val {val[final]} / test {test[final]}, not the locked 73.81 / 75.07")
    report = {"protocol": "IndustReal paper protocol: 3,678 annotated test segments, 74 actions, top-1; diagnostic, "
                          "nothing chosen; settings = the deployed recipe (locked on val)",
              "val_full_recipe": val[final], "test": test}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
