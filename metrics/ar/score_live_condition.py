"""The three AR models under the LIVE condition on the 32 IndustReal test recordings: no true start / end and no future frames.

The system finds the actions itself (the causal TCN's events, assist.engine.events.build_events, as live). Each model then
labels each detected action using only frames up to that action's confirmed end:
  TCN                      mean TCN probability over the detected action's frames (their tcn_on_segment)
  V-JEPA 2 (fine-tuned)    the action-beginning V-JEPA 2 on the detected action's final window (its last live update)
  Final labels (ensemble)  3 V-JEPA 2 + VideoMAE + TCN + order prior (action_end.jsonl, their end_labels)
Score = share of the 3,678 annotated test actions whose most-overlapping detection has the right label (no overlapping
detection = wrong): the teammate's assist.extract.run_action_end.best_overlap_accuracy. Also reported for the per-frame labels:
frame accuracy (every frame, background included).
Inputs (system/assistant/live outputs): live_probs.npz, vjepa_members.npz, action_end.jsonl, ar_vjepa_prefix.jsonl.
Run from system/assistant:  python ../../metrics/ar/score_live_condition.py"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from assist.common import load_config
from assist.engine.events import build_events
from assist.extract.run_action_end import best_overlap_accuracy, final_spans
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping
from scripts.recognition.pipeline_on_segments import tcn_on_segment

MEMBER = "vjft_hand_k12_p10"


def main() -> None:
    cfg, actions = load_config(), load_mapping(); idx = {a: i for i, a in enumerate(actions)}
    out, lab, e = cfg["paths"]["out"], cfg["paths"]["labels"], cfg["events"]
    tot = {"TCN": [0, 0], "V-JEPA 2 (fine-tuned, action-beginning)": [0, 0], "Final labels (V-JEPA 2 ensemble + VideoMAE + TCN)": [0, 0]}
    frames = {"TCN": [0, 0], "Live label: TCN + V-JEPA 2 (w 0.1)": [0, 0]}
    n_det = 0
    for rid in SPLITS["test"]:
        live = np.load(out / rid / "live_probs.npz")["probs"]; n = len(live)
        top = live.argmax(1)
        events = build_events([(f / cfg["fps"], actions[a], float(live[f, a])) for f, a in enumerate(top)], cfg["fps"],
                              e["min_frames"], e["min_conf"], tuple(e["ignore"]))
        spans = final_spans(events, cfg["fps"], n); n_det += len(spans)
        segs, gt = [], np.zeros(n, np.int64)
        for r in csv.reader(open(lab / rid / "AR_labels.csv", encoding="utf-8")):
            a, s, en = r[2], int(Path(r[3]).stem), min(int(Path(r[4]).stem), n - 1)
            if a in idx:
                segs.append((s, en, idx[a])); gt[s:en + 1] = idx[a]
        tcn = [int(tcn_on_segment(live, s, en).argmax()) + 1 for s, en in spans]
        z = np.load(out / rid / "vjepa_members.npz"); upd, mem = z["upd"], z[MEMBER]
        last = {}                                              # each event's final window = its last update
        for j, (i, _, _, _) in enumerate(upd):
            last[int(i)] = j
        vj = [int(mem[last[i]].argmax()) + 1 for i in range(len(events))]
        fin = [idx[json.loads(l)["label"]] for l in open(out / rid / "action_end.jsonl", encoding="utf-8")]
        assert len(fin) == len(spans) == len(vj), rid
        for name, labels in zip(tot, (tcn, vj, fin)):
            r_, t_ = best_overlap_accuracy(spans, labels, segs); tot[name][0] += r_; tot[name][1] += t_
        shown = [json.loads(l)["top"][0][0] for l in open(out / rid / "ar_vjepa_prefix.jsonl", encoding="utf-8")]
        for name, pred in (("TCN", top), ("Live label: TCN + V-JEPA 2 (w 0.1)", np.array([idx[s] for s in shown]))):
            frames[name][0] += int((pred == gt).sum()); frames[name][1] += n
    res = {"condition": "live: actions detected by the system, each labelled with frames up to its end; 32 test recordings",
           "detected_actions": n_det,
           "true_actions_labelled_right_pct": {k: round(100 * r / t, 2) for k, (r, t) in tot.items()},
           "true_actions": next(iter(tot.values()))[1],
           "frame_accuracy_pct": {k: round(100 * r / t, 2) for k, (r, t) in frames.items()}}
    dest = Path(__file__).resolve().parent / "results" / "live_condition_test.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
