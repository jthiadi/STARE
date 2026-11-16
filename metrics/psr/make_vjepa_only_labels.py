"""Per-frame live labels from V-JEPA 2 alone (the action-beginning run), causal, for feeding v14: the teammate's
online_vjepa.shown_probs with w = 1.0 (their live system uses w = 0.1), same prior (1.5) / smoothing (0.01) / 0.5 s delay.
Writes <out_dir>/<rid>/ar.jsonl for the 32 test recordings.  Run from system/assistant:
    python ../../metrics/psr/make_vjepa_only_labels.py <out_dir>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from assist.common import load_config, write_jsonl
from assist.engine.events import build_events
from assist.extract import run_ar_vjepa as R
from assist.extract.online_vjepa import shown_probs
from assist.extract.run_ar import topk_rows
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping
from scripts.recognition.sequence_recognition import transition_counts
from scripts.segmentation.run_official_split_eval import TRAIN

cfg, actions = load_config(), load_mapping(); out = cfg["paths"]["out"]; e = cfg["events"]
first, pairs = transition_counts(TRAIN, actions)
dest = Path(sys.argv[1])
for rid in SPLITS["test"]:
    live = np.load(out / rid / "live_probs.npz")["probs"]; top = live.argmax(1)
    events = build_events([(f / cfg["fps"], actions[a], float(live[f, a])) for f, a in enumerate(top)], cfg["fps"],
                          e["min_frames"], e["min_conf"], tuple(e["ignore"]))
    z = np.load(out / rid / "vjepa_members.npz")
    upd = [tuple(int(x) for x in u) for u in z["upd"]]
    shown = shown_probs(live, events, upd, z["vjft_hand_k12_p10"], 1.0, 1.5, 0.01, first, pairs, R.DELAY)
    write_jsonl(dest / rid / "ar.jsonl", topk_rows(shown, actions, cfg["fps"], 5))
print("written", dest)
