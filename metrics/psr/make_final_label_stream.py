"""Per-frame label stream for feeding v14 from the final-label ensemble (3 V-JEPA 2 + VideoMAE + TCN), causally.
A final label exists only after its action's end is confirmed (ended_at) plus the ensemble pass (1.21 s, measured by the
teammate). So v14 gets the stream with a fixed delay D = 1.7 s (0.5 s confirmation + 1.21 s pass, rounded; fixed in advance):
at frame t it receives the label of frame f = t - D, which is
  the ensemble's final label of the detected action containing f, if that label existed at t (ended_at + 1.21 s <= t),
  else the TCN live label of frame f (known at f).
No future frames are used. Writes <out_dir>/<rid>/ar.jsonl for the 32 test recordings.
Run from system/assistant:  python ../../metrics/psr/make_final_label_stream.py <out_dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from assist.common import load_config, write_jsonl
from assist.industreal import SPLITS

PASS_S, DELAY = 1.21, 17                          # ensemble latency (s); stream delay (frames at 10 fps)
cfg = load_config(); out = cfg["paths"]["out"]; fps = cfg["fps"]
dest = Path(sys.argv[1])
for rid in SPLITS["test"]:
    tcn = [json.loads(l)["top"][0] for l in open(out / rid / "ar_live.jsonl", encoding="utf-8")]
    n = len(tcn)
    final = [None] * n                                # (label, conf, frame from which it exists)
    for r in (json.loads(l) for l in open(out / rid / "action_end.jsonl", encoding="utf-8")):
        ready = r["ended_at"] * fps + PASS_S * fps
        for f in range(round(r["start"] * fps), min(round(r["end"] * fps), n)):
            final[f] = (r["label"], r["conf"], ready)
    rows = []
    for t in range(n):
        f = t - DELAY
        if f < 0:
            top = ["background", 1.0]
        elif final[f] is not None and final[f][2] <= t:
            top = [final[f][0], final[f][1]]
        else:
            top = tcn[f]
        rows.append({"f": t, "t": round(t / fps, 2), "top": [top]})
    write_jsonl(dest / rid / "ar.jsonl", rows)
print("written", dest)
