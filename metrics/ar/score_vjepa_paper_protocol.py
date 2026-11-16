"""AR on the IndustReal paper's protocol from the saved per-segment outputs (no model run):
each annotated segment (true start / end) classified among the 74 actions, background excluded, top-1.
Scores every saved V-JEPA 2 run (output/official_split_eval/vjepa2/clip_probs_*.npz) on val and test, next to
the val / test numbers the training run recorded (its .json), and the TCN check from ar.jsonl (top-5 kept)."""
import glob, json, os
import numpy as np
D = "output/official_split_eval/vjepa2"
rows = []
for f in sorted(glob.glob(f"{D}/clip_probs_*.npz")):
    name = os.path.basename(f)[len("clip_probs_"):-4]
    z = np.load(f)
    key = next((k for k in z.files if k.endswith("/test") and not k.startswith(("keys", "labels"))), None)
    if key is None: continue
    m = key.split("/")[0]
    acc = {s: round(100 * float((z[f"{m}/{s}"].argmax(1) + 1 == z[f"labels/{s}"]).mean()), 2) for s in ("val", "test")}   # labels count from 1
    rec = {}
    if os.path.exists(f"{D}/{name}.json"):
        j = json.load(open(f"{D}/{name}.json")); rec = {k: j.get(k) for k in ("val_top1", "test_top1") if k in j}
    rows.append((name, z[f"labels/test"].shape[0], acc, rec))
for r in rows: print(f"{r[0]:28s} test segs {r[1]}  val {r[2]['val']:5.2f}  test {r[2]['test']:5.2f}   recorded {r[3]}")
