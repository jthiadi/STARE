"""Render the teammate's card UI (their renderer, unchanged: card, progress ring, checklist, timeline, 3D "Recognised state"
inset) to mp4 for the 32 test recordings, showing the reported system "v14 + TCN": the app data made by build_v14_tcn.py
(engine_pred_v14.json = v14's steps and its own mistake alerts, TCN actions; asd_v14.jsonl = v14's state for the 3D inset).
The ensemble's final-label chip is left out: it is not part of v14 + TCN.
Run from the teammate's checkout (needs Windows fonts, the CAD part_geometries, the videos and hands.jsonl):
    python render_cards.py <out_dir> [rid ...] [--outputs <app_dir from build_v14_tcn.py>] [--clip SECONDS]"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import assist.tools.overlay_cards as OC
from assist.common import load_config
from assist.industreal import SPLITS

RUN = "pred_v14"
from assist.run_engine import ASD_FILES
ASD_FILES["v14"] = "asd_v14.jsonl"          # v14's own state (3D inset)
args = sys.argv[1:]
opts = {}
for key in ("--clip", "--outputs"):
    if key in args:
        k = args.index(key); opts[key] = args[k + 1]; del args[k:k + 2]
out = Path(args[0]).resolve(); rids = args[1:] or SPLITS["test"]
OC.ROOT = Path(out.anchor)                   # their end-of-video print shows the path relative to ROOT
cfg = load_config()
local_out = cfg["paths"]["out"]
if "--outputs" in opts:                       # our outputs; hands.jsonl (UI only, not in the submission) from the checkout
    cfg["paths"]["out"] = Path(opts["--outputs"]).resolve()
    _read = OC.read_jsonl
    OC.read_jsonl = lambda p: _read(local_out / Path(p).parent.name / "hands.jsonl" if Path(p).name == "hands.jsonl" else p)
clip = float(opts["--clip"]) if "--clip" in opts else None
for rid in rids:
    dest = out / f"{rid}.mp4"
    if dest.exists() and clip is None:
        continue
    t0 = time.time()
    OC.render(rid, RUN, cfg, dest, cad=True, action_end=False, stop=clip)
    print(json.dumps({"rid": rid, "seconds": round(time.time() - t0, 1)}), flush=True)
