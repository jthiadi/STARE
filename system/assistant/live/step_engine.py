"""Step 5 of the on-demand pipeline: hands (their MediaPipe code) + the procedure engine with the newest live actions
(``pred_vjepap`` = ar_vjepa_prefix.jsonl, step 2) and v14's live steps (step 4) as its state, as compare_v9 / compare_v14 do.
Writes output/assist/<rid>/hands.jsonl (if missing), asd_v14.jsonl and engine_pred_vjepap_asdv14.json.
    python -m live.step_engine <rid>"""
import json
import sys
import time

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.engine.graph import load_tasks
from assist.evaluation.psr_eval import Data
from assist.extract.run_hands import compute as hands_compute, rows as hands_rows
from assist.industreal import n_frames, task_of
from assist.run_engine import run
from scripts.mistakes.compare_v9 import state_rows

ACTIONS, STATE = "pred_vjepap", "asdv14"


def main(rid: str) -> dict:
    cfg = load_config(); tasks = load_tasks(cfg["graph_path"]); proc_info = Data(cfg, "release").proc_info
    out = cfg["paths"]["out"] / rid
    t0 = time.time()
    if not (out / "hands.jsonl").exists():
        write_jsonl(out / "hands.jsonl", hands_rows(hands_compute(cfg["paths"]["videos"] / f"{rid}.mp4"), cfg["fps"]))
    t_hands = time.time() - t0
    v14 = json.loads((out / "v14_live.json").read_text(encoding="utf-8"))
    boxes = read_jsonl(out / "asd.jsonl") if (out / "asd.jsonl").exists() else []
    frames = n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"])
    write_jsonl(out / "asd_v14.jsonl", state_rows(v14["predicted_steps"], list(tasks[task_of(rid)].initial_state),
                                                  proc_info, frames, boxes))
    res = run(rid, ACTIONS, STATE, cfg, tasks)
    write_json(out / f"engine_{ACTIONS}_{STATE}.json", res)
    rep = {"video": rid, "hands_s": round(t_hands, 1), "seconds": round(time.time() - t0, 1),
           "alerts": {k: sum(1 for a in res["alerts"] if a["type"] == k) for k in sorted({a["type"] for a in res["alerts"]})}}
    print(json.dumps(rep), flush=True)
    return rep


if __name__ == "__main__":
    main(sys.argv[1])
