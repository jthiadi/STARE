"""App data for the card demo = the reported row "v14 + TCN": v14's frozen test output (steps + its own mistake alerts,
eval_v14.py --ar predicted with the TCN actions; reproduced by metrics/reproduce_all.py) in the card UI's format, using
the teammate's own converter (scripts/mistakes/compare_v14.v14_engine_result). The actions shown are the TCN's (ar.jsonl).
Writes <app_dir>/<rid>/{engine_pred_v14.json, asd_v14.jsonl (v14's state, for the 3D inset), ar.jsonl, asd.jsonl} and
<app_dir>/check.json: the app's steps and alerts rescored, which must equal the table row.
Run from the teammate's checkout:  python build_v14_tcn.py <v14_predictedAR_test.json> <assist outputs> <app_dir>"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.engine.graph import load_tasks
from assist.evaluation.psr_eval import Data, score
from assist.industreal import SPLITS, n_frames, task_of
from assist.run_engine import run
from scripts.mistakes.compare_v9 import _mistakes, _summary, state_rows, step_entries
from scripts.mistakes.compare_v14 import v14_engine_result

saved_path, src, app = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
cfg = load_config(); tasks = load_tasks(cfg["graph_path"]); data = Data(cfg, "release"); rids = SPLITS["test"]
saved = json.loads(saved_path.read_text(encoding="utf-8")); vids = {v["video"]: v for v in saved["videos"]}
assert set(vids) == set(rids)
cfg["paths"]["out"] = app
proc = {q["description"]: q for q in data.proc_info}
for rid in rids:
    d = app / rid; d.mkdir(parents=True, exist_ok=True)
    for f in ("ar.jsonl", "asd.jsonl"):                      # TCN actions (v14's input), released ASD boxes (drawing only)
        shutil.copyfile(src / rid / f, d / f)
    shutil.copyfile(Path("output/assist") / rid / "hands.jsonl", d / "hands.jsonl")   # hand tracks (drawing; checkout only)
    frames = n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"])
    write_jsonl(d / "asd_v14.jsonl", state_rows(vids[rid]["predicted_steps"], list(tasks[task_of(rid)].initial_state),
                                                data.proc_info, frames, read_jsonl(d / "asd.jsonl")))
    base = run(rid, "pred", "asdv14", cfg, tasks)            # only its TCN action events are kept
    write_json(d / "engine_pred_v14.json", v14_engine_result(rid, vids[rid], tasks[task_of(rid)], frames / cfg["fps"], base,
                                                             proc, cfg["components"]))
check = {"table_row_v14_TCN": {k: saved["summary"][k] for k in ("all", "with_errors", "mistakes")},
         "app_steps_official_scoring": _summary(score(rids, data, lambda rid, v: step_entries(vids[rid]["predicted_steps"], data.proc_info))),
         "app_alerts_teammate_scoring": _mistakes("pred_v14", rids, cfg, tasks)}
write_json(app / "check.json", check)
print(json.dumps(check, indent=1))
