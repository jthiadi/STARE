"""Run the procedure engine on one or all test recordings, replaying the inputs in time order.

Input sources (``--actions`` x ``--state``):
  actions  gt    annotated actions (the "oracle")          pred  causal AR model -> event builder
           pred_vjepa  the same with online V-JEPA 2 labels (``assist.extract.run_ar_vjepa``)
           pred_vjepap the same with the V-JEPA 2 model trained on action beginnings (``--variant prefix``)
  state    gt    annotated PSR assembly state (2 fps)      asd   released ASD detector      none  no state
           asdours  our own ASD detector (assist/extract/train_asd.py; val + test)
           asdanom  the ASD rows with anomalous crops marked error_state (scripts/mistakes/state_anomaly.py)
           asdpart  each part's state read from the assembly crop (scripts/mistakes/part_state.py)

Writes ``output/assist/<rid>/engine_<actions>_<state>.json``: alerts (with explanations), the step
status log, the action events and the final checklist.

    python -m assist.run_engine --video all --actions gt --state gt
    python -m assist.run_engine --video 18_assy_2_5 --actions pred --state asd
"""
from __future__ import annotations

import argparse
import json

from assist.common import load_config, read_jsonl, write_json
from assist.engine.engine import STATUS_NAMES, Engine
from assist.engine.events import Event, build_events, events_from_segments
from assist.engine.graph import load_tasks
from assist.industreal import SPLITS, gt_actions, n_frames, psr_states, task_of

IMAGE_W, IMAGE_H = 1280, 720
AR_FILES = {"pred": "ar.jsonl", "pred_vjepa": "ar_vjepa.jsonl", "pred_vjepap": "ar_vjepa_prefix.jsonl"}
ASD_FILES = {"asd": "asd.jsonl", "asdours": "asd_ours.jsonl", "asdanom": "asd_anom.jsonl", "asdpart": "asd_part.jsonl",
             "asdv9": "asd_v9.jsonl", "asdv14": "asd_v14.jsonl"}
STATE_FPS = 2.0


def action_events(rid: str, source: str, cfg: dict) -> list[Event]:
    if source == "gt":
        return events_from_segments(gt_actions(rid, cfg["paths"]["labels"], cfg["fps"]))
    rows = read_jsonl(cfg["paths"]["out"] / rid / AR_FILES[source])
    e = cfg["events"]
    return build_events([(r["t"], *r["top"][0]) for r in rows], cfg["fps"], e["min_frames"], e["min_conf"],
                        tuple(e["ignore"]))


def state_stream(rid: str, source: str, cfg: dict, frames: int) -> list[tuple[float, object, float]]:
    """(time, state, confidence) observations."""
    if source == "none":
        return []
    if source in ASD_FILES:
        return [(r["t"], r["state"], r["conf"]) for r in read_jsonl(cfg["paths"]["out"] / rid / ASD_FILES[source])]
    changes = psr_states(rid, cfg["paths"]["labels"])
    out, k = [], 0
    step = round(cfg["fps"] / STATE_FPS)
    for f in range(0, frames, step):
        while k + 1 < len(changes) and changes[k + 1][0] <= f:
            k += 1
        out.append((f / cfg["fps"], changes[k][1], 1.0))
    return out


def hand_box_cover(hands: list, box: list[float]) -> float:
    """Share of the assembly box covered by the hands' bounding boxes (0..1)."""
    x, y, w, h = box
    covered = 0.0
    for hand in hands:
        xs = [p[0] * IMAGE_W for p in hand]
        ys = [p[1] * IMAGE_H for p in hand]
        ix = max(0.0, min(x + w, max(xs)) - max(x, min(xs)))
        iy = max(0.0, min(y + h, max(ys)) - max(y, min(ys)))
        covered += ix * iy
    return min(1.0, covered / max(w * h, 1.0))


def occlusion_stream(rid: str, cfg: dict, source: str = "asd") -> list[tuple[float, bool]]:
    """Every 0.5 s: do the hands cover the last seen assembly box?"""
    out_dir = cfg["paths"]["out"] / rid
    if not (out_dir / ASD_FILES[source]).is_file():
        return []
    boxes = read_jsonl(out_dir / ASD_FILES[source])
    hands = read_jsonl(out_dir / "hands.jsonl")
    step = round(cfg["fps"] / STATE_FPS)
    out, k, box = [], 0, None
    for row in hands[::step]:
        while k < len(boxes) and boxes[k]["t"] <= row["t"]:
            box = boxes[k]["box"]
            k += 1
        if box:
            out.append((row["t"], hand_box_cover(row["hands"], box) >= cfg["engine"]["hand_occlusion_iou"]))
    return out


def run(rid: str, actions: str, state: str, cfg: dict, tasks: dict, part: tuple | None = None,
        final: tuple | None = None) -> dict:
    """``part``: optional (rows of (time, P(wrong) per component, confidence), settings) for the combined check in
    ``scripts/mistakes/part_check.py``. ``final``: optional (rows of (time the label is ready, label, confidence),
    settings) for the action-end label rule in ``scripts/mistakes/end_label_check.py``."""
    task = tasks[task_of(rid)]
    frames = n_frames(rid, cfg["paths"]["labels"], cfg["paths"]["videos"])
    duration = frames / cfg["fps"]
    events = action_events(rid, actions, cfg)
    stream = [(e.ended_at, 0, "end", e) for e in events] + [(e.known_at, 1, "start", e) for e in events]
    stream += [(t, 2, "state", (s, c)) for t, s, c in state_stream(rid, state, cfg, frames)]
    if state in ASD_FILES:
        stream += [(t, 3, "hands", o) for t, o in occlusion_stream(rid, cfg, state)]
    if part:
        stream += [(t, 4, "part", (w, c)) for t, w, c in part[0]]
    if final:
        stream += [(t, 5, "final", (label, c)) for t, label, c in final[0]]
    eng = Engine(task, cfg["components"], {**cfg["engine"], "min_frames": cfg["events"]["min_frames"],
                                           "part_check": part[1] if part else None,
                                           "final_label": final[1] if final else None},
                 use_state=state != "none")
    for t, _, kind, x in sorted(stream, key=lambda r: (r[0], r[1])):
        if kind == "start":
            eng.on_action_start(t, x)
        elif kind == "end":
            eng.on_action_end(t, x)
        elif kind == "state":
            eng.on_state(t, *x)
        elif kind == "final":
            eng.on_final_label(t, *x)
        elif kind == "part":
            (eng.on_state_match if isinstance(x[0], dict) else eng.on_part_wrong)(t, *x)
        else:
            eng.on_hands(t, x)
    eng.finish(duration)
    return {"rid": rid, "task": task.name, "task_title": task.title, "actions": actions, "state": state,
            "duration_s": round(duration, 1), "alerts": eng.alerts, "log": eng.log,
            "steps": [{"id": s, "name": task.steps[s].name, "severity": task.steps[s].severity,
                       "requires": task.steps[s].requires, "subgoal": task.steps[s].subgoal,
                       "final": STATUS_NAMES[eng.status[s]]} for s in task.order],
            "events": [{"label": e.label, "start": round(e.start, 2), "end": round(e.end, 2),
                        "known_at": round(e.known_at, 2), "ended_at": round(e.ended_at, 2), "conf": e.conf}
                       for e in events]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="all", help='Recording id, or "all" (the test split).')
    parser.add_argument("--actions", choices=("gt", *AR_FILES), default="gt")
    parser.add_argument("--state", choices=("gt", *ASD_FILES, "none"), default="gt")
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    args = parser.parse_args()
    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    rids = SPLITS[args.split] if args.video == "all" else [args.video]
    for rid in rids:
        result = run(rid, args.actions, args.state, cfg, tasks)
        write_json(cfg["paths"]["out"] / rid / f"engine_{args.actions}_{args.state}.json", result)
        kinds = {}
        for a in result["alerts"]:
            kinds[a["type"]] = kinds.get(a["type"], 0) + 1
        print(json.dumps({"video": rid, "alerts": kinds}), flush=True)


if __name__ == "__main__":
    main()
