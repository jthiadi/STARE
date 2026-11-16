"""AR metrics of the live action labels on the 32 IndustReal test recordings (and val), for any label file:
  frame accuracy       top-1 per frame vs the annotated action (background counted)
  end accuracy         top-1 at the last frame of each annotated action (the teammate's "label right at each action's end")
  segment top-1        the IndustReal paper's protocol on the same labels: each annotated segment (true start / end) gets
                       the action with the highest summed top-5 probability inside it, background excluded (approximate:
                       only the top-5 per frame are stored)
  detected actions     action-end labels (action_end.jsonl): a true action is right if the detection overlapping it most
                       has its label (their best_overlap_accuracy) -- the teammate's 55.46 % row
Run from system/assistant:   python ../../metrics/ar/score_live_labels.py <split> <label file> [<label file> ...]
  e.g. test ar_live.jsonl ar_vjepa_prefix.jsonl ar.jsonl        (+ action_end.jsonl is scored when present)"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from assist.common import load_config
from assist.extract.run_action_end import best_overlap_accuracy
from assist.industreal import SPLITS
from scripts.data.tas_labels import load_mapping

cfg = load_config(); OUT = cfg["paths"]["out"]; LAB = cfg["paths"]["labels"]; IDX = {a: i for i, a in enumerate(load_mapping())}


def gt(rid, n):
    g, segs = ["background"] * n, []
    for r in csv.reader(open(LAB / rid / "AR_labels.csv", encoding="utf-8")):
        a, s, e = r[2], int(Path(r[3]).stem), int(Path(r[4]).stem)
        segs.append((a, s, min(e, n - 1)))
        for f in range(s, min(e, n - 1) + 1):
            g[f] = a
    return g, segs


def score(split, name):
    fr = fn = end_ok = end_n = seg_ok = seg_n = 0
    for rid in SPLITS[split]:
        rows = [json.loads(l) for l in open(OUT / rid / name, encoding="utf-8")]
        n = len(rows); g, segs = gt(rid, n)
        top = [r["top"][0][0] for r in rows]
        fr += sum(a == b for a, b in zip(top, g)); fn += n
        for a, s, e in segs:
            end_n += 1; end_ok += top[e] == a
            acc = defaultdict(float)
            for f in range(s, e + 1):
                for nm, p in rows[f]["top"]:
                    if nm != "background":
                        acc[nm] += p
            seg_n += 1; seg_ok += bool(acc) and max(acc, key=acc.get) == a
    return {"file": name, "split": split, "frame_acc": round(100 * fr / fn, 2), "end_acc": round(100 * end_ok / end_n, 2),
            "segment_top1_paper_protocol_approx": round(100 * seg_ok / seg_n, 2), "segments": seg_n, "frames": fn}


def detected(split):
    right = total = 0
    for rid in SPLITS[split]:
        f = OUT / rid / "action_end.jsonl"
        if not f.exists():
            return None
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        spans = [(round(r["start"] * cfg["fps"]), round(r["end"] * cfg["fps"]) - 1) for r in rows]
        labels = [IDX[r["label"]] for r in rows]
        n = sum(1 for _ in open(OUT / rid / "ar_vjepa_prefix.jsonl", encoding="utf-8"))
        segs = [(s, e, IDX[a]) for a, s, e in gt(rid, n)[1] if a in IDX]
        r, t = best_overlap_accuracy(spans, labels, segs); right += r; total += t
    return {"file": "action_end.jsonl", "split": split, "true_actions_labelled_right_detected": round(100 * right / total, 2),
            "true_actions": total}


if __name__ == "__main__":
    split, files = sys.argv[1], sys.argv[2:]
    res = [score(split, f) for f in files]
    d = detected(split)
    if d:
        res.append(d)
    for r in res:
        print(json.dumps(r))
    out = Path(__file__).resolve().parent / "results" / f"live_labels_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
