"""Frame-wise IndustReal actions (verb_noun) from official-protocol verb TCNs x noun TCNs.

score(verb_noun) = log P(verb) + log P(noun); background = log P(no verb) + log P(no noun) + bias.
Only the 74 verb_noun pairs of the official action vocabulary are allowed. P(verb) / P(noun) are
weighted mixes of the given TCN runs. Everything tunable (mix weights, switch penalty, background
bias, weight of the action-order prior learned from the training GT) is chosen on the 16 val videos;
the 32 test videos are scored once (standard TAS metrics, see ``scripts/segmentation/tas_metrics.py``).
The chosen decoder is written to ``decoder.json`` so ``scripts/predict_actions.py`` can reuse it.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from monitor.settings import ROOT
from scripts.segmentation.evaluate_ar_object_segments import gt_noun_from_action
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL
from scripts.data.tas_labels import frame_labels, load_mapping, read_segments
from scripts.segmentation.tas_metrics import score

EVAL = ROOT / "output/official_split_eval"


def run_classes(run: Path) -> list[str]:
    report = json.loads((run / "ar_object_paper_metrics.json").read_text(encoding="utf-8"))
    if "classes" in report:
        return report["classes"]
    summary = json.loads((EVAL / "result_tcn_v7_tcn_v8/summary.json").read_text(encoding="utf-8"))
    return summary["classes"]


def joint_scores(pv: np.ndarray, pn: np.ndarray, verbs: list[str], nouns: list[str],
                 actions: list[str]) -> np.ndarray:
    """(T, len(actions)) log scores; actions[0] is background."""
    lv, ln = np.log(pv + 1e-6), np.log(pn + 1e-6)
    out = np.empty((len(pv), len(actions)), np.float32)
    out[:, 0] = lv[:, 0] + ln[:, 0]
    for j, action in enumerate(actions[1:], 1):
        verb, noun = action.split("_", 1)[0], gt_noun_from_action(action)
        out[:, j] = lv[:, verbs.index(verb)] + ln[:, nouns.index(noun)]
    return out


def learn_transitions(gts: list[np.ndarray], k: int, alpha: float = 0.1) -> np.ndarray:
    """log P(next action | action) * k-normalised (uniform = 0), from segment order in training GT."""
    counts = np.full((k, k), alpha)
    for gt in gts:
        seg = gt[np.r_[0, np.flatnonzero(np.diff(gt)) + 1]]
        np.add.at(counts, (seg[:-1], seg[1:]), 1)
    np.fill_diagonal(counts, 0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    trans = np.log(probs * k + 1e-9)
    np.fill_diagonal(trans, 0.0)
    return trans


def viterbi(scores: np.ndarray, switch: float, trans: np.ndarray | None = None,
            trans_weight: float = 0.0) -> np.ndarray:
    """Max-score label path: every label change costs ``switch`` minus the weighted order prior."""
    t, k = scores.shape
    if switch <= 0 and trans_weight <= 0:
        return scores.argmax(axis=1)
    back = np.empty((t, k), np.int16)
    cur = scores[0].astype(np.float64)
    if trans_weight > 0:
        cost = trans_weight * trans - switch
        np.fill_diagonal(cost, 0.0)
        for i in range(1, t):
            cand = cur[:, None] + cost
            back[i] = cand.argmax(axis=0)
            cur = cand[back[i], np.arange(k)] + scores[i]
    else:
        labels = np.arange(k)
        for i in range(1, t):
            best = int(cur.argmax())
            jump = cur[best] - switch
            stay = cur >= jump
            back[i] = np.where(stay, labels, best)
            cur = np.where(stay, cur, jump) + scores[i]
    path = np.empty(t, np.int64)
    path[-1] = int(cur.argmax())
    for i in range(t - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(scores: np.ndarray, switch: float, bias: float, trans=None, trans_weight: float = 0.0):
    s = scores.copy()
    s[:, 0] += bias
    pred = viterbi(s, switch, trans, trans_weight)
    prob = np.exp(s - s.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    return pred, prob[np.arange(len(pred)), pred]


def clip_accuracy(preds: dict) -> float:
    hit = total = 0
    for p in preds.values():
        pred, gt = p["pred"], p["gt"]
        start = 0
        for i in range(1, len(gt) + 1):
            if i == len(gt) or gt[i] != gt[start]:
                if gt[start] != 0:
                    total += 1
                    hit += Counter(pred[start:i].tolist()).most_common(1)[0][0] == gt[start]
                start = i
    return 100 * hit / total


def given_boundary_top1(joint: dict, rids: list[str], actions: list[str]) -> tuple[float, int]:
    """IndustReal recognition protocol: each annotated segment, cut at its GT start/end, gets the action
    (background excluded) with the highest mean probability inside it."""
    index = {a: i for i, a in enumerate(actions)}
    hit = total = 0
    for rid in rids:
        s = joint[rid][:, 1:]
        for action, start, end in read_segments(rid):
            start, end = max(0, start), min(len(s) - 1, end)
            if end < start:
                continue
            w = s[start:end + 1]
            p = np.exp(w - w.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            hit += int(p.mean(axis=0).argmax()) + 1 == index[action]
            total += 1
    return 100 * hit / total, total


def _simplex(n: int, step: float = 0.25):
    m = int(round(1 / step))
    for combo in itertools.product(range(m + 1), repeat=n):
        if sum(combo) == m:
            yield tuple(c * step for c in combo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verb-runs", type=Path, nargs="+", default=[EVAL / "tcn_verb"])
    parser.add_argument("--noun-runs", type=Path, nargs="+", default=[EVAL / "tcn_v7", EVAL / "tcn_v8"])
    parser.add_argument("--output", type=Path, default=ROOT / "output/actions/verb_noun")
    args = parser.parse_args()

    actions = load_mapping()
    verbs, nouns = run_classes(args.verb_runs[0]), run_classes(args.noun_runs[0])
    assert all(run_classes(r) == verbs for r in args.verb_runs)
    assert all(run_classes(r) == nouns for r in args.noun_runs)
    missing = {a for a in actions[1:] if a.split("_", 1)[0] not in verbs or gt_noun_from_action(a) not in nouns}
    assert not missing, f"actions not covered by verb/noun classes: {sorted(missing)}"

    # Training videos only contribute segment order, so any length past the last action works.
    trans = learn_transitions([frame_labels(r, max(e for _, _, e in read_segments(r)) + 2, actions)
                               for r in TRAIN], len(actions))
    data = {}
    for rid in VAL + TEST:
        pv = [np.load(r / rid / "test_probs.npy") for r in args.verb_runs]
        pn = [np.load(r / rid / "test_probs.npy") for r in args.noun_runs]
        n = min(len(p) for p in pv + pn)
        data[rid] = ([p[:n] for p in pv], [p[:n] for p in pn], frame_labels(rid, n, actions))

    def scores_for(rid, wv, wn):
        pv, pn, _ = data[rid]
        return joint_scores(sum(w * p for w, p in zip(wv, pv)), sum(w * p for w, p in zip(wn, pn)),
                            verbs, nouns, actions)

    mixes = []
    for wv in _simplex(len(args.verb_runs)):
        for wn in _simplex(len(args.noun_runs)):
            ok = tot = 0
            for rid in VAL:
                pred = scores_for(rid, wv, wn).argmax(axis=1)
                ok += int((pred == data[rid][2]).sum())
                tot += len(pred)
            mixes.append({"verb_weights": wv, "noun_weights": wn, "val_frame_acc": round(100 * ok / tot, 2)})
    mix = max(mixes, key=lambda r: r["val_frame_acc"])
    print(json.dumps({"mix_top5": sorted(mixes, key=lambda r: -r["val_frame_acc"])[:5]}), flush=True)
    joint = {rid: scores_for(rid, mix["verb_weights"], mix["noun_weights"]) for rid in VAL + TEST}

    def run(rids, switch, bias, tw):
        out = {}
        for rid in rids:
            pred, conf = decode(joint[rid], switch, bias, trans, tw)
            out[rid] = {"pred": pred, "conf": conf, "gt": data[rid][2]}
        return out

    tuning = []
    for tw in (0.0, 0.5, 1.0, 2.0):
        for switch in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            for bias in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
                m = score(run(VAL, switch, bias, tw), include_bg=True)
                key = (m["F1@10"] + m["F1@25"] + m["F1@50"] + m["Edit"] + m["Acc"]) / 5
                tuning.append({"trans_weight": tw, "switch": switch, "bias": bias, "key": round(key, 2),
                               **{k: round(v, 2) for k, v in m.items()}})
                print(json.dumps(tuning[-1]), flush=True)
    best = max(tuning, key=lambda r: r["key"])

    preds = run(TEST, best["switch"], best["bias"], best["trans_weight"])
    assy = {r: p for r, p in preds.items() if "_assy_" in r}
    main_ = {r: p for r, p in preds.items() if "_main_" in r}
    top1, segments = given_boundary_top1(joint, TEST, actions)
    report = {
        "method": "DINOv2 + TCN: verb TCNs x noun TCNs -> 74 official verb_noun actions + background",
        "protocol": "train 36 official train videos, choose mix + decoding on 16 val, test on 32 test (full untrimmed videos)",
        "gt_used_at_inference": False,
        "verb_runs": [str(r) for r in args.verb_runs],
        "noun_runs": [str(r) for r in args.noun_runs],
        "chosen_mix_on_val": mix,
        "chosen_on_val": best,
        "test_with_background": score(preds, include_bg=True),
        "test_without_background": score(preds, include_bg=False),
        "test_assy_with_background": score(assy, include_bg=True),
        "test_main_with_background": score(main_, include_bg=True),
        "test_clip_top1_majority_in_gt_segment": clip_accuracy(preds),
        "test_top1_given_gt_boundaries": top1,
        "test_segments_given_gt_boundaries": segments,
        "per_video_acc_with_background": {
            r: round(100 * float((p["pred"] == p["gt"]).mean()), 1) for r, p in preds.items()},
        "mix_tuning": mixes,
        "val_tuning": tuning,
        "index2label": {i: a for i, a in enumerate(actions)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    decoder = {"actions": actions, "verbs": verbs, "nouns": nouns,
               "verb_runs": [r.name for r in args.verb_runs], "noun_runs": [r.name for r in args.noun_runs],
               "verb_weights": list(mix["verb_weights"]), "noun_weights": list(mix["noun_weights"]),
               "switch": best["switch"], "bias": best["bias"], "trans_weight": best["trans_weight"],
               "transitions": np.round(trans, 4).tolist()}
    (args.output / "decoder.json").write_text(json.dumps(decoder), encoding="utf-8")
    np.savez(args.output / "test_predictions.npz", actions=np.array(actions),
             **{f"{r}/{k}": v for r, p in preds.items() for k, v in p.items()})
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("chosen_mix_on_val", "chosen_on_val", "test_with_background",
                                             "test_without_background",
                                             "test_clip_top1_majority_in_gt_segment",
                                             "test_top1_given_gt_boundaries")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
