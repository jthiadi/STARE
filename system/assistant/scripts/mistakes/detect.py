"""IndustReal mistake detection on assembly install steps (independent of action segmentation).

Labels come from PSR (state -1 = incorrect install). Inputs for the appearance model are
precomputed per-frame features around the install frame (DINOv2 + VideoMAE-SSv2). The order
baseline reuses ``order_check`` preconditions. Action-order rarity uses GT ``AR_labels`` so this
side of the stack can run before predicted segments exist; swap in a predicted action list later.

    python -m scripts.mistakes.detect
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from monitor.settings import ROOT
from scripts.data.ar_labels import load_ar_intervals
from scripts.mistakes.events import FPS, LABELS, install_events, taxonomy
from scripts.mistakes.order_check import has_mistake, learn, violation
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL
from scripts.segmentation.train_ar_object_tcn import load_features

OUT = ROOT / "output/mistakes"
DINO = ROOT / "output/ar_frame_features"
VMAE = ROOT / "output/videomae_ssv2_features"
# Mistakes are usually logged during the following tighten_*; cover that span.
PRE_FRAMES = 8
POST_FRAMES = 48


def _feat_vector(x: np.ndarray, frame: int, end: int | None = None) -> np.ndarray:
    i = int(np.clip(frame, 0, len(x) - 1))
    j = int(np.clip(end if end is not None else frame + POST_FRAMES, 0, len(x) - 1))
    a, b = max(0, i - PRE_FRAMES), min(len(x), max(j, i) + 1)
    local = x[a:b].mean(axis=0)
    return np.concatenate([local, [i / max(len(x) - 1, 1), (j - i) / max(len(x), 1)]])


def _matrix(events: list[dict], n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    """One feature row per event; load each video's features once then free them."""
    by_vid: dict[str, list[dict]] = {}
    for e in events:
        by_vid.setdefault(e["vid"], []).append(e)
    xs, ys = [], []
    for vid, rows in by_vid.items():
        x = load_features(vid, DINO, progress=True, extra_dirs=[VMAE]).astype(np.float32)
        ar = load_ar_intervals(LABELS / vid / "AR_labels.csv", source_fps=FPS)
        for e in rows:
            t = e["frame"] / FPS
            # Extend the pool through the next tighten_* if it starts soon after the install.
            end = e["frame"] + POST_FRAMES
            for r in ar:
                if r["start_s"] + 1e-6 < t:
                    continue
                if r["start_s"] > t + 6.0:
                    break
                if r["action"].startswith("tighten"):
                    end = max(end, int(round(r["end_s"] * FPS)))
                    break
            one = np.zeros(n_comp, np.float32)
            one[e["comp"]] = 1.0
            xs.append(np.concatenate([_feat_vector(x, e["frame"], end), one]))
            ys.append(1.0 if e["mistake"] else 0.0)
        del x
    return np.stack(xs), np.asarray(ys, np.float32)


def _fit_logistic(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """L2 logistic regression via gradient descent (no sklearn dependency). Pack: mean|std|w|b."""
    rng = np.random.default_rng(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    w = rng.normal(0, 0.01, z.shape[1]).astype(np.float64)
    b = 0.0
    pos, neg = max(float(y.sum()), 1.0), max(float((1 - y).sum()), 1.0)
    weight = np.where(y > 0.5, neg / pos, 1.0)
    lr, l2 = 0.05, 1e-2
    for _ in range(800):
        logit = z @ w + b
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        err = (p - y) * weight
        w -= lr * (z.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return np.concatenate([mean, std, w, [b]])


def _predict_logistic(pack: np.ndarray, x: np.ndarray) -> np.ndarray:
    d = (len(pack) - 1) // 3
    mean, std, w = pack[:d], pack[d:2 * d], pack[2 * d:3 * d]
    b = pack[-1]
    z = (x - mean) / std
    logit = z @ w + b
    return 1 / (1 + np.exp(-np.clip(logit, -30, 30)))


def _metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"threshold": round(float(thr), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "mistakes": int(y.sum()), "flagged": int(pred.sum())}


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(y.sum()), 1.0)
    # Step-wise AP over recall
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_r)
        prev_r = float(r)
    return ap


def _pick_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Prefer precision>=0.2 then max F1; else best precision among recall>=0.2; else median score."""
    cands = [_metrics(y, score, float(thr)) | {"_thr": float(thr)} for thr in np.unique(score)]
    usable = [m for m in cands if m["precision"] >= 0.2]
    if usable:
        return max(usable, key=lambda m: (m["f1"], m["precision"], m["recall"]))["_thr"]
    soft = [m for m in cands if m["recall"] >= 0.2]
    if soft:
        return max(soft, key=lambda m: (m["precision"], m["f1"], m["recall"]))["_thr"]
    return float(np.median(score))


def order_flags(events: list[dict], rules) -> np.ndarray:
    return np.array([violation(rules, {"comp": e["comp"], "kind": "install", "before": e["before_state"]})
                     is not None for e in events], dtype=bool)


def action_rarity(events: list[dict], pair_prob: dict[tuple[str, str], float],
                  unk: float) -> np.ndarray:
    """Score = -log p(prev -> next) around the install using GT AR (lower rarity = more normal)."""
    scores = []
    for e in events:
        ar = load_ar_intervals(LABELS / e["vid"] / "AR_labels.csv", source_fps=FPS)
        t = e["frame"] / FPS
        seq = [r["action"] for r in ar if r["end_s"] <= t + 1e-6]
        if len(seq) < 2:
            scores.append(unk)
            continue
        a, b = seq[-2], seq[-1]
        scores.append(-np.log(pair_prob.get((a, b), unk)))
    return np.asarray(scores, np.float64)


def learn_pair_prob(vids: list[str], alpha: float = 0.5) -> tuple[dict, float]:
    from collections import Counter
    pairs, nxt = Counter(), Counter()
    for vid in vids:
        seq = [r["action"] for r in load_ar_intervals(LABELS / vid / "AR_labels.csv", source_fps=FPS)]
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += 1
            nxt[a] += 1
    vocab = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(vocab)
    prob = {(a, b): (pairs[(a, b)] + alpha) / (nxt[a] + alpha * n) for a in vocab for b in vocab
            if nxt[a]}
    # Also keep observed only denser dict for lookup; unseen handled by unk
    observed = {(a, b): (pairs[(a, b)] + alpha) / (nxt[a] + alpha * n) for a, b in pairs}
    unk = alpha / (alpha * n + 1)
    return observed, unk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    events = install_events()
    analysis = taxonomy(events)
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    train_e = [e for e in events if e["vid"] in TRAIN]
    val_e = [e for e in events if e["vid"] in VAL]
    test_e = [e for e in events if e["vid"] in TEST]

    n_comp = 1 + max(e["comp"] for e in events)
    x_tr, y_tr = _matrix(train_e, n_comp)
    x_va, y_va = _matrix(val_e, n_comp)
    x_te, y_te = _matrix(test_e, n_comp)
    pack = _fit_logistic(x_tr, y_tr, args.seed)
    s_va = _predict_logistic(pack, x_va)
    s_te = _predict_logistic(pack, x_te)
    thr = _pick_threshold(y_va, s_va)

    clean = [v for v in TRAIN if not has_mistake(v)]
    rules = learn(clean)
    pair_prob, unk = learn_pair_prob(clean)

    def bundle(split_events, y, appearance_score):
        ord_f = order_flags(split_events, rules)
        rare = action_rarity(split_events, pair_prob, unk)
        return {
            "appearance_ap": round(_average_precision(y, appearance_score), 4),
            "appearance": _metrics(y, appearance_score, thr),
            "appearance_top10": _metrics(
                y, appearance_score, float(np.partition(appearance_score, max(len(appearance_score) - 10, 0))
                                          [max(len(appearance_score) - 10, 0)])
                if len(appearance_score) else 1.0),
            "order_precondition": _metrics(y, ord_f.astype(np.float64), 0.5),
            "appearance_or_order": _metrics(
                y, np.maximum(appearance_score, ord_f.astype(np.float64)), thr),
            "rarity_scores": {
                "mistake_mean": round(float(rare[y > 0.5].mean()) if (y > 0.5).any() else 0, 3),
                "correct_mean": round(float(rare[y < 0.5].mean()) if (y < 0.5).any() else 0, 3),
            },
        }

    # Choose rarity threshold on val (flag high rarity)
    rare_va = action_rarity(val_e, pair_prob, unk)
    best_r_f1, best_r_rec, thr_r = -1.0, -1.0, float(np.median(rare_va))
    for cand in np.unique(rare_va):
        m = _metrics(y_va, rare_va, float(cand))
        if (m["f1"], m["recall"]) > (best_r_f1, best_r_rec):
            best_r_f1, best_r_rec, thr_r = m["f1"], m["recall"], float(cand)

    def with_rarity(split_events, y, appearance_score):
        base = bundle(split_events, y, appearance_score)
        rare = action_rarity(split_events, pair_prob, unk)
        base["action_transition_rarity"] = _metrics(y, rare, thr_r)
        base["any_of_three"] = _metrics(
            y, np.maximum(np.maximum(appearance_score, order_flags(split_events, rules).astype(float)),
                          (rare >= thr_r).astype(float)), thr)
        return base

    report = {
        "task": "Detect incorrect IndustReal installs (PSR state -1) at GT install times",
        "inputs": "DINOv2 + VideoMAE-SSv2 window features; order rules from error-free train; "
                  "AR transition rarity from GT AR_labels (replaceable by predicted actions)",
        "analysis": analysis,
        "val_threshold_appearance": thr,
        "val_threshold_rarity": thr_r,
        "val": with_rarity(val_e, y_va, s_va),
        "test": with_rarity(test_e, y_te, s_te),
        "details_test": [
            {"vid": e["vid"], "frame": e["frame"], "name": e["name"], "mistake": e["mistake"],
             "appearance": round(float(s), 4),
             "order": bool(o),
             "rarity": round(float(r), 3),
             "flagged": bool(s >= thr or o or r >= thr_r)}
            for e, s, o, r in zip(
                test_e, s_te, order_flags(test_e, rules), action_rarity(test_e, pair_prob, unk))
            if e["mistake"] or s >= thr or o or r >= thr_r
        ],
    }
    (OUT / "detect.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    slim = {k: v for k, v in report.items() if k not in ("details_test", "analysis")}
    slim["analysis"] = analysis
    slim["val"] = {k: ({kk: vv for kk, vv in v.items() if kk != "details"} if isinstance(v, dict) else v)
                   for k, v in report["val"].items()}
    slim["test"] = report["test"]
    print(json.dumps({
        "analysis": analysis,
        "val": report["val"],
        "test": {k: v for k, v in report["test"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
