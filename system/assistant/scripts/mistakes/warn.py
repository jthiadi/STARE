"""Pre-action mistake warnings on IndustReal, isolating the mistake logic.

Placeholders (so only the warning rules are scored):
  * GT PSR state just before each action
  * GT action onset from ``AR_labels.csv``

Two branches (PREGO-style procedure + execution prototypes):
  1. ``procedure`` — official SOP partial order + learned install preconditions. Warns when the
     worker starts an action that implies installing a component whose prerequisites are missing.
  2. ``execution`` — DINO window features at action onset, compared to train exemplars of
     correct (0→1) vs incorrect (0→-1) installs for the inferred component. Catches bad
     placements that are still *in order* (most IndustReal mistakes).

Groq LLM was tried as a third branch; the API returns 403 from this host, so it is skipped and
noted in the report. Local Qwen3-VL weights are not present under ``models/``.

    python -m scripts.mistakes.warn
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from monitor.readiness import EXPECTED_PREREQUISITES, OFFICIAL_STEPS
from monitor.settings import ROOT
from scripts.data.ar_labels import load_ar_intervals
from scripts.mistakes.order_check import has_mistake, learn, names, states, steps, violation
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

LABELS = ROOT / "data/evaluation/official"
OUT = ROOT / "output/mistakes"
FPS = 10.0
DINO = ROOT / "output/ar_frame_features"
COMPONENTS = [
    "base", "front_chassis", "front_chassis_pin", "rear_chassis", "short_rear_chassis",
    "front_rear_chassis_pin", "rear_rear_chassis_pin", "front_bracket", "front_bracket_screw",
    "front_wheel_assy", "rear_wheel_assy",
]
# official step id -> PSR component index
STEP_COMP = {sid: vals[1] for sid, vals in OFFICIAL_STEPS.items()}
COMP_STEP = {comp: sid for sid, comp in STEP_COMP.items()}

# Rough AR → component candidates (execution / procedure targeting).
ACTION_COMPS: dict[str, list[int]] = {
    "take_wheel": [9, 10], "fit_wheel": [9, 10], "put_wheel": [9, 10], "plug_wheel": [9, 10],
    "pull_wheel": [9, 10],
    "tighten_acorn_nut": [9, 10], "take_acorn_nut": [9, 10], "fit_acorn_nut": [9, 10],
    "tighten_nut": [2, 5, 6, 8, 9, 10], "fit_nut": [2, 5, 6, 8, 9, 10], "take_nut": [2, 5, 6, 8],
    "plug_short_pin": [2], "take_pin_short": [2],
    "plug_pin_middle": [5], "take_pin_middle": [5],
    "plug_pin_long": [6], "take_pin_long": [6], "put_pin_long": [6],
    "plug_screw_pin": [8], "take_screw_pin": [8], "put_screw_pin": [8],
}


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def state_before(vid: str, frame: int) -> tuple[int, ...]:
    seq = states(vid)
    before = seq[0][1]
    for f, st in seq:
        if f >= frame:
            break
        before = st
    return before


def present(st: tuple[int, ...], idx: int) -> bool:
    return st[idx] != 0


def prereq_ok(st: tuple[int, ...], step_id: str) -> bool:
    for pre in EXPECTED_PREREQUISITES[step_id]:
        if not present(st, STEP_COMP[pre]):
            return False
    return True


def action_events(vids: list[str]) -> list[dict]:
    """One event per GT AR action onset, with GT state-before and mistake linkage."""
    mist_by_vid = {v: [s for s in steps(v) if s["mistake"]] for v in vids}
    out = []
    for vid in vids:
        ar = load_ar_intervals(LABELS / vid / "AR_labels.csv", source_fps=FPS)
        for r in ar:
            start_f = int(round(r["start_s"] * FPS))
            end_f = int(round(r["end_s"] * FPS))
            st = state_before(vid, start_f)
            # Link to a real mistake if the mistake PSR change falls inside this action
            # (or within 0.5 s after it ends — tighten often finishes just as state flips).
            linked = None
            for m in mist_by_vid[vid]:
                if start_f - 5 <= m["frame"] <= end_f + 5:
                    linked = m
                    break
            out.append({
                "vid": vid, "action": r["action"], "start_f": start_f, "end_f": end_f,
                "state": st, "mistake": linked is not None,
                "mistake_comp": None if linked is None else linked["comp"],
                "mistake_name": None if linked is None else names(vid).get(linked["frame"]),
            })
    return out


def infer_targets(action: str, st: tuple[int, ...]) -> list[int]:
    """Components this action might be installing: still absent, matching the action lexicon."""
    cands = ACTION_COMPS.get(action, [])
    return [c for c in cands if st[c] == 0 and c in COMP_STEP]


def procedure_warn(action: str, st: tuple[int, ...], rules) -> tuple[bool, str]:
    """Warn only when every plausible install target for this action is SOP-illegal right now.

    Learned pair-order rules are *not* OR-ed in here: they fire too often on legal alternate
    orders (IndustReal allows partial-order assembly). Kept available via ``rules`` for ablations.
    """
    del rules  # reserved for ablation hooks
    targets = infer_targets(action, st)
    if not targets:
        return False, ""
    blocked = []
    for comp in targets:
        sid = COMP_STEP[comp]
        if not prereq_ok(st, sid):
            missing = [p for p in EXPECTED_PREREQUISITES[sid] if not present(st, STEP_COMP[p])]
            blocked.append(f"{sid}/{COMPONENTS[comp]} needs {[COMPONENTS[STEP_COMP[p]] for p in missing]}")
    # All candidates illegal → this action cannot be a legal install right now.
    if blocked and len(blocked) == len(targets):
        return True, "; ".join(blocked)
    return False, ""


def _dino_mean(vid: str, start_f: int, end_f: int, cache: dict) -> np.ndarray | None:
    path = DINO / f"{vid}.npz"
    if not path.is_file():
        return None
    if vid not in cache:
        cache[vid] = np.load(path)["feats"].astype(np.float32).mean(axis=1)  # (T, 768) over views
    x = cache[vid]
    a = int(np.clip(start_f, 0, len(x) - 1))
    # Early window only (warn while the action is starting), ~1 s.
    b = int(np.clip(min(end_f, start_f + 10), a + 1, len(x)))
    return x[a:b].mean(axis=0)


def build_execution_prototypes(train_vids: list[str]) -> dict[int, dict[str, np.ndarray]]:
    """Per-component mean DINO of correct vs incorrect install completion moments (train)."""
    cache: dict[str, np.ndarray] = {}
    buckets: dict[int, dict[str, list]] = defaultdict(lambda: {"ok": [], "bad": []})
    for vid in train_vids:
        for s in steps(vid):
            if s["kind"] != "install":
                continue
            key = "bad" if s["mistake"] else "ok"
            vec = _dino_mean(vid, max(0, s["frame"] - 5), s["frame"] + 5, cache)
            if vec is not None:
                buckets[s["comp"]][key].append(vec)
        if vid in cache:
            del cache[vid]
    out = {}
    for comp, parts in buckets.items():
        proto = {}
        if parts["ok"]:
            proto["ok"] = np.stack(parts["ok"]).mean(0)
        if parts["bad"]:
            proto["bad"] = np.stack(parts["bad"]).mean(0)
        if proto:
            out[comp] = proto
    return out


def execution_warn(action: str, st: tuple[int, ...], vid: str, start_f: int, end_f: int,
                   protos: dict, cache: dict, margin: float) -> tuple[bool, str, float]:
    """Warn if early-action appearance is closer to incorrect-install prototypes than correct ones."""
    targets = infer_targets(action, st)
    if not targets:
        # tighten with no absent candidate: use components that just became present wrongly is future;
        # fall back to recently present targets that have bad prototypes (wheel/pin tighten).
        targets = [c for c in ACTION_COMPS.get(action, []) if c in protos and "bad" in protos[c]]
    vec = _dino_mean(vid, start_f, end_f, cache)
    if vec is None or not targets:
        return False, "", 0.0
    best_score, best_why = -1e9, ""
    for comp in targets:
        proto = protos.get(comp)
        if not proto or "ok" not in proto or "bad" not in proto:
            continue
        # Higher = more like bad than ok (cosine).
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        score = cos(vec, proto["bad"]) - cos(vec, proto["ok"])
        if score > best_score:
            best_score, best_why = score, f"exec[{COMPONENTS[comp]}] bad-ok={score:.3f}"
    return best_score >= margin, best_why, float(best_score)


def groq_status() -> dict:
    _load_env()
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    if not key:
        return {"available": False, "reason": "no GROQ_API_KEY"}
    body = json.dumps({
        "model": model, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply OK"}],
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
        return {"available": True, "model": model,
                "sample": data["choices"][0]["message"]["content"][:80]}
    except Exception as exc:
        detail = ""
        if isinstance(exc, urllib.error.HTTPError):
            detail = exc.read().decode("utf-8", "ignore")[:200]
        return {"available": False, "model": model, "reason": f"{type(exc).__name__}: {exc} {detail}"}


def _metrics(events: list[dict], flag_key: str) -> dict:
    y = np.array([e["mistake"] for e in events], dtype=bool)
    p = np.array([e[flag_key] for e in events], dtype=bool)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"events": len(events), "mistakes": int(y.sum()), "warned": int(p.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "false_alarms_per_video": round(fp / max(len({e["vid"] for e in events}), 1), 2)}


def pick_margin(val_events: list[dict], protos: dict) -> float:
    cache: dict = {}
    scores, labels = [], []
    for e in val_events:
        _, _, sc = execution_warn(e["action"], e["state"], e["vid"], e["start_f"], e["end_f"],
                                  protos, cache, margin=1e9)
        scores.append(sc); labels.append(e["mistake"])
        if e["vid"] in cache and len(cache) > 4:
            # keep RAM low
            cache.clear()
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    best_key = (-1, -1.0, -1.0)
    best_thr = 0.0
    for thr in np.unique(scores):
        pred = scores >= float(thr)
        tp = int((labels & pred).sum())
        fp = int((~labels & pred).sum())
        fn = int((labels & ~pred).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        key = (1 if prec >= 0.15 else 0, f1, prec)
        if key > best_key:
            best_key, best_thr = key, float(thr)
    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-groq", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    thinking = {
        "product": "Warn at GT action onset if the step looks illegal or about to be an incorrect install",
        "why_gt": "Holds AR and PSR perfect so only warning rules are measured",
        "why_two_branches": (
            "IndustReal mistakes are mostly execution (Incorrectly installed …) while tighten_nut; "
            "order/SOP alone cannot see wrong orientation. Procedure branch covers missing-prereq / "
            "order mistakes; execution branch uses appearance prototypes from train."
        ),
        "llm_vlm": (
            "Groq text LLM intended for PREGO-style expected-next-action checks; local Qwen3-VL for "
            "frame-level install quality. Groq is probed at runtime; VLM weights are not on disk."
        ),
        "papers": "PREGO=action vs expected next; MistSense=procedural+execution; IndustReal PSR≠mistake warn",
    }

    groq = {"skipped": True} if args.skip_groq else groq_status()
    clean = [v for v in TRAIN if not has_mistake(v)]
    rules = learn(clean)
    protos = build_execution_prototypes(TRAIN)

    val_e = action_events(VAL)
    margin = pick_margin(val_e, protos)

    def annotate(events: list[dict]) -> list[dict]:
        cache: dict = {}
        out = []
        for e in events:
            p_warn, p_why = procedure_warn(e["action"], e["state"], rules)
            x_warn, x_why, x_sc = execution_warn(
                e["action"], e["state"], e["vid"], e["start_f"], e["end_f"], protos, cache, margin)
            if len(cache) > 6:
                cache.clear()
            row = {**e, "state": list(e["state"]),
                   "procedure_warn": p_warn, "procedure_why": p_why,
                   "execution_warn": x_warn, "execution_why": x_why, "execution_score": round(x_sc, 4),
                   "warn": p_warn or x_warn}
            out.append(row)
        return out

    val_a = annotate(val_e)
    test_a = annotate(action_events(TEST))

    report = {
        "thinking": thinking,
        "groq": groq,
        "vlm_local": {"available": (ROOT / "models/qwen3-vl-2b").is_dir()},
        "execution_margin_val": margin,
        "prototypes": {COMPONENTS[c]: {k: "yes" for k in p} for c, p in protos.items()},
        "val": {
            "procedure": _metrics(val_a, "procedure_warn"),
            "execution": _metrics(val_a, "execution_warn"),
            "combined": _metrics(val_a, "warn"),
        },
        "test": {
            "procedure": _metrics(test_a, "procedure_warn"),
            "execution": _metrics(test_a, "execution_warn"),
            "combined": _metrics(test_a, "warn"),
        },
        "test_mistakes": [
            {k: r[k] for k in ("vid", "action", "start_f", "mistake_name", "mistake_comp",
                               "procedure_warn", "execution_warn", "warn", "procedure_why",
                               "execution_why")}
            for r in test_a if r["mistake"]
        ],
    }
    (OUT / "warn.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "thinking_summary": thinking["why_two_branches"],
        "groq": groq,
        "val": report["val"],
        "test": report["test"],
        "test_mistakes_caught": sum(1 for r in report["test_mistakes"] if r["warn"]),
        "test_mistakes_total": len(report["test_mistakes"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
