"""The action-end label on the actions the live assistant really detects, and how late it arrives.

The TCN's events give each action's span, exactly as the running label's final window does (``online_vjepa.updates``), and
``action_end.end_labels`` gives the label, with the recipe locked on val. Nothing here is tuned. Steps:

    python -m assist.extract.run_action_end --spans                 # CPU -> output/assist/action_end/spans_test.json
    python -m assist.extract.run_action_end --score --part 0/2      # GPU, one part per GPU -> probs/<member>/<video>.npz
    python -m assist.extract.run_action_end --benchmark             # GPU: ms for one pass over an action -> benchmark.json
    python -m assist.extract.run_action_end --running               # beside --benchmark: the running label's load, until stopped
    python -m assist.extract.run_action_end --labels                # CPU, one look at test -> detected_test.json, <video>/action_end.jsonl

Videos outside the scored test, such as the training videos in the six-case reel, are never scored here:

    python -m assist.extract.run_action_end --spans --videos 11_assy_3_3 22_assy_2_3 --plan spans_reel.json
    python -m assist.extract.run_action_end --score --plan spans_reel.json     # GPU
    python -m assist.extract.run_action_end --write --videos 11_assy_3_3 22_assy_2_3
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np

from assist.extract.action_end import MEMBERS, SAMPLINGS, SMOOTHING, STRENGTH, WEIGHT, end_labels, mean_clip
from monitor.settings import ROOT

OUT = ROOT / "output/assist/action_end"
VJEPA = ROOT / "output/official_split_eval/vjepa2"
VMAE_WEIGHTS = ROOT / "release_vjepa/vmae_full.pt"
RUNS = {"vjft_area_k12": "area", "vjft_hand_k12": "hand", "vjft_hand_k12_s1": "hand"}
LAYERS, EVERY_S = 12, 0.5


def final_spans(events: list, fps: float, n_frames: int) -> list[tuple[int, int]]:
    """Inclusive frames of each event's final window: from its start to its end, never past when the end was confirmed."""
    last = n_frames - 1
    spans = []
    for e in events:
        start, ended = min(round(e.start * fps), last), min(round(e.ended_at * fps), last)
        spans.append((start, max(start, min(round(e.end * fps) - 1, ended))))
    return spans


def best_overlap_accuracy(spans: list[tuple[int, int]], labels: list[int], segments: list[tuple[int, int, int]]) -> tuple[int, int]:
    """(right, total): each true action takes the label of the detection overlapping it most; no detection counts as wrong."""
    right = 0
    for s, e, label in segments:
        overlap = [min(e, b) - max(s, a) + 1 for a, b in spans]
        if overlap and max(overlap) > 0 and labels[int(np.argmax(overlap))] == label:
            right += 1
    return right, len(segments)


def label_delays(ended: list[float], service_s: float) -> list[float]:
    """Seconds from each action's confirmed end to its label, with one pass at a time, in order."""
    free, delays = 0.0, []
    for t in ended:
        free = max(free, t) + service_s
        delays.append(round(free - t, 6))
    return delays


def spans(videos: list[str], name: str) -> None:
    from assist.common import load_config
    from assist.extract.run_ar_vjepa import live_and_events
    from assist.industreal import SPLITS
    from scripts.data.tas_labels import load_mapping

    cfg, actions = load_config(), load_mapping()
    out = {}
    for rid in videos or SPLITS["test"]:
        live, events = live_and_events(rid, cfg, actions)
        out[rid] = final_spans(events, cfg["fps"], len(live))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"videos": len(out), "actions": sum(len(v) for v in out.values())}), flush=True)


def write_videos(videos: list[str]) -> None:
    """``action_end.jsonl`` for videos outside the scored test, such as the training videos in the six-case reel. No score."""
    from assist.common import load_config, write_jsonl
    from assist.extract.run_ar_vjepa import live_and_events
    from scripts.data.tas_labels import load_mapping
    from scripts.recognition.sequence_recognition import transition_counts
    from scripts.segmentation.run_official_split_eval import TRAIN

    cfg, actions = load_config(), load_mapping()
    first, pairs = transition_counts(TRAIN, actions)
    for rid in videos:
        live, events = live_and_events(rid, cfg, actions)
        clip = mean_clip({m: np.load(OUT / "probs" / m / f"{rid}.npz")["probs"] for m in MEMBERS})
        pred = end_labels(live, final_spans(events, cfg["fps"], len(live)), clip, WEIGHT, STRENGTH, SMOOTHING, first, pairs)
        write_jsonl(cfg["paths"]["out"] / rid / "action_end.jsonl", end_rows(events, pred.tolist(), clip, actions))
        print(json.dumps({"written": rid, "actions": len(events)}), flush=True)


def end_rows(events: list, labels: list[int], clip: np.ndarray, actions: list[str]) -> list[dict]:
    return [{"start": e.start, "end": e.end, "ended_at": e.ended_at, "label": actions[p], "conf": round(float(c[p - 1]), 3)}
            for e, p, c in zip(events, labels, clip)]


def _vjepa(name: str):
    import torch

    from scripts.features import vjepa2_finetune as ft

    model = ft.Finetuned(LAYERS, ft.SIZE).cuda()
    model.load_state_dict(torch.load(VJEPA / f"{name}.pt", map_location="cuda", weights_only=True), strict=False)
    return model, SimpleNamespace(crop=RUNS[name], size=ft.SIZE, context=0.0, workers=2)


def score(part: str, batch: int, name: str) -> None:
    import torch

    from scripts.data.tas_labels import load_mapping
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features import vjepa2_finetune as ft

    k, n = (int(x) for x in part.split("/"))
    plan = json.loads((OUT / name).read_text(encoding="utf-8"))
    rids = sorted(plan)[k::n]

    def run(member: str, predict) -> None:
        folder = OUT / "probs" / member
        folder.mkdir(parents=True, exist_ok=True)
        for rid in rids:
            if not (folder / f"{rid}.npz").is_file():
                np.savez(folder / f"{rid}.npz", probs=predict(rid, plan[rid]))
                print(json.dumps({"member": member, "video": rid, "actions": len(plan[rid])}), flush=True)

    for name in RUNS:
        model, args = _vjepa(name)

        def predict(rid: str, rows: list) -> np.ndarray:
            p = ft.predict(model, [(rid, s, e, 1, f) for s, e in rows for f in SAMPLINGS], args, batch)
            return p.reshape(-1, len(SAMPLINGS), p.shape[1]).mean(1)

        run(name, predict)
        del model
        torch.cuda.empty_cache()
    model = vft.load_model(len(load_mapping()), VMAE_WEIGHTS)
    run("vmae_full", lambda rid, rows: vft.tta_probs(model, [(rid, s, e, 1) for s, e in rows], batch, SAMPLINGS))


def benchmark(repeats: int = 15) -> None:
    """ms for one pass over one 4 s action: every member, every sampling, in one batch per member, frames from the cache."""
    import torch

    from scripts.data.tas_labels import load_mapping
    from scripts.features import finetune_videomae_ar as vft
    from scripts.features import vjepa2_finetune as ft

    rid, s, e = "03_assy_0_1", 300, 340
    items = [(rid, s, e, 1, f) for f in SAMPLINGS]

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return round(1000 * (time.perf_counter() - t0) / repeats, 1)

    ms = {}
    for name in RUNS:
        model, args = _vjepa(name)
        args.workers = 0
        ms[name] = timed(lambda: ft.predict(model, items, args, len(SAMPLINGS)))
        del model
        torch.cuda.empty_cache()
    model = vft.load_model(len(load_mapping()), VMAE_WEIGHTS)
    ms["vmae_full"] = timed(lambda: vft.tta_probs(model, [(rid, s, e, 1)], len(SAMPLINGS), SAMPLINGS))
    report = {"gpu": torch.cuda.get_device_name(0), "ms_per_member": ms, "ms_per_action": round(sum(ms.values()), 1)}
    print(json.dumps(report, indent=1), flush=True)


def running() -> None:
    """The running label's V-JEPA 2 update every 0.5 s, to load a GPU the way the live assistant does."""
    import torch

    from scripts.features import vjepa2_finetune as ft

    model = ft.Finetuned(LAYERS, ft.SIZE).cuda()
    model.load_state_dict(torch.load(VJEPA / "vjft_hand_k12_p10.pt", map_location="cuda", weights_only=True), strict=False)
    args = SimpleNamespace(crop="hand", size=ft.SIZE, context=0.0, workers=0)
    while True:
        t0 = time.perf_counter()
        ft.predict(model, [("03_assy_0_1", 300, 340, 1, 0.5)], args, 1)
        torch.cuda.synchronize()
        time.sleep(max(0.0, EVERY_S - (time.perf_counter() - t0)))


def labels() -> None:
    from assist.common import load_config, write_jsonl
    from assist.extract.run_ar_vjepa import gt_frames, live_and_events
    from assist.industreal import SPLITS
    from scripts.data.tas_labels import load_mapping
    from scripts.recognition.pipeline_on_segments import guard_single_look
    from scripts.recognition.sequence_recognition import transition_counts
    from scripts.segmentation.run_official_split_eval import TRAIN
    from scripts.segmentation.tas_metrics import f_counts

    result = OUT / "detected_test.json"
    guard_single_look(result)
    cfg, actions = load_config(), load_mapping()
    index = {a: i for i, a in enumerate(actions)}
    first, pairs = transition_counts(TRAIN, actions)
    totals = {name: {"right": 0, "of": 0, **{o: [0, 0, 0] for o in ("0.1", "0.25", "0.5")}} for name in ("tcn_event_label", "action_end_label")}
    ended_all = {}
    for rid in SPLITS["test"]:
        live, events = live_and_events(rid, cfg, actions)
        spans_ = final_spans(events, cfg["fps"], len(live))
        clip = mean_clip({m: np.load(OUT / "probs" / m / f"{rid}.npz")["probs"] for m in MEMBERS})
        pred = end_labels(live, spans_, clip, WEIGHT, STRENGTH, SMOOTHING, first, pairs)
        write_jsonl(cfg["paths"]["out"] / rid / "action_end.jsonl", end_rows(events, pred.tolist(), clip, actions))
        gt, segs = gt_frames(rid, cfg, actions, len(live))
        ended_all[rid] = [e.ended_at for e in events]
        for name, chosen in (("tcn_event_label", [index[e.label] for e in events]), ("action_end_label", pred.tolist())):
            right, of = best_overlap_accuracy(spans_, chosen, segs)
            row = totals[name]
            row["right"] += right
            row["of"] += of
            painted = np.zeros(len(live), np.int64)
            for (s, e), label in zip(spans_, chosen):
                painted[s:e + 1] = label
            for overlap in ("0.1", "0.25", "0.5"):
                row[overlap] = [a + b for a, b in zip(row[overlap], f_counts(painted, gt, float(overlap), {0}))]

    def f1(tp: int, fp: int, fn: int) -> float:
        return round(200 * tp / max(1, 2 * tp + fp + fn), 2)

    report = {"protocol": "test videos, boundaries detected by the TCN's events; locked recipe; no settings chosen here",
              "actions_detected": sum(len(v) for v in ended_all.values()),
              **{name: {"true_actions_labelled_right": round(100 * r["right"] / r["of"], 2),
                        **{f"f1@{o}": f1(*r[o]) for o in ("0.1", "0.25", "0.5")}} for name, r in totals.items()}}
    speed = OUT / "benchmark_results.json"
    if speed.is_file():
        for case, ms in json.loads(speed.read_text(encoding="utf-8")).items():
            delays = np.array([d for ended in ended_all.values() for d in label_delays(ended, ms / 1000)])
            report.setdefault("label_delay_s", {})[case] = {"ms_per_action": ms, "median": round(float(np.median(delays)), 2),
                                                            "p95": round(float(np.percentile(delays, 95)), 2),
                                                            "max": round(float(delays.max()), 2)}
    result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--spans", "--score", "--benchmark", "--running", "--labels", "--write"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--videos", nargs="*", default=[], help="Videos outside the scored test: --spans, then --write.")
    parser.add_argument("--plan", default="spans_test.json", help="The spans file in output/assist/action_end.")
    args = parser.parse_args()
    if args.spans:
        spans(args.videos, args.plan)
    if args.score:
        score(args.part, args.batch, args.plan)
    if args.write:
        write_videos(args.videos)
    if args.benchmark:
        benchmark()
    if args.running:
        running()
    if args.labels:
        labels()


if __name__ == "__main__":
    main()
