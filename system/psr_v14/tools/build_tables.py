"""Build the step-group model from TRAIN data only.

For every causal AR trigger in a train video we label which GT step group (or none) completed
near it, then store simple count tables:
  * group vocabulary and P(group | procedure, state before)
  * P(trigger action | group), P(action in window | group), P(action since last step | group)
Output: model/tables.json.  `build(videos)` is reused for leave-one-video-out tuning.
"""
import collections
import json
from functools import lru_cache
from pathlib import Path

from psr.ar import OracleARProvider
from psr.core import (INITIAL_STATE, apply_steps, bits, code, gt_groups, list_videos,
                      load_psr_gt, procedure_type, frame_paths)
from psr.events import EventExtractor

MATCH_BEFORE, MATCH_AFTER = 70, 15   # GT may sit up to 70 frames before the trigger (long tightening), 15 after


POST_MAX = 150


def triggers_for(split, video, n):
    ar = OracleARProvider(); ar.reset(split, video, n)
    ex = EventExtractor(); trig = []
    obs = [[o.name for o in ar.observe(t)] for t in range(n + 1)]
    for t in range(n + 1):
        trig += ex.update(t, obs[t])
    for tr in trig:   # actions seen after the trigger: {name: first offset}; the tracker only uses offsets <= its decision delay
        post = {}
        for dt in range(1, POST_MAX + 1):
            if tr.frame + dt > n:
                break
            for nm in obs[tr.frame + dt]:
                post.setdefault(nm, dt)
        tr.post = post
    return trig


def label_triggers(trig, groups):
    """Greedy one-to-one assignment of GT groups to nearest trigger within tolerance."""
    lab = [None] * len(trig)
    used = set()
    pairs = []
    for gi, (gf, g) in enumerate(groups):
        for ti, tr in enumerate(trig):
            d = tr.frame - gf
            if -MATCH_AFTER <= d <= MATCH_BEFORE:
                pairs.append((abs(d), gi, ti))
    for _, gi, ti in sorted(pairs):
        if gi in used or lab[ti] is not None:
            continue
        lab[ti] = gi; used.add(gi)
    return lab, used


@lru_cache(maxsize=None)
def video_samples(split, video):
    n = len(frame_paths(split, video))
    proc = procedure_type(video)
    groups = gt_groups(load_psr_gt(split, video))
    trig = triggers_for(split, video, n)
    lab, used = label_triggers(trig, groups)
    samples = []
    state = bits(INITIAL_STATE[proc]); gi_done = 0; since = set(); ntrig = 0; last_step = 0
    for ti, tr in enumerate(trig):
        ntrig += 1
        while gi_done < len(groups) and groups[gi_done][0] < tr.frame - MATCH_BEFORE and gi_done not in used:
            state = apply_steps(state, groups[gi_done][1]); gi_done += 1
        since |= set(tr.window)
        g = groups[lab[ti]][1] if lab[ti] is not None else ()
        samples.append({"proc": proc, "state": code(state), "trigger": tr.name,
                        "window": sorted(tr.window), "since": sorted(since), "ntrig": min(ntrig, 3), "ntrig_raw": ntrig,
                        "duration": tr.duration, "elapsed": tr.frame - last_step, "frame": tr.frame,
                        "post": tr.post, "tsince": tr.tsince, "group": list(g)})
        if lab[ti] is not None:
            state = apply_steps(state, g)
            gi_done = max(gi_done, lab[ti] + 1)
            since = set(); ntrig = 0; last_step = tr.frame
    return tuple(json.dumps(s) for s in samples), len(groups), len(used)


def build(videos, split="train") -> dict:
    """videos: list of names (all from `split`) or list of (split, name) pairs."""
    samples = []
    for v in videos:
        sp, name = v if isinstance(v, tuple) else (split, v)
        ss, _, _ = video_samples(sp, name)
        samples += [json.loads(s) for s in ss]
    vocab = sorted({tuple(s["group"]) for s in samples})
    key = lambda g: ",".join(map(str, g))
    C = collections.Counter
    return {
        "source": f"{sorted({v[0] if isinstance(v, tuple) else split for v in videos})}: {len(videos)} videos",
        "groups": [list(g) for g in vocab],
        "actions": sorted({a for s in samples for a in s["window"] + s["since"]}),
        "triggers": sorted({s["trigger"] for s in samples}),
        "n_group": {key(g): c for g, c in C(tuple(s["group"]) for s in samples).items()},
        "n_group_proc": {f"{p}|{key(g)}": c for (p, g), c in C((s["proc"], tuple(s["group"])) for s in samples).items()},
        "n_group_proc_state": {f"{p}|{st}|{key(g)}": c for (p, st, g), c in
                               C((s["proc"], s["state"], tuple(s["group"])) for s in samples).items()},
        "n_proc_state": {f"{p}|{st}": c for (p, st), c in C((s["proc"], s["state"]) for s in samples).items()},
        "n_trig": {f"{key(g)}|{t}": c for (g, t), c in C((tuple(s["group"]), s["trigger"]) for s in samples).items()},
        "n_act": {f"{key(g)}|{a}": c for (g, a), c in C((tuple(s["group"]), a) for s in samples for a in s["window"]).items()},
        "n_ntrig": {f"{key(g)}|{n}": c for (g, n), c in C((tuple(s["group"]), s["ntrig"]) for s in samples).items()},
        "n_since": {f"{key(g)}|{a}": c for (g, a), c in C((tuple(s["group"]), a) for s in samples for a in s["since"]).items()},
    }


def main():
    vids = list_videos("train")
    tot = sum(video_samples("train", v)[1] for v in vids)
    matched = sum(video_samples("train", v)[2] for v in vids)
    print(f"train trigger recall: {matched}/{tot} GT groups have a trigger within tolerance")
    tables = build(vids)
    out = Path(__file__).parent / "tables.json"
    out.write_text(json.dumps(tables, indent=1))
    print("groups:", tables["groups"]); print("saved", out)


if __name__ == "__main__":
    main()
