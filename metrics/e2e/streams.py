"""Shifted label streams (PREREGISTRATION.md): each label reaches v14 when the live system would have it.
Needs results/latency.json (latency.py) and, for the validation ensemble, work/val_ensemble (val_ensemble.py).
Writes work/streams/{tcn,ens}_{val,test}/<rid>/ar.jsonl (one row per frame, top 1 only, as eval_v14 reads it)
and, for the v9 ablation (PREREGISTRATION_ablations.md; needs results/latency_v9.json), work/streams/v9_test.
The oracle ablation needs no file: its shift is applied to the annotated actions in retune.py / test_once.py.
    python metrics/e2e/streams.py [tcn] [ens] [v9] [v13]"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("E2E_WORK", HERE / "work"))           # outputs (reproduce_all.py points these elsewhere)
RES = Path(os.environ.get("E2E_RESULTS", HERE / "results"))
A = HERE.parents[1] / "system" / "assistant"
sys.path.insert(0, str(A))
from assist.industreal import SPLITS

FPS = 10.0
OUT = A / "output/assist"


def read(p: Path) -> list:
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert all(r["f"] == i for i, r in enumerate(rows)), p
    return rows


def write(path: Path, tops: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t, top in enumerate(tops):
            f.write(json.dumps({"f": t, "t": round(t / FPS, 2), "top": [top]}) + "\n")


def tcn_stream(rows: list, D: int) -> list:
    """At frame t: the TCN label of frame t - D (computed by then); background before."""
    return [rows[t - D]["top"][0] if t >= D else ["background", 1.0] for t in range(len(rows))]


def ens_stream(rows: list, ends: list, D: int, ready_after_end: float) -> list:
    """At frame t, for f = t - D: the final label of the action containing f if it exists by t
    (ended_at + ready_after_end), else the TCN label of f (make_final_label_stream.py, with the measured times)."""
    n = len(rows)
    final = [None] * n
    for r in ends:
        ready = (r["ended_at"] + ready_after_end) * FPS
        for f in range(round(r["start"] * FPS), min(round(r["end"] * FPS), n)):
            final[f] = ([r["label"], r["conf"]], ready)
    out = []
    for t in range(n):
        f = t - D
        if f < 0:
            out.append(["background", 1.0])
        elif final[f] is not None and final[f][1] <= t:
            out.append(final[f][0])
        else:
            out.append(rows[f]["top"][0])
    return out


def main() -> None:
    names = sys.argv[1:] or ["tcn", "ens"]
    lat = json.loads((HERE / "results/latency.json").read_text())
    Dt, De, R = lat["D_tcn_frames"], lat["D_ens_frames"], lat["ens_ready_after_end_s"]
    W = WORK / "streams"
    VE = HERE / "work/val_ensemble"
    for rid in SPLITS["val"]:
        oof = read(OUT / rid / "ar.jsonl")                                   # out-of-fold TCN (v14's selection input)
        if "tcn" in names:
            write(W / "tcn_val" / rid / "ar.jsonl", tcn_stream(oof, Dt))
        if "ens" in names:
            write(W / "ens_val" / rid / "ar.jsonl", ens_stream(oof, read_plain(VE / rid / "action_end.jsonl"), De, R))
    D9 = json.loads((HERE / "results/latency_v9.json").read_text())["D_v9_frames"] if {"v9", "v13"} & set(names) else None
    for rid in SPLITS["test"]:
        if "v9" in names:                                                    # same TCN labels, v9's own compute
            write(W / "v9_test" / rid / "ar.jsonl", tcn_stream(read(OUT / rid / "ar.jsonl"), D9))
        if "v13" in names:                                                   # no ASD: same work and shift as v9
            write(W / "v13_test" / rid / "ar.jsonl", tcn_stream(read(OUT / rid / "ar.jsonl"), D9))
        if "tcn" in names:                                                   # the v14 + TCN row's input
            write(W / "tcn_test" / rid / "ar.jsonl", tcn_stream(read(OUT / rid / "ar.jsonl"), Dt))
        if "ens" in names:                                                   # as the ensemble row
            ends = read_plain(OUT / rid / "action_end.jsonl")
            write(W / "ens_test" / rid / "ar.jsonl", ens_stream(read(OUT / rid / "ar_live.jsonl"), ends, De, R))
    print(json.dumps({"D_tcn_frames": Dt, "D_ens_frames": De, "ens_ready_after_end_s": R, "D_v9_frames": D9, "written": str(W)}))


def read_plain(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8")]


if __name__ == "__main__":
    main()
