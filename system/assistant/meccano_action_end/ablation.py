"""Component breakdown of the MECCANO action-end result (diagnostic only: nothing here is chosen, every setting is the
fixed IndustReal recipe or the VideoMAE run already picked on val by ``action_end --val``).

    python -m meccano_action_end.ablation      # CPU -> output/meccano_action_end/ablation.json
"""
from __future__ import annotations

import json

import numpy as np

from assist.extract.action_end import SMOOTHING, STRENGTH, WEIGHT, end_labels
from meccano_action_end import data
from meccano_action_end.action_end import TCN_RUN, VAL_RESULT, VJ_MEMBERS, per_video, top1
from meccano_action_end.recognition import PROBS, TCN_DIR, _vjepa_probs


def run(split: str, vmae_run: str) -> dict[str, float]:
    clips = data.clips(data.split(split))
    labels = np.array([c[3] for c in clips])
    vj = _vjepa_probs()
    members = {"videomae_" + vmae_run: np.load(PROBS)[f"{vmae_run}/{split}/whole"]}
    members.update({m: vj[m][split] for m in VJ_MEMBERS})
    out = {f"alone/{name}": top1(p.argmax(1) + 1, labels) for name, p in members.items()}
    mean = sum(members.values()) / len(members)
    vj_mean = sum(vj[m][split] for m in VJ_MEMBERS) / len(VJ_MEMBERS)
    out["vjepa_3_runs_mean"] = top1(vj_mean.argmax(1) + 1, labels)
    out["ensemble_mean"] = top1(mean.argmax(1) + 1, labels)
    first, pairs = data.transition_counts(data.split("train"))
    for name, w, strength in (("tcn_only", 0.0, 0.0), ("ensemble+tcn", WEIGHT, 0.0),
                              ("ensemble+prior", 1.0, STRENGTH), ("full_recipe", WEIGHT, STRENGTH)):
        pred = np.zeros(len(clips), np.int64)
        for idx in per_video(clips):
            frames = np.load(TCN_DIR / TCN_RUN / f"{clips[idx[0]][0]}.npy").astype(np.float32)
            spans = [(clips[i][1], clips[i][2]) for i in idx]
            pred[idx] = end_labels(frames, spans, mean[idx], w, strength, SMOOTHING, first, pairs)
        out[name] = top1(pred, labels)
    return {k: round(v, 2) for k, v in out.items()}


def main() -> None:
    vmae_run = json.loads(VAL_RESULT.read_text(encoding="utf-8"))["videomae_run"]
    report = {"note": "diagnostic breakdown; fixed IndustReal recipe settings, VideoMAE run chosen on val",
              "videomae_run": vmae_run, "val": run("val", vmae_run), "test": run("test", vmae_run)}
    (data.OUT / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
