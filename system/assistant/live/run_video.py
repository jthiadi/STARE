"""Run the whole pipeline on recordings, every model in inference, each stage skipped when its output exists:
  1 causal TCN pipeline (their code) -> live_probs.npz, ar_live.jsonl
  2 online V-JEPA 2 label (theirs)     -> ar_vjepa_prefix.jsonl
  3 action-end label (theirs)          -> action_end.jsonl
  4 frozen v14 live (verifiers + ASD)  -> v14_live.json
  5 hands + engine                     -> hands.jsonl, asd_v14.jsonl, engine_pred_vjepap_asdv14.json
Progress per recording in output/assist/<rid>/live_status.json (read by the job API).
    CUDA_VISIBLE_DEVICES=0 python -m live.run_video <rid> [<rid> ...]
    CUDA_VISIBLE_DEVICES=0 python -m live.run_video --test 0/2      # every 2nd test recording, starting at 0"""
import json
import sys
import time
import traceback
from pathlib import Path

OUT = Path("output/assist")
STAGES = [("ar", "live_probs.npz", "live.step_ar"), ("vjepa", "ar_vjepa_prefix.jsonl", "live.step_vjepa"),
          ("final_labels", "action_end.jsonl", "live.step_end"), ("v14", "v14_live.json", "live.step_v14"),
          ("engine", "engine_pred_vjepap_asdv14.json", "live.step_engine")]


def status(rid: str, **kw) -> None:
    p = OUT / rid / "live_status.json"
    s = json.loads(p.read_text()) if p.exists() else {"video": rid}
    s.update(kw, updated=time.time())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s))


def run_one(rid: str) -> bool:
    import importlib
    done = [name for name, f, _ in STAGES if (OUT / rid / f).exists()]
    status(rid, state="running", done=done, error=None, started=time.time())
    for name, f, mod in STAGES:
        if (OUT / rid / f).exists():
            continue
        status(rid, stage=name)
        t0 = time.time()
        try:
            importlib.import_module(mod).main(rid)
        except Exception:
            status(rid, state="failed", stage=name, error=traceback.format_exc()[-2000:])
            print(f"{rid}: {name} FAILED", flush=True)
            return False
        done.append(name)
        status(rid, done=done, **{f"{name}_s": round(time.time() - t0, 1)})
    status(rid, state="done", stage=None)
    print(f"{rid}: done", flush=True)
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--test":
        from assist.industreal import SPLITS
        k, n = (int(x) for x in args[1].split("/"))
        args = SPLITS["test"][k::n]
    for rid in args:
        run_one(rid)
    print("ALL DONE", flush=True)
