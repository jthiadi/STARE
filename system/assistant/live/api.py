"""Job service for the on-demand pipeline (server side). Listens on 127.0.0.1 only; reach it through the SSH tunnel.
  GET  /api/recordings          every recording (test / val / train) and what is computed for it
  POST /api/run/<rid>           start live.run_video for <rid> on the GPU with fewer of our jobs (queued if both busy)
  GET  /api/status/<rid>        stage, finished stages, error
  GET  /api/bundle/<rid>        zip of the files the browser UI needs (engine run, actions, final labels, hands, v14)
    python -m live.api [--port 8020]"""
import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from assist.industreal import SPLITS
from live.run_video import OUT, STAGES

BUNDLE = ("engine_pred_vjepap_asdv14.json", "ar_vjepa_prefix.jsonl", "ar_live.jsonl", "action_end.jsonl", "hands.jsonl",
          "asd_v14.jsonl", "v14_live.json", "live_status.json")
ALL = {rid: split for split in ("test", "val", "train") for rid in SPLITS[split]}
app = FastAPI()
_jobs: dict[str, tuple[subprocess.Popen, int]] = {}
_queue: list[str] = []
_lock = threading.Lock()


def _info(rid: str) -> dict:
    d = OUT / rid
    s = json.loads((d / "live_status.json").read_text()) if (d / "live_status.json").exists() else {}
    done = [name for name, f, _ in STAGES if (d / f).exists()]
    running = rid in _jobs and _jobs[rid][0].poll() is None
    state = "done" if len(done) == len(STAGES) else "running" if running or s.get("state") == "running" and \
        time.time() - s.get("updated", 0) < 1800 else "queued" if rid in _queue else s.get("state", "not computed")
    return {"rid": rid, "split": ALL[rid], "state": state, "stage": s.get("stage"), "done": done,
            "stages": [n for n, _, _ in STAGES], "error": s.get("error")}


def _pump() -> None:
    """start queued jobs while a GPU has none of ours"""
    with _lock:
        busy = {g for rid, (p, g) in _jobs.items() if p.poll() is None}
        for g in (0, 1):
            if g in busy or not _queue:
                continue
            rid = _queue.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            log = open(Path("..") / "logs" / f"job_{rid}.log", "a")
            _jobs[rid] = (subprocess.Popen([sys.executable, "-m", "live.run_video", rid], env=env, stdout=log,
                                           stderr=subprocess.STDOUT), g)
            busy.add(g)


def _loop() -> None:
    while True:
        _pump(); time.sleep(5)


@app.get("/api/recordings")
def recordings() -> list[dict]:
    return [_info(rid) for rid in ALL]


@app.post("/api/run/{rid}")
def run(rid: str) -> dict:
    if rid not in ALL:
        raise HTTPException(404, "unknown recording")
    info = _info(rid)
    if info["state"] in ("done", "running", "queued"):
        return info
    with _lock:
        _queue.append(rid)
    _pump()
    return _info(rid)


@app.get("/api/status/{rid}")
def status(rid: str) -> dict:
    if rid not in ALL:
        raise HTTPException(404, "unknown recording")
    return _info(rid)


@app.get("/api/bundle/{rid}")
def bundle(rid: str) -> Response:
    if rid not in ALL or _info(rid)["state"] != "done":
        raise HTTPException(409, "not computed yet")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in BUNDLE:
            if (OUT / rid / f).exists():
                z.write(OUT / rid / f, f)
    return Response(buf.getvalue(), media_type="application/zip")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    a = ap.parse_args()
    threading.Thread(target=_loop, daemon=True).start()
    print(f"job service on http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
