"""Drive the running v14 demo through its HTTP API: play recordings to the end and compare each final score with the
frozen v14 per-recording result (results/v14_predictedAR_<split>.json) and v14's saved alerts.
    python test_demo.py <port> <cached|live> <split> <video> [<video> ...]"""
import json, sys, time, urllib.request
from pathlib import Path

port, vmode, split, vids = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
U = f"http://127.0.0.1:{port}"
ref = {r["video"]: r for r in json.loads((Path(__file__).resolve().parent.parent / "psr_v14/results"
                                         / f"v14_predictedAR_{split}.json").read_text())["videos"]}


def call(path, body=None):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


for v in vids:
    t0 = time.time()
    print(call("/api/start", {"split": split, "video": v, "verifier": vmode, "speed": 0}), flush=True)
    while True:
        s = call("/api/state")
        if s.get("error"):
            print("ERROR", s["error"][-800:]); sys.exit(1)
        if s.get("done"):
            break
        time.sleep(2)
    f = s["final"]["live"]; r = ref[v]
    same_steps = [[e["frame"], e["description"]] for e in s["pred"]] == [list(x) for x in r["predicted_steps"]]
    same_alerts = [(a["type"], a.get("step"), a["t"]) for a in s["alerts"]] == [(a["type"], a.get("step"), a["t"]) for a in r["mistake_alerts"]]
    print(f"{v}: demo POS {f['POS']} F1 {f['F1']} delay {f['delay_s']}s | frozen POS {r['pos']} F1 {r['f1']} delay "
          f"{r['delay_frames'] / 10}s | steps identical {same_steps} | alerts identical {same_alerts} ({len(s['alerts'])}) | "
          f"latency {s['lat_stats']} | {time.time() - t0:.0f}s", flush=True)
