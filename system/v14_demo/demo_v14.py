"""Live demo of frozen v14 with the teammate's causal AR predictions (the deployable setting).
Made from the v9 demo (industreal_extra/v9/PSR/demo/demo_v9.py); the frozen package is read from PSR_v14/, unchanged.

    python demo_v14.py           ->  http://127.0.0.1:8767

Per frame, strictly in order: decode -> teammate AR top-1 (precomputed causal output, ar.jsonl) ->
even frames: localizer + both DINOv2 verifiers; odd frames: official IndustReal ASD -> v14 state filter ->
v14 mistake alerts (task-graph rules on the steps reported so far; the end-of-recording check when the video ends).
Exactly the evaluated pipeline (eval_v14.py); nothing is tuned here.

Shown for comparison only (never fed to the v9 + teammate-AR tracker):
  * PSR ground truth and AR ground truth (timelines, state table, step table, final scoring);
  * a second v9 tracker driven by GT AR on the same verifier outputs: ORACLE, diagnostic only.

Verifier source: "live" runs the two verifiers and the ASD on the GPU now; "cached" reuses the verifier and ASD outputs
that the reported scores were computed from (cache/verif_*, cache/asd_*), which reproduces those scores exactly.
Frames come from <INDUSTREAL_DATA>/<split>/<rid>/rgb/ if present, else from the recording's .mp4 (frames_compat.py).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VBACK", "base")
os.environ.setdefault("VRES", "336")

import sys
HERE0 = Path(__file__).resolve().parent
PSR_V14 = Path(os.environ.get("PSR_V14", HERE0.parent / "psr_v14"))     # the frozen v14 package, read only
sys.path.insert(0, str(PSR_V14)); sys.path.insert(0, str(HERE0))
import frames_compat  # noqa: F401,E402  (frames from video when the dataset has no JPEG frames)
import cv2
import numpy as np

from psr.ar import OracleARProvider
from psr.blend import Blend
from psr.core import (COMPONENTS, INITIAL_STATE, PROC_INFO, STEP_DESC, apply_steps, bits, code, frame_paths,
                      list_videos, load_ar_labels, load_psr_gt, procedure_type, video_dir)
from psr.filter_tracker import FilterConfig
from psr.lr_model import LRGroupModel
from psr.official_eval import evaluate, psr_utils
from psr.tracker import GroupModel, TrackerConfig
from psr.vision import CACHE, crop
from psr.v11 import mistake_alerts
from psr.v14 import V14Tracker, asd_states, ASD_CLASSES

HERE = HERE0                                      # demo folder
ROOT = PSR_V14
FZ = ROOT / "model"
WEIGHTS = Path(os.environ.get("PSR_WEIGHTS", HERE.parent / "weights"))   # verifier + ASD weights (not in the package)
RES = ROOT / "results"
# predicted AR: folder with <video>/ar.jsonl (set PSR_AR_DIR)
LIVE_AR = Path(os.environ.get("PSR_AR_DIR", ROOT / "ar_predictions"))
FPS = 10
CFG = json.loads((FZ / "config.json").read_text())
SM = CFG["step_model"]
B3 = {"all": {"POS": 0.797, "F1": 0.883, "delay_s": 22.4}, "with_errors": {"POS": 0.731, "F1": 0.816, "delay_s": 20.4}}


AC = CFG["asd"]


def make_tracker(video, asd):
    model = Blend(GroupModel(FZ / "tables.json"), LRGroupModel(path=FZ / SM["lr_path"]), SM["w_lr"])
    tcfg = TrackerConfig(w_action=SM["w_action"], w_since=SM["w_since"], null_bias=SM["null_bias"])
    return V14Tracker(procedure_type(video), FilterConfig(**CFG["filter"]), model, tcfg, d0=CFG["v11"]["d0"],
                      asd=asd, w=AC["w"], eps=AC["eps"], rate=AC["rate"])


def cached_asd(split, video):
    return asd_states(np.load(CACHE / f"asd_{split}" / f"{video}.npz")["rows"], AC["cmin"])


def load_live_ar(video, n):
    rows = [json.loads(l) for l in open(LIVE_AR / video / "ar.jsonl", encoding="utf-8")]
    assert len(rows) == n and all(r["f"] == i for i, r in enumerate(rows)), f"{video}: ar.jsonl does not match frames"
    return [r["top"] for r in rows]


def cached_feats(split, video):
    """Verifier outputs + localizer boxes exactly as used by eval_v6_livear.py."""
    acc, cnt, qv = {}, {}, {}
    for ver in CFG["verifiers"]:
        z = np.load(CACHE / f"{ver['tag']}_{split}" / f"{video}.npz")
        for t, q in zip(z["t"], z["q"]):
            qv[int(t)] = float(q)
        for t, p in zip(z["t"], z["p"]):
            p = p.astype(np.float32)
            p3 = np.stack([1 - p, p, np.full(10, 1e-3, np.float32)], 1).reshape(-1)
            acc[int(t)] = acc.get(int(t), 0) + p3; cnt[int(t)] = cnt.get(int(t), 0) + 1
    boxes = {}
    if (CACHE / "crops" / split / f"{video}.npz").exists():          # localizer boxes: display only, optional
        z = np.load(CACHE / "crops" / split / f"{video}.npz")
        boxes = {int(t): [float(x) for x in b] for t, b in zip(z["t"], z["box"])}
    return {t: {"cls": acc[t] / cnt[t], "q": qv.get(t, 0.0), "box": boxes.get(t)} for t in acc}


class LiveVision:
    """Localizer + both frozen verifiers on the GPU (same code path as bench_v6.py)."""

    def __init__(self):
        import torch
        import train_verifier as TV
        from psr.vision import VisionVerifier
        self.torch, self.TV = torch, TV
        self.models = []
        for v in CFG["verifiers"]:
            TV.UNFREEZE = v.get("unfreeze", 4)
            m = TV.Verifier().cuda()
            m.load_state_dict(torch.load(WEIGHTS / f"{v['tag']}.pt", map_location="cuda"))
            self.models.append(m.eval().half())
        self.vv = VisionVerifier(bank="exemplar")
        from ultralytics import YOLO
        self.asd = YOLO(str(WEIGHTS / "asd_best_IndustRealandSynthetic.pt"))

    def asd_state(self, bgr):
        """official IndustReal ASD on one frame -> state of its most confident box (>= cmin), else None"""
        a = time.perf_counter()
        res = self.asd.predict(bgr, conf=0.05, imgsz=640, verbose=False)[0]
        st = None
        if len(res.boxes):
            i = int(res.boxes.conf.argmax()); name = ASD_CLASSES[int(res.boxes.cls[i])]
            if float(res.boxes.conf[i]) >= AC["cmin"] and name not in ("background", "error_state"):
                st = name
        return st, (time.perf_counter() - a) * 1000

    def reset(self):
        self.vv.prev_box = None

    def __call__(self, bgr):
        torch, TV, vv = self.torch, self.TV, self.vv
        lat = {}
        a = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, pt = vv.enc.encode(rgb, vv.FRAME_SIZE)
        box, _ = vv.localize(pt)
        b = time.perf_counter(); lat["localize_ms"] = (b - a) * 1000
        if box is None:
            lat["verifiers_ms"] = 0.0
            return None, lat
        roi = cv2.resize(crop(rgb, box, pad=0.1), (160, 160), interpolation=cv2.INTER_AREA)
        q, _ = vv.view_quality(roi)
        hr = cv2.resize(crop(bgr, box, pad=0.1), (336, 336), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", hr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        x = TV.to_tensor([jpg]).cuda().half()
        with torch.no_grad():
            ps = [torch.sigmoid(m(x).float())[0].cpu().numpy() for m in self.models]
        torch.cuda.synchronize()
        lat["verifiers_ms"] = (time.perf_counter() - b) * 1000
        p = np.mean([np.stack([1 - s, s, np.full(10, 1e-3, np.float32)], 1).reshape(-1) for s in ps], 0)
        return {"cls": p, "q": float(q), "box": [float(v) for v in box]}, lat


def verb(name):
    return name.split("_")[0] if name else "background"


def metrics(gt, y):
    m = evaluate(gt, y, PROC_INFO)
    return {"POS": round(float(m["pos"]), 3), "F1": round(float(m["f1"]), 3),
            "delay_s": round(float(m["avg_delay"]) / FPS, 2), "TP": int(m["system_TPs"]),
            "FP": int(m["system_FPs"]), "FN": int(m["system_FNs"])}


def official_pairs(gt, pred):
    """Per GT entry, the prediction the official evaluator pairs it with (psr_utils.determine_performance logic,
    using its own match_indices): status TP (pred at/after GT), early (pred before GT -> system FP),
    missed (FN).  Predictions not paired with any GT entry are returned as extra (FP).  Display only."""
    g_t = [e["frame"] for e in gt]; p_t = [e["frame"] for e in pred]
    out = [{"pred": None, "status": "missed"} for _ in gt]
    extra = []
    for sid in sorted({e["id"] for e in gt} | {e["id"] for e in pred}):
        ig = [i for i, e in enumerate(gt) if e["id"] == sid]
        ip0 = ip = [i for i, e in enumerate(pred) if e["id"] == sid]
        if not ig:                                  # predicted, never in GT: all FP
            extra += [pred[i] for i in ip]
            continue
        if not ip:                                  # in GT, never predicted: all FN
            continue
        if len(ig) == len(ip) and len(ip) > 1:
            ip = psr_utils.match_indices(ip, p_t, ig, g_t)
        elif len(ig) > len(ip):                     # surplus GT entries stay "missed"
            ig = psr_utils.match_indices(ig, g_t, ip, p_t)
        else:                                       # official code also re-matches the 1:1 case here
            ip = psr_utils.match_indices(ip, p_t, ig, g_t)
            left = [i for i in ip0 if i not in set(int(x) for x in ip)]
            extra += [pred[i] for i in left[:len(ip0) - len(ig)]]   # surplus predictions: FP (by count)
        free = [i for i, e in enumerate(gt) if e["id"] == sid]
        own = list(ip0)
        for a, b in zip(ig, ip):
            a, b = int(a), int(b)
            status = "TP" if p_t[b] >= g_t[a] else "early"          # exactly the official comparison
            # match_indices falls back to index 0 (possibly another step) when no candidate lies after the
            # reference time; that only happens when this step's own prediction is early too, so show that one
            row = a if a in free else free[0]
            show = b if b in own else own[0]
            free.remove(row); own.remove(show)
            # quirk: official verdict disagrees with the displayed times (it compared against another GT entry)
            quirk = (status == "TP") != (p_t[show] >= g_t[row])
            out[row] = {"pred": p_t[show], "status": status, "quirk": quirk}
    return out, extra


def state_after(proc, events, t):
    s = bits(INITIAL_STATE[proc])
    by = {}
    for e in events:
        if e["frame"] <= t:
            by.setdefault(e["frame"], []).append(e["id"])
    for f in sorted(by):
        s = apply_steps(s, by[f])
    return code(s)


def state_after_err(proc, events, t):
    """Like state_after on the error labels, but keeps 'x' for an incorrectly installed component (display only)."""
    s = bits(INITIAL_STATE[proc])
    by = {}
    for e in events:
        if e["frame"] <= t:
            by.setdefault(e["frame"], []).append(e["id"])
    for f in sorted(by):
        s = apply_steps(s, by[f])
    return "".join("x" if v == -1 else str(v) for v in s)


class Session:
    def __init__(self, split, video, vmode, speed, vision):
        self.split, self.video, self.vmode, self.speed = split, video, vmode, speed
        self.paths = frame_paths(split, video)
        self.n = len(self.paths)
        self.proc = procedure_type(video)
        self.ar_top = load_live_ar(video, self.n)                 # causal predictions; read one frame at a time
        self.oracle = OracleARProvider(); self.oracle.reset(split, video, self.n)
        self.vision = vision
        if vision is not None:
            vision.reset()
        self.cache = cached_feats(split, video) if vmode == "cached" else None
        self.asd = cached_asd(split, video) if vmode == "cached" else {}      # live: filled frame by frame
        self.tr = make_tracker(video, self.asd)       # v14 + teammate AR (deployable)
        self.tr_or = make_tracker(video, self.asd)    # v14 + GT AR (oracle diagnostic)
        self.alerts = []                              # v14 mistake alerts so far (causal)
        # ground truth: display / scoring only
        self.gt = load_psr_gt(split, video)
        self.gt_err = load_psr_gt(split, video, with_errors=True)
        self.ar_gt = load_ar_labels(split, video)
        self.has_errors = bool(psr_utils.video_contains_errors(self.gt, PROC_INFO, video_dir(split, video)))
        self.t = -1
        self.jpg = b""
        self.runs = []                            # teammate top-1 runs [start, end, name] revealed so far
        self.ar_hits = 0
        self.now = {}
        self.feat = None
        self.lat, self.lat_hist = {}, []
        self.final = None
        self.done = self.stop = self.paused = False
        self.error = None
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def info(self):
        return {"video": self.video, "split": self.split, "proc": self.proc, "n_frames": self.n, "fps": FPS,
                "has_errors": self.has_errors, "vmode": self.vmode, "speed": self.speed,
                "components": COMPONENTS, "initial": INITIAL_STATE[self.proc],
                "gt_events": self.gt, "gt_events_err": self.gt_err,
                "ar_gt": [[s.start, s.end, s.name] for s in sorted(self.ar_gt, key=lambda s: s.start)]}

    def run(self):
        try:
            nxt = time.perf_counter()
            for t in range(self.n):
                while self.paused and not self.stop:
                    time.sleep(0.05); nxt = time.perf_counter()
                if self.stop:
                    return
                if self.speed > 0:                      # pace to video time x speed (0 = as fast as possible)
                    nxt += 1 / (FPS * self.speed)
                    now = time.perf_counter()
                    if nxt > now:
                        time.sleep(nxt - now)
                    elif now - nxt > 0.5:               # fell behind (slow frame): do not burst to catch up
                        nxt = now
                self.step(t)
            self.tr.finish(self.n - 1); self.tr_or.finish(self.n - 1)
            alerts = mistake_alerts(self.video, self.tr, self.n)            # with the end-of-recording check
            with self.lock:
                self.alerts = alerts
            fin = {"live": metrics(self.gt, self.tr.y_hat), "oracle": metrics(self.gt, self.tr_or.y_hat),
                   "live_err_labels": metrics(self.gt_err, self.tr.y_hat_err),
                   "ar_frame_acc": round(self.ar_hits / self.n, 3)}
            with self.lock:
                self.final, self.done = fin, True
        except Exception:
            self.error = traceback.format_exc()
            print(self.error)

    def step(self, t):
        lat = {}
        a = time.perf_counter()
        bgr = cv2.imread(str(self.paths[t]))
        b = time.perf_counter(); lat["decode_ms"] = (b - a) * 1000
        top = self.ar_top[t]                                      # teammate output for frame t only
        name, p = top[0]
        names = [] if name == "background" else [name]
        c = time.perf_counter(); lat["ar_lookup_ms"] = (c - b) * 1000
        feat = None
        if self.cache is not None:
            feat = self.cache.get(t)
        elif t % 2 == 0:
            feat, vl = self.vision(bgr); lat.update(vl)
        elif AC["rate"] == 1 or t % 2 == 1:
            st, ms = self.vision.asd_state(bgr); lat["asd_ms"] = ms
            if st is not None:
                self.asd[t] = st
        d = time.perf_counter()
        new = self.tr.update(t, names, feat)
        if new:                                    # alerts from the steps reported so far (no end-of-recording check)
            al = mistake_alerts(self.video, self.tr, t + 1, end_check=False)
            with self.lock:
                self.alerts = al
        lat["filter_ms"] = (time.perf_counter() - d) * 1000
        lat["end_to_end_ms"] = (time.perf_counter() - a) * 1000
        # ---- comparison only below: oracle tracker + GT (never reaches self.tr) ----
        gt_now = [o.name for o in self.oracle.observe(t)]
        self.tr_or.update(t, gt_now, feat)
        hit = (name in gt_now) if gt_now else (name == "background")
        self.ar_hits += hit
        img = bgr
        if feat is not None and feat.get("box"):
            x, y, w, h = map(int, feat["box"])
            ok = feat["q"] >= CFG["filter"]["q_ref"]
            img = bgr.copy()
            cv2.rectangle(img, (x, y), (x + w, y + h), (90, 210, 90) if ok else (40, 170, 250), 3)
        jpg = cv2.imencode(".jpg", cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        with self.lock:
            self.t, self.jpg = t, jpg
            if self.runs and self.runs[-1][2] == name and self.runs[-1][1] == t - 1:
                self.runs[-1][1] = t
            else:
                self.runs.append([t, t, name])
            self.now = {"top": [[n_, round(float(p_), 3)] for n_, p_ in top], "gt": gt_now or ["background"],
                        "hit": bool(hit), "new_steps": [e["description"] for e in new]}
            if feat is not None:
                cls = feat["cls"]
                self.feat = {"t": t, "q": round(float(feat["q"]), 2),
                             "p": [round(float(cls[3 * k + 1]), 3) for k in range(10)]}
            self.lat = {k: round(v, 1) for k, v in lat.items()}
            self.lat_hist.append(lat["end_to_end_ms"])

    def belief(self, tr):
        top = sorted(tr.belief.items(), key=lambda kv: -kv[1])[:3]
        rep = tr.reported
        out = []
        for s, lp in top:
            diff = [("+" if s[k] == 1 else "-") + COMPONENTS[k] for k in range(11) if s[k] != rep[k]]
            out.append({"code": code(s), "p": round(float(np.exp(lp)), 3), "diff": diff})
        return out

    def snapshot(self, since_run=0):
        with self.lock:
            t = self.t
            lh = self.lat_hist
            pl, xl = official_pairs(self.gt, self.tr.y_hat)
            po, xo = official_pairs(self.gt, self.tr_or.y_hat)
            return {
                "pairs_live": pl, "extra_live": xl, "pairs_oracle": po, "extra_oracle": xo,
                "t": t, "done": self.done, "paused": self.paused, "error": self.error,
                "now": self.now, "feat": self.feat, "lat": self.lat,
                "lat_stats": ({"mean_ms": round(float(np.mean(lh)), 1), "median_ms": round(float(np.median(lh)), 1),
                               "p95_ms": round(float(np.percentile(lh, 95)), 1)} if lh else {}),
                "runs": [list(r) for r in self.runs[max(0, since_run - 1):]], "runs_from": max(0, since_run - 1),
                "ar_acc": round(self.ar_hits / (t + 1), 3) if t >= 0 else None,
                "pred": list(self.tr.y_hat), "pred_err": list(self.tr.y_hat_err), "pred_oracle": list(self.tr_or.y_hat),
                "state": code(self.tr.reported), "state_oracle": code(self.tr_or.reported),
                "state_gt": state_after(self.proc, self.gt, t),
                "state_gt_err": state_after_err(self.proc, self.gt_err, t),
                "belief": self.belief(self.tr), "alerts": list(self.alerts),
                "asd_now": self.asd.get(t) if t >= 0 else None,
                "final": self.final,
            }


class App:
    session: Session | None = None
    vision = None
    vlock = threading.Lock()

    def get_vision(self):
        with self.vlock:
            if self.vision is None:
                print("loading localizer + verifiers on GPU ...", flush=True)
                self.vision = LiveVision()
            return self.vision


APP = App()


def catalog():
    out = {}
    for split in ("val", "test"):
        live = {r["video"]: r for r in json.loads((RES / f"v14_predictedAR_{split}.json").read_text())["videos"]}
        gp = RES / f"v14_gtAR_{split}.json"
        orc = {r["video"]: r for r in json.loads(gp.read_text()).get("videos", [])} if gp.exists() else {}
        out[split] = [{"video": v, "errors": live.get(v, {}).get("errors"),
                       "live": {k: live[v][k] for k in ("pos", "f1", "delay_frames")} if v in live else None,
                       "oracle": {k: orc[v][k] for k in ("pos", "f1", "delay_frames")} if v in orc else None}
                      for v in list_videos(split) if (LIVE_AR / v / "ar.jsonl").exists()]
    return out


def summary():
    s = {}
    for split in ("val", "test"):
        live = json.loads((RES / f"v14_predictedAR_{split}.json").read_text())["summary"]
        gp = RES / f"v14_gtAR_{split}.json"
        orc = json.loads(gp.read_text()).get("summary", {}) if gp.exists() else {}
        s[split] = {"live": live, "oracle": orc}
    s["B3"] = B3
    lat = json.loads((RES / "latency_v14.json").read_text())
    s["latency_v9"] = {"mean_ms": lat["mean_end_to_end_ms"], **lat["end_to_end"]}
    return s


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), status=status)

    def do_GET(self):
        u = urlparse(self.path)
        q = dict(x.split("=", 1) for x in u.query.split("&") if "=" in x)
        if u.path in ("/", "/index.html"):
            return self._send((HERE / "ui" / "v14.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/catalog":
            return self._json(catalog())
        if u.path == "/api/summary":
            return self._json(summary())
        s = APP.session
        if u.path == "/api/info":
            return self._json(s.info() if s else {"idle": True})
        if u.path == "/api/state":
            return self._json(s.snapshot(int(q.get("runs", 0))) if s else {"idle": True})
        if u.path == "/api/frame.jpg":
            return self._send(s.jpg if s and s.jpg else b"", "image/jpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        u = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if u.path == "/api/start":
            if APP.session:
                APP.session.stop = True
            vmode = body.get("verifier", "cached")
            try:
                vision = APP.get_vision() if vmode == "live" else None
                APP.session = Session(body.get("split", "test"), body["video"], vmode,
                                      float(body.get("speed", 1.0)), vision)
            except Exception as e:
                traceback.print_exc()
                return self._json({"error": str(e)}, 400)
            return self._json({"ok": True})
        if u.path == "/api/pause" and APP.session:
            APP.session.paused = not APP.session.paused
        if u.path == "/api/speed" and APP.session:
            APP.session.speed = float(body.get("speed", 1.0))
        if u.path == "/api/stop" and APP.session:
            APP.session.stop = True
        self._json({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--preload", action="store_true", help="load the GPU verifiers at start")
    a = ap.parse_args()
    if a.preload:
        APP.get_vision()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"v14 + teammate-AR demo on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
