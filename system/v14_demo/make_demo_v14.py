"""Build demo_v14.py and ui/v14.html from the server's v9 demo (industreal_extra/v9/PSR/demo) by explicit replacements."""
from pathlib import Path

s = Path("demo_v9_orig.py").read_text(encoding="utf-8")
R = [
('''"""Live demo of frozen v9 with the teammate's causal AR predictions (the deployable setting).

    python demo_v9.py            ->  http://127.0.0.1:8766

Per frame, strictly in order: decode -> teammate AR top-1 (precomputed causal output, ar.jsonl) ->
every 2nd frame: localizer + both DINOv2 verifiers -> v9 state filter.  Exactly the evaluated pipeline
(eval_v6_livear.py); nothing is tuned here.''',
'''"""Live demo of frozen v14 with the teammate's causal AR predictions (the deployable setting).
Made from the v9 demo (industreal_extra/v9/PSR/demo/demo_v9.py); the frozen package is read from PSR_v14/, unchanged.

    python demo_v14.py           ->  http://127.0.0.1:8767

Per frame, strictly in order: decode -> teammate AR top-1 (precomputed causal output, ar.jsonl) ->
even frames: localizer + both DINOv2 verifiers; odd frames: official IndustReal ASD -> v14 state filter ->
v14 mistake alerts (task-graph rules on the steps reported so far; the end-of-recording check when the video ends).
Exactly the evaluated pipeline (eval_v14.py); nothing is tuned here.'''),
('''Verifier source: "live" runs the two verifiers on the GPU now; "cached" reuses the verifier outputs that
the reported scores were computed from (cache/verif_*), which is faster and reproduces those scores exactly.''',
'''Verifier source: "live" runs the two verifiers and the ASD on the GPU now; "cached" reuses the verifier and ASD outputs
that the reported scores were computed from (cache/verif_*, cache/asd_*), which reproduces those scores exactly.
Frames come from <INDUSTREAL_DATA>/<split>/<rid>/rgb/ if present, else from the recording's .mp4 (frames_compat.py).'''),
('''import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))''',
'''import sys
HERE0 = Path(__file__).resolve().parent
PSR_V14 = Path(os.environ.get("PSR_V14", HERE0.parent / "psr_v14"))     # the frozen v14 package, read only
sys.path.insert(0, str(PSR_V14)); sys.path.insert(0, str(HERE0))
import frames_compat  # noqa: F401,E402  (frames from video when the dataset has no JPEG frames)'''),
("from psr.filter_tracker import FilterConfig, FilterTracker", "from psr.filter_tracker import FilterConfig"),
("from psr.vision import CACHE, crop",
 "from psr.vision import CACHE, crop\nfrom psr.v11 import mistake_alerts\nfrom psr.v14 import V14Tracker, asd_states, ASD_CLASSES"),
('''HERE = Path(__file__).resolve().parent            # demo/
ROOT = HERE.parent
FZ = ROOT / "model"
WEIGHTS = ROOT / "weights"
RES = ROOT / "results"''',
'''HERE = HERE0                                      # demo folder
ROOT = PSR_V14
FZ = ROOT / "model"
WEIGHTS = Path(os.environ.get("PSR_WEIGHTS", HERE.parent / "weights"))   # verifier + ASD weights (not in the package)
RES = ROOT / "results"'''),
('''def make_tracker(video):
    model = Blend(GroupModel(FZ / "tables.json"), LRGroupModel(path=FZ / SM["lr_path"]), SM["w_lr"])
    tcfg = TrackerConfig(w_action=SM["w_action"], w_since=SM["w_since"], null_bias=SM["null_bias"])
    return FilterTracker(procedure_type(video), FilterConfig(**CFG["filter"]), model, tcfg)''',
'''AC = CFG["asd"]


def make_tracker(video, asd):
    model = Blend(GroupModel(FZ / "tables.json"), LRGroupModel(path=FZ / SM["lr_path"]), SM["w_lr"])
    tcfg = TrackerConfig(w_action=SM["w_action"], w_since=SM["w_since"], null_bias=SM["null_bias"])
    return V14Tracker(procedure_type(video), FilterConfig(**CFG["filter"]), model, tcfg, d0=CFG["v11"]["d0"],
                      asd=asd, w=AC["w"], eps=AC["eps"], rate=AC["rate"])


def cached_asd(split, video):
    return asd_states(np.load(CACHE / f"asd_{split}" / f"{video}.npz")["rows"], AC["cmin"])'''),
('''    z = np.load(CACHE / "crops" / split / f"{video}.npz")
    boxes = {int(t): [float(x) for x in b] for t, b in zip(z["t"], z["box"])}''',
'''    boxes = {}
    if (CACHE / "crops" / split / f"{video}.npz").exists():          # localizer boxes: display only, optional
        z = np.load(CACHE / "crops" / split / f"{video}.npz")
        boxes = {int(t): [float(x) for x in b] for t, b in zip(z["t"], z["box"])}'''),
('''        self.vv = VisionVerifier(bank="exemplar")''',
'''        self.vv = VisionVerifier(bank="exemplar")
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
        return st, (time.perf_counter() - a) * 1000'''),
('''        self.cache = cached_feats(split, video) if vmode == "cached" else None
        self.tr = make_tracker(video)            # v9 + teammate AR (deployable)
        self.tr_or = make_tracker(video)         # v9 + GT AR (oracle diagnostic)''',
'''        self.cache = cached_feats(split, video) if vmode == "cached" else None
        self.asd = cached_asd(split, video) if vmode == "cached" else {}      # live: filled frame by frame
        self.tr = make_tracker(video, self.asd)       # v14 + teammate AR (deployable)
        self.tr_or = make_tracker(video, self.asd)    # v14 + GT AR (oracle diagnostic)
        self.alerts = []                              # v14 mistake alerts so far (causal)'''),
('''            self.tr.finish(self.n - 1); self.tr_or.finish(self.n - 1)''',
'''            self.tr.finish(self.n - 1); self.tr_or.finish(self.n - 1)
            alerts = mistake_alerts(self.video, self.tr, self.n)            # with the end-of-recording check
            with self.lock:
                self.alerts = alerts'''),
('''        elif t % 2 == 0:
            feat, vl = self.vision(bgr); lat.update(vl)''',
'''        elif t % 2 == 0:
            feat, vl = self.vision(bgr); lat.update(vl)
        elif AC["rate"] == 1 or t % 2 == 1:
            st, ms = self.vision.asd_state(bgr); lat["asd_ms"] = ms
            if st is not None:
                self.asd[t] = st'''),
('''        new = self.tr.update(t, names, feat)
        lat["filter_ms"] = (time.perf_counter() - d) * 1000''',
'''        new = self.tr.update(t, names, feat)
        if new:                                    # alerts from the steps reported so far (no end-of-recording check)
            al = mistake_alerts(self.video, self.tr, t + 1, end_check=False)
            with self.lock:
                self.alerts = al
        lat["filter_ms"] = (time.perf_counter() - d) * 1000'''),
('''                "belief": self.belief(self.tr),''',
'''                "belief": self.belief(self.tr), "alerts": list(self.alerts),
                "asd_now": self.asd.get(t) if t >= 0 else None,'''),
('''        live = {r["video"]: r for r in json.loads((RES / f"v9_predictedAR_{split}.json").read_text())["videos"]}
        orc = {r["video"]: r for r in json.loads((RES / f"v9_gtAR_{split}.json").read_text()).get("videos", [])}''',
'''        live = {r["video"]: r for r in json.loads((RES / f"v14_predictedAR_{split}.json").read_text())["videos"]}
        gp = RES / f"v14_gtAR_{split}.json"
        orc = {r["video"]: r for r in json.loads(gp.read_text()).get("videos", [])} if gp.exists() else {}'''),
('''        live = json.loads((RES / f"v9_predictedAR_{split}.json").read_text())["summary"]
        orc = json.loads((RES / f"v9_gtAR_{split}.json").read_text())
        orc = orc.get("summary", orc)''',
'''        live = json.loads((RES / f"v14_predictedAR_{split}.json").read_text())["summary"]
        gp = RES / f"v14_gtAR_{split}.json"
        orc = json.loads(gp.read_text()).get("summary", {}) if gp.exists() else {}'''),
('''    lat = json.loads((RES / "latency_v9.json").read_text())''', '''    lat = json.loads((RES / "latency_v14.json").read_text())'''),
('''            return self._send((HERE / "ui" / "v9.html").read_bytes(), "text/html; charset=utf-8")''',
'''            return self._send((HERE / "ui" / "v14.html").read_bytes(), "text/html; charset=utf-8")'''),
('''    ap.add_argument("--port", type=int, default=8766)''', '''    ap.add_argument("--port", type=int, default=8767)'''),
('''    print(f"v9 + teammate-AR demo on http://127.0.0.1:{a.port}", flush=True)''',
'''    print(f"v14 + teammate-AR demo on http://127.0.0.1:{a.port}", flush=True)'''),
]
for a, b in R:
    assert s.count(a) == 1, ("not found once", a[:80])
    s = s.replace(a, b)
Path("demo_v14.py").write_text(s, encoding="utf-8")

h = Path("ui/v9_orig.html").read_text(encoding="utf-8")
for a, b in [("v9 PSR live demo", "v14 PSR live demo"), ("IndustReal PSR — v9 live demo", "IndustReal PSR — v14 live demo"),
             ("Input to v9:", "Input to v14:"), ("v9 + teammate AR", "v14 + teammate AR"), ("v9 + team AR", "v14 + team AR"),
             ("v9 + GT AR", "v14 + GT AR"), ("→ v9 input", "→ v14 input"), ("→ v9 gets no action", "→ v14 gets no action"),
             ("v9 does not predict errors", "v14 alerts are listed below"), ("Benchmarked v9 mean", "Benchmarked v14 mean")]:
    h = h.replace(a, b)
panel = '''  <section class="card full">
    <h2>Mistake alerts (v14) <span class="hint">task-graph rules on the steps reported so far; end-of-recording check when the video ends</span></h2>
    <div id="alerts" class="note">none yet</div>
    <div class="note" id="asdnow"></div>
  </section>
'''
anchor = '''  <section class="card full">
    <h2>Whole test set'''
assert h.count(anchor) == 1
h = h.replace(anchor, panel + anchor)
js_anchor = "function render(){"
assert h.count(js_anchor) == 1
h = h.replace(js_anchor, '''function renderAlerts(){
  const a = (ST && ST.alerts) || [];
  $("alerts").innerHTML = a.length ? a.map(x => `<div><b>${fmtT(Math.round(x.t * 10))}</b> · ${x.type.replace(/_/g, " ")} · ${(x.step || x.component || "").replace(/_/g, " ")}</div>`).join("") : "none yet";
  $("asdnow").textContent = ST && ST.asd_now ? `ASD this frame: ${ST.asd_now}` : "";
}
''' + js_anchor + "\n  try { renderAlerts(); } catch (e) {}")
Path("ui/v14.html").write_text(h, encoding="utf-8")
print("ok;", h.count("v9"), "remaining 'v9' mentions in the page")
