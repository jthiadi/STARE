"""The card overlay (assist/tools/overlay_cards.py) in the browser: pick a recording, a run and a view, press play.

Each frame is drawn by overlay_cards' own functions, in the same order as its ``render()`` loop (draw_video, the CAD inset,
the final-label chip, draw_panel, card_text, the timeline and its final-label strip, with_shadows), so the picture is the one
its MP4 shows; it is streamed to the page as MJPEG instead of written to a file. Nothing in overlay_cards, render_overlay,
cad_state or end_label_view is changed. Views, as six_cases makes them:
  replay   3D state inset + "Final labels" row over each whole action (six_cases --replay; filled in after each action ends)
  final    3D state inset + live labels + "Final label" chip 1.2 s after each action ends (six_cases --final)
  cad      3D state inset only (six_cases --cad)        plain   no inset (overlay_cards default)

The full system's results come from the GPU server (v14_integration/assistant/live, job service on port 8020 through the
SSH tunnel): "Compute on server" runs every model on the video there; the results are then copied here and played.

    python -m assist.tools.cards_web [--port 8010] [--server http://127.0.0.1:8020]     # open http://localhost:8010
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import urllib.request
import zipfile

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from assist.common import load_config, read_jsonl
from assist.engine.graph import load_tasks
from assist.evaluation.eval import gt_mistakes
from assist.industreal import SPLITS, gt_actions, task_of
from assist.tools.cad_state import inset, shown_state, state_track
from assist.tools.end_label_view import draw_chip, draw_strip, final_events, shown_final, with_times
from assist.tools.overlay_cards import (PX, STATES, TL_LEFT, TL_RIGHT, build_timeline, card_text, draw_panel, draw_video,
                                        label_column, timeline_at, with_shadows)
from assist.tools.render_overlay import H, PAPER, VH, VW, W, active_event, current_alert, status_at

CFG = load_config()
TASKS = load_tasks(CFG["graph_path"])
OUT = CFG["paths"]["out"]
FULL = "pred_vjepap_asdv14"
RUN_NAMES = {FULL: "Full system: TCN + V-JEPA 2 actions + v14 PSR",
             "pred_asdv14": "TCN actions + v14 PSR", "pred_v14": "v14 PSR alone", "pred_asd": "TCN actions + released ASD",
             "pred_vjepap_asd": "TCN + V-JEPA 2 actions + released ASD", "pred_vjepap_asdv9": "TCN + V-JEPA 2 actions + v9 PSR",
             "gt_gt": "Oracle (annotated actions and states)"}
VIEWS = {"replay": dict(cad=True, action_end=False, final_actions=True), "final": dict(cad=True, action_end=True, final_actions=False),
         "cad": dict(cad=True, action_end=False, final_actions=False), "plain": dict(cad=False, action_end=False, final_actions=False)}
ALL = {rid: split for split in ("test", "val", "train") for rid in SPLITS[split]}
SERVER = "http://127.0.0.1:8020"
app = FastAPI()
_ctx: dict = {}
_lock = threading.Lock()


def runs_for(rid: str) -> list[str]:
    return [r for r in RUN_NAMES if (OUT / rid / f"engine_{r}.json").exists()]


def views_for(rid: str) -> list[str]:
    has_end = (OUT / rid / "action_end.jsonl").exists()
    return [v for v in VIEWS if has_end or not (VIEWS[v]["action_end"] or VIEWS[v]["final_actions"])]


class Replay:
    """Everything overlay_cards.render() prepares before its frame loop, for one view (tracking on)."""

    def __init__(self, rid: str, run: str, view: str):
        cfg = CFG
        opt = VIEWS[view]
        self.cad, self.action_end, self.final_actions = opt["cad"], opt["action_end"], opt["final_actions"]
        self.task = task = TASKS[task_of(rid)]
        out_dir = OUT / rid
        res = json.loads((out_dir / f"engine_{run}.json").read_text(encoding="utf-8"))
        self.res, self.rid = res, rid
        self.cad_track, self.categories = [], ()
        if self.cad:
            from assist.run_engine import ASD_FILES
            self.categories = tuple(cfg["asd_categories"])
            self.cad_track = state_track(read_jsonl(out_dir / ASD_FILES[res["state"]]), list(self.categories))
        self.end_rows = with_times(read_jsonl(out_dir / "action_end.jsonl")) if self.action_end else []
        self.hands = read_jsonl(out_dir / "hands.jsonl") if (out_dir / "hands.jsonl").exists() else []
        self.asd = read_jsonl(out_dir / "asd.jsonl") if res["state"] == "asd" else []
        gts = gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], cfg["fps"], res["duration_s"],
                          cfg["engine"]["order_grace_s"])
        self.duration = res["duration_s"]
        self.events = final_events(read_jsonl(out_dir / "action_end.jsonl")) if self.final_actions else res["events"]
        self.full, self.future = build_timeline({**res, "events": self.events}, gt_actions(rid, cfg["paths"]["labels"], cfg["fps"]),
                                                gts, self.duration)
        self.mistakes = [a for a in res["alerts"] if a["kind"] == "mistake"]
        self.shown = [a for a in res["alerts"] if a["kind"] != "status"]
        self.title = f"{'Assembly' if task.name == 'assy' else 'Maintenance'} video"
        oracle = not res["actions"].startswith("pred")
        self.labels = {s: label_column(s, oracle, self.action_end, "Final labels" if self.final_actions else None) for s in STATES}
        self.panels: dict = {}
        self.n = int(round(self.duration * cfg["fps"]))

    def add_cad(self, image: Image.Image, at: float) -> None:
        if self.cad:
            card = inset(*shown_state(self.cad_track, at), self.categories)
            image.paste(card, (VW - card.width - 16, VH - card.height - 16), card)

    def add_final_chip(self, image: Image.Image, at: float) -> None:
        row = shown_final(self.end_rows, at) if self.action_end else None
        if row:
            draw_chip(image, row["label"])

    def timeline(self, at: float, labels_image: Image.Image) -> Image.Image:
        image = timeline_at(self.full, self.future, at, self.duration, labels_image)
        if self.action_end:
            draw_strip(image, self.end_rows, at, self.duration, TL_LEFT, TL_RIGHT)
        return image

    def frame(self, i: int, frame_bgr, last_asd) -> Image.Image:
        """the body of render()'s loop for frame i"""
        task, t = self.task, i / CFG["fps"]
        status, verified = status_at(self.res["log"], t)
        flagged = {a["step"] for a in self.mistakes if a["t"] <= t and a["step"]}
        alert = current_alert(self.shown, t)
        nxt = [s for s in task.order if status.get(s) != "done" and all(status.get(p) == "done" for p in task.steps[s].requires)]
        card = card_text(alert, task, nxt, False, 0)
        key = (tuple(sorted(status.items())), tuple(sorted(verified)), tuple(sorted(flagged)), tuple(card.values()))
        if key not in self.panels:
            self.panels = {key: draw_panel(task, self.title, status, verified, flagged, card)}
        canvas = Image.new("RGB", (W, H), PAPER)
        row = last_asd if last_asd and t - last_asd["t"] <= 1.5 else None
        hands = self.hands[i]["hands"] if i < len(self.hands) else []
        canvas.paste(draw_video(frame_bgr, active_event(self.events, t), card["state"], hands, row), (0, 0))
        self.add_cad(canvas, t)
        self.add_final_chip(canvas, t)
        canvas.paste(self.panels[key], (PX, 0))
        canvas.paste(self.timeline(t, self.labels[card["state"]]), (0, VH))
        with_shadows(canvas)
        return canvas

    def end_card(self, last_frame) -> Image.Image:
        """render()'s end card (the summary shown after the video ends)"""
        task = self.task
        status, verified = status_at(self.res["log"], self.duration + 1)
        card = card_text(None, task, [], True, len(self.mistakes))
        end = Image.new("RGB", (W, H), PAPER)
        end.paste(draw_video(last_frame, None, card["state"], [], None), (0, 0))
        self.add_cad(end, self.duration)
        self.add_final_chip(end, self.duration)
        end.paste(draw_panel(task, self.title, status, verified, {a["step"] for a in self.mistakes if a["step"]}, card), (PX, 0))
        end.paste(self.timeline(self.duration, self.labels[card["state"]]), (0, VH))
        with_shadows(end)
        return end


def replay(rid: str, run: str, view: str) -> Replay:
    with _lock:
        key = (rid, run, view)
        if key not in _ctx:
            if len(_ctx) > 6:
                _ctx.clear()
            _ctx[key] = Replay(rid, run, view)
        return _ctx[key]


def jpeg(img: Image.Image, quality: int) -> bytes:
    b = io.BytesIO(); img.save(b, "JPEG", quality=quality); return b.getvalue()


def _server(path: str, method: str = "GET", timeout: float = 10):
    req = urllib.request.Request(SERVER + path, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


@app.get("/api/recordings")
def recordings() -> list[dict]:
    try:
        remote = {r["rid"]: r for r in json.loads(_server("/api/recordings"))}
    except Exception:
        remote = None
    out = []
    for rid, split in ALL.items():
        if not (CFG["paths"]["videos"] / f"{rid}.mp4").exists():
            continue
        runs = runs_for(rid)
        out.append({"rid": rid, "split": split, "task": task_of(rid), "runs": [{"run": r, "name": RUN_NAMES[r]} for r in runs],
                    "views": views_for(rid), "full_local": FULL in runs,
                    "server": (remote or {}).get(rid, {}).get("state") if remote is not None else "server not reachable"})
    return out


@app.post("/api/compute/{rid}")
def compute(rid: str) -> dict:
    if rid not in ALL:
        raise HTTPException(404, "unknown recording")
    try:
        return json.loads(_server(f"/api/run/{rid}", "POST"))
    except Exception as e:
        raise HTTPException(502, f"server not reachable ({e}); is the SSH tunnel to port 8020 open?")


@app.get("/api/progress/{rid}")
def progress(rid: str) -> dict:
    try:
        s = json.loads(_server(f"/api/status/{rid}"))
    except Exception as e:
        raise HTTPException(502, f"server not reachable ({e})")
    if s["state"] == "done" and not (OUT / rid / f"engine_{FULL}.json").exists():
        z = zipfile.ZipFile(io.BytesIO(_server(f"/api/bundle/{rid}", timeout=300)))
        (OUT / rid).mkdir(parents=True, exist_ok=True)
        z.extractall(OUT / rid)
        with _lock:
            for k in [k for k in _ctx if k[0] == rid]:
                del _ctx[k]
        s["synced"] = True
    return s


@app.get("/api/info")
def info(rid: str, run: str, view: str = "replay") -> dict:
    if run not in runs_for(rid) or view not in views_for(rid):
        raise HTTPException(404, "no such run / view for this recording")
    r = replay(rid, run, view)
    return {"duration": r.duration, "frames": r.n, "fps": CFG["fps"]}


@app.get("/stream")
def stream(rid: str, run: str, view: str = "replay", start: float = 0.0, speed: float = 1.0, quality: int = 85, still: int = 0):
    if run not in runs_for(rid) or view not in views_for(rid):
        raise HTTPException(404, "no such run / view for this recording")
    r = replay(rid, run, view)

    def gen():
        cap = cv2.VideoCapture(str(CFG["paths"]["videos"] / f"{rid}.mp4"))
        i = max(0, min(int(round(start * CFG["fps"])), r.n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        k, last_asd, last_frame = 0, None, None
        while k < len(r.asd) and r.asd[k]["t"] <= i / CFG["fps"]:
            last_asd = r.asd[k]; k += 1
        nxt = time.perf_counter()
        try:
            while i < r.n:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / CFG["fps"]
                while k < len(r.asd) and r.asd[k]["t"] <= t:
                    last_asd = r.asd[k]; k += 1
                last_frame = frame
                body = jpeg(r.frame(i, frame, last_asd), quality)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n"
                       + body + b"\r\n")
                i += 1
                if still:                                  # paused / seek preview: one frame only
                    return
                if speed > 0:                              # pace to video time x speed; never burst to catch up
                    nxt += 1 / (CFG["fps"] * speed)
                    now = time.perf_counter()
                    if nxt > now:
                        time.sleep(nxt - now)
                    elif now - nxt > 0.5:
                        nxt = now
            if last_frame is not None:
                body = jpeg(r.end_card(last_frame), quality)
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body + b"\r\n"
        finally:
            cap.release()

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Assembly assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{margin:0;background:#f1f0ec;font:15px "Segoe UI",system-ui,sans-serif;color:#222}
 header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd}
 header b{font-size:17px;margin-right:6px} select,button,input{font:inherit;padding:5px 8px}
 button.primary{background:#d9381e;color:#fff;border:0;border-radius:4px;padding:6px 14px}
 #wrap{padding:12px 16px} #view{width:100%;max-width:1920px;aspect-ratio:16/9;background:#ddd;display:block}
 #seek{flex:1;min-width:180px} .t{font-variant-numeric:tabular-nums;min-width:90px} #msg{padding:6px 16px;color:#555}
</style></head><body>
<header><b>Assembly assistant</b>
 <select id="rid" title="Recording"></select><select id="run" title="System"></select>
 <select id="viewsel" title="View"><option value="replay">3D state + final labels (replay)</option>
  <option value="final">3D state + live labels + final-label chip</option><option value="cad">3D state</option>
  <option value="plain">No 3D state</option></select>
 <select id="speed"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select>
 <button class="primary" id="play">Play</button><button id="stop">Stop</button><button id="compute">Compute on server</button>
 <input type="range" id="seek" min="0" max="100" step="0.1" value="0"><span class="t" id="time">0:00</span>
</header>
<div id="msg"></div>
<div id="wrap"><img id="view" alt=""></div>
<script>
const $ = id => document.getElementById(id); let REC = [], DUR = 0, T0 = 0, started = 0, playing = false, poller = null;
const fmt = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`;
const cur = () => REC.find(x => x.rid === $("rid").value);
async function load(keep){ const sel = $("rid").value; REC = await (await fetch("/api/recordings")).json();
  $("rid").innerHTML = ["test","val","train"].map(sp => `<optgroup label="${sp}">` + REC.filter(r => r.split === sp).map(r =>
    `<option value="${r.rid}">${r.rid} (${r.task === "assy" ? "assembly" : "maintenance"})${r.full_local ? "" : " · " + (r.server || "")}</option>`).join("") + "</optgroup>").join("");
  if (keep && sel) $("rid").value = sel; fillRuns(); }
function fillRuns(){ const r = cur(); if (!r) return;
  $("run").innerHTML = r.runs.map(x => `<option value="${x.run}">${x.name}</option>`).join("") || `<option value="">(nothing computed yet)</option>`;
  [...$("viewsel").options].forEach(o => o.disabled = !r.views.includes(o.value));
  if (!r.views.includes($("viewsel").value)) $("viewsel").value = ["replay", "final", "cad", "plain"].find(v => r.views.includes(v));
  if (r.full_local && $("run").querySelector('option[value="pred_vjepap_asdv14"]')) $("run").value = "pred_vjepap_asdv14";
  $("compute").style.display = r.full_local ? "none" : "";
  if (!r.full_local && r.server === "done") { $("msg").textContent = "Copying this recording's results from the server ..."; compute(); }
  else $("msg").textContent = r.full_local ? "" : `Full system not computed for this recording yet (server: ${r.server}). Press "Compute on server" (about 20-30 min while the precompute runs).`;
  if (r.runs.length) info(); else { stop(); $("view").removeAttribute("src"); } }
async function info(){ const i = await (await fetch(`/api/info?rid=${$("rid").value}&run=${$("run").value}&view=${$("viewsel").value}`)).json();
  DUR = i.duration; $("seek").max = DUR; stop(); $("seek").value = 0; $("time").textContent = `0:00 / ${fmt(DUR)}`; play(0, true); }
function src(from, still){ return `/stream?rid=${$("rid").value}&run=${$("run").value}&view=${$("viewsel").value}&start=${from}&speed=${still ? 0 : $("speed").value}` + (still ? "&still=1" : ""); }
function play(from, still){ $("view").src = src(from, still); T0 = from; started = performance.now(); playing = !still; $("play").textContent = playing ? "Pause" : "Play"; }
function stop(){ playing = false; $("play").textContent = "Play"; $("view").removeAttribute("src"); }
function curT(){ return playing ? Math.min(DUR, T0 + (performance.now() - started) / 1000 * $("speed").value) : +$("seek").value; }
async function compute(){ const rid = $("rid").value; const r = await fetch(`/api/compute/${rid}`, {method: "POST"});
  if (!r.ok) { $("msg").textContent = (await r.json()).detail; return; }
  clearInterval(poller); poller = setInterval(async () => { const q = await fetch(`/api/progress/${rid}`); const s = await q.json();
    if (!q.ok) { $("msg").textContent = s.detail; return; }
    $("msg").textContent = s.state === "done" ? "Done: results copied from the server." : s.state === "failed" ? `Failed at ${s.stage}: see the server log.` :
      `${rid}: ${s.state}${s.stage ? " · running " + s.stage : ""} · finished ${s.done.length}/${s.stages.length} (${s.done.join(", ") || "none"})`;
    if (s.state === "done" || s.state === "failed") { clearInterval(poller); if (s.state === "done") await load(true); } }, 5000); }
$("rid").onchange = fillRuns; $("run").onchange = info; $("viewsel").onchange = info; $("compute").onclick = compute;
$("play").onclick = () => { if (playing) { const t = curT(); stop(); $("seek").value = t; play(t, true); } else play(+$("seek").value, false); };
$("stop").onclick = () => { stop(); $("seek").value = 0; play(0, true); };
$("seek").onchange = () => play(+$("seek").value, !playing);
$("speed").onchange = () => { if (playing) play(curT(), false); };
setInterval(() => { if (playing) { const t = curT(); $("seek").value = t; $("time").textContent = `${fmt(t)} / ${fmt(DUR)}`; } }, 250);
load(false);
</script></body></html>"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


def main() -> None:
    global SERVER
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--server", default=SERVER, help="the GPU server's job service, through the SSH tunnel")
    a = ap.parse_args()
    SERVER = a.server
    print(f"Open http://localhost:{a.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
