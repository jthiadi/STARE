"""Overlay video of the assistant on one recording, styled as the station's instruction sheet.

Layout (1920 x 1080, H.264): the video (1440 x 810) sits in an L-shaped paper sheet. Right: the task, the
step count, the procedure drawn as a routing line (one node per step, filling in as steps are done) and the
current alert with why + how to fix (or the next step when all is well). Bottom: a timeline of what the
system recognised, the annotated ground truth, its alerts and the real mistakes.
Colour only means status (andon green / amber / red); magenta only marks the part in the worker's hand.
Everything drawn at time t was known at t, except the rows labelled as annotations.

    python -m assist.tools.render_overlay --video 13_main_2_1 --run pred_vjepap_asd
    python -m assist.tools.render_overlay --video 18_assy_2_5 --run gt_gt --snapshot 330
"""
from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from assist.common import ROOT, load_config, read_jsonl
from assist.engine.graph import load_tasks
from assist.evaluation.eval import gt_mistakes
from assist.industreal import gt_actions, task_of

W, H = 1920, 1080
VW, VH = 1440, 810
PX = VW                                  # panel x
PW = W - VW
FONTS = Path("C:/Windows/Fonts")

PAPER, CARD, RULE = (238, 240, 236), (250, 251, 248), (214, 218, 212)
INK, GRAPHITE, FAINT = (28, 36, 48), (110, 118, 130), (190, 195, 190)
GREEN, AMBER, RED, MAGENTA = (31, 157, 87), (229, 160, 25), (211, 38, 58), (194, 24, 91)
RED_TEXT, AMBER_TEXT, GREEN_TEXT = (170, 24, 42), (150, 96, 0), (18, 112, 60)
STATUS = {"stop": RED, "warn": AMBER, "info": GRAPHITE}
STATUS_TEXT = {RED: RED_TEXT, AMBER: AMBER_TEXT, GRAPHITE: GRAPHITE, GREEN: GREEN_TEXT}

VERB_ING = {"take": "Taking", "fit": "Fitting", "plug": "Plugging in", "tighten": "Tightening", "put": "Putting down",
            "pull": "Pulling out", "loosen": "Loosening", "check": "Checking", "browse": "Browsing",
            "align": "Aligning"}
NOUNS = {"pin_short": "short pin", "pin_long": "long pin", "pin_middle": "middle pin", "partial_model": "model",
         "objects": "parts", "instruction": "instructions", "small_screw_pin": "small screw pin"}
IN_HAND = {**NOUNS, "objects": "a part", "partial_model": "the model"}
TAKE, INSTALL, REMOVE = {"take"}, {"fit", "plug", "tighten"}, {"pull", "loosen"}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ----------------------------------------------------------------------------- text helpers
@lru_cache(maxsize=None)
def bahn(size: int, style: str = "SemiBold") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / "bahnschrift.ttf"), size)
    f.set_variation_by_name(style)
    return f


@lru_cache(maxsize=None)
def ui(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / {"regular": "segoeui.ttf", "semibold": "seguisb.ttf"}[weight]), size)


def wrap(text: str, font, width: int, max_lines: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while font.getlength(lines[-1] + "...") > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "..."
    return lines


def clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def describe(label: str) -> str:
    verb, _, noun = label.partition("_")
    noun = NOUNS.get(noun, noun.replace("_", " "))
    return f"{VERB_ING.get(verb, verb.capitalize())} the {noun}"


def verb_color(label: str):
    verb = label.split("_", 1)[0]
    return MAGENTA if verb in TAKE else INK if verb in INSTALL else GRAPHITE if verb in REMOVE else FAINT


def step_label(task, sid: str) -> str:
    name = task.steps[sid].name
    return name[0].upper() + name[1:]


def step_action(task, sid: str) -> str:
    """'install the front bracket (wing beam)' / 'remove the rear wheel assembly'."""
    name = task.steps[sid].name
    return name if task.steps[sid].value == 0 else f"install the {name}"


# ----------------------------------------------------------------------------- state over time
def status_at(log: list, t: float) -> tuple[dict, set]:
    status, verified = {}, set()
    for e in log:
        if e["t"] > t:
            break
        if e["status"] == "verified":
            verified.add(e["step"])
        else:
            status[e["step"]] = e["status"]
            if e["status"] != "done":
                verified.discard(e["step"])
    return status, verified


def alert_key(a: dict) -> tuple:
    return (a["t"], a["type"], a["step"])


def current_alert(alerts: list, t: float) -> dict | None:
    """Mistakes stay on the sheet for 20 s, warnings 12 s, "cannot verify" notes 8 s; most severe wins."""
    keep = {"mistake": 20, "warning": 12, "status": 8}
    live = [a for a in alerts if a["t"] <= t < a["t"] + keep[a["kind"]]]
    live.sort(key=lambda a: ({"mistake": 0, "warning": 1, "status": 2}[a["kind"]], -a["t"]))
    return live[0] if live else None


def alert_color(a: dict):
    return GRAPHITE if a["kind"] == "status" else AMBER if a["kind"] == "warning" else STATUS.get(a["severity"], RED)


# ----------------------------------------------------------------------------- the sheet (right panel)
def draw_panel(task, res: dict, rid: str, t: float, status: dict, verified: set, flagged: dict,
               alert: dict | None, ended: bool) -> Image.Image:
    img = Image.new("RGB", (PW, H), PAPER)
    d = ImageDraw.Draw(img)
    x0 = 36
    d.text((x0, 30), res["task_title"].split(" (")[0], font=bahn(36), fill=INK)
    sub = "Swap the rear chassis for the short one" if task.name == "main" else "Build the car from the base"
    d.text((x0, 76), sub, font=ui(16), fill=GRAPHITE)
    source = ("Live: V-JEPA 2 actions + state detector" if res["actions"] == "pred_vjepa" else
              "Live: recognised actions + state detector" if res["actions"].startswith("pred") else
              "Oracle: annotated actions and state" if res["state"] == "gt" else "Annotated actions + state detector")
    d.text((x0, 98), f"{rid}   {source}", font=ui(14), fill=GRAPHITE)

    done = sum(1 for s in task.order if status.get(s) == "done")
    total = len(task.order)
    big = bahn(84, "Bold Condensed")
    d.text((x0 - 2, 128), str(done), font=big, fill=INK)
    w = big.getlength(str(done))
    d.text((x0 + w + 2, 128), f"/{total}", font=bahn(84, "Light Condensed"), fill=GRAPHITE)
    d.text((x0 + w + bahn(84, "Light Condensed").getlength(f"/{total}") + 14, 186), "steps done",
           font=ui(17), fill=GRAPHITE)

    seg_w = (PW - 2 * x0 - (total - 1) * 4) / total
    for i, s in enumerate(task.order):
        c = RED if s in flagged else GREEN if s in verified else INK if status.get(s) == "done" else RULE
        xa = x0 + i * (seg_w + 4)
        d.rectangle([xa, 236, xa + seg_w, 243], fill=c)

    nxt = [s for s in task.order if status.get(s) != "done"
           and all(status.get(p) == "done" for p in task.steps[s].requires)]
    y = 270
    spine_x = x0 + 12
    rows = []                                # (step, y centre, subgoal index)
    for gi, (gname, members) in enumerate(task.subgoals):
        d.text((x0, y), {"chassis": "Chassis", "front_and_wheels": "Front and wheels", "strip": "Take apart",
                         "rebuild": "Rebuild"}.get(gname, gname), font=ui(15, "semibold"), fill=GRAPHITE)
        y += 30
        for s in [m for m in task.order if m in members]:
            rows.append((s, y + 12, gi))
            y += 36
        y += 8
    for (_, ya, ga), (_, yb, gb) in zip(rows, rows[1:]):
        if ga == gb:                         # the line runs within a subgoal; a gap marks the next one
            d.line([spine_x, ya + 11, spine_x, yb - 11], fill=FAINT, width=2)
    for s, yc, _ in rows:
        st = status.get(s, "pending")
        r = 10
        box = [spine_x - r, yc - r, spine_x + r, yc + r]
        if s in flagged:
            d.ellipse(box, fill=RED)
            d.text((spine_x, yc), "!", font=bahn(17, "Bold"), fill=CARD, anchor="mm")
        elif st == "done":
            d.ellipse(box, fill=GREEN if s in verified else INK)
            d.line([spine_x - 5, yc, spine_x - 1, yc + 4, spine_x + 5, yc - 4], fill=CARD, width=2)
        elif st in ("started", "placed"):
            d.ellipse(box, outline=MAGENTA, width=3)
            d.ellipse([spine_x - 4, yc - 4, spine_x + 4, yc + 4], fill=MAGENTA)
        elif s in nxt:
            d.ellipse(box, fill=PAPER, outline=INK, width=3)
        else:
            d.ellipse(box, fill=PAPER, outline=FAINT, width=2)
        label = step_label(task, s)
        strong = s in nxt or st in ("started", "placed")
        font = ui(17, "semibold") if strong else ui(17)
        color = INK if (strong or st == "done" or s in flagged) else GRAPHITE
        d.text((x0 + 38, yc), label, font=font, fill=color, anchor="lm")
        if s in nxt and st == "pending":
            d.text((PW - x0, yc), "next", font=ui(14), fill=GRAPHITE, anchor="rm")

    card_top = max(y + 10, 650)
    draw_card(d, task, res, t, alert, nxt, ended, card_top)
    return img


def draw_card(d, task, res, t, alert, nxt, ended, top):
    x0, x1, bottom = 36, PW - 28, H - 32
    d.rectangle([x0 - 12, top, x1, bottom], fill=CARD)
    tx, width = x0 + 14, x1 - x0 - 30
    if ended:
        mistakes = [a for a in res["alerts"] if a["kind"] == "mistake"]
        color = RED if mistakes else GREEN
        d.rectangle([x0 - 12, top, x0 - 4, bottom], fill=color)
        y = top + 22
        d.text((tx, y), "Task ended", font=ui(16, "semibold"), fill=STATUS_TEXT[color])
        y += 30
        head = f"{len(mistakes)} mistake{'s' if len(mistakes) != 1 else ''} flagged" if mistakes else \
            "No mistakes flagged"
        for line in wrap(head, bahn(28), width, 2):
            d.text((tx, y), line, font=bahn(28), fill=INK)
            y += 36
        y += 12
        for a in mistakes[:6]:
            for line in wrap(f"{clock(a['t'])}  {a['title']}", ui(16), width, 1):
                d.text((tx, y), line, font=ui(16), fill=INK)
                y += 26
        return
    if alert is None:
        d.rectangle([x0 - 12, top, x0 - 4, bottom], fill=GREEN)
        y = top + 22
        d.text((tx, y), "On track", font=ui(16, "semibold"), fill=GREEN_TEXT)
        y += 30
        if nxt:
            s = nxt[0]
            for line in wrap(f"Next: {step_action(task, s)}", bahn(28), width, 2):
                d.text((tx, y), line, font=bahn(28), fill=INK)
                y += 36
            y += 8
            for line in wrap(task.steps[s].note, ui(16), width, 4):
                d.text((tx, y), line, font=ui(16), fill=GRAPHITE)
                y += 25
            if len(nxt) > 1:
                y += 8
                others = ", ".join(task.steps[o].name for o in nxt[1:3])
                for line in wrap(f"Also allowed now: {others}", ui(15), width, 2):
                    d.text((tx, y), line, font=ui(15), fill=GRAPHITE)
                    y += 23
        else:
            d.text((tx, y), "All steps done", font=bahn(28), fill=INK)
        return
    color = alert_color(alert)
    d.rectangle([x0 - 12, top, x0 - 4, bottom], fill=color)
    y = top + 20
    kind = {"mistake": "Mistake", "warning": "Early warning", "status": "Cannot verify"}[alert["kind"]]
    d.text((tx, y), f"{kind} at {clock(alert['t'])}", font=ui(16, "semibold"), fill=STATUS_TEXT[color])
    y += 30
    for line in wrap(alert["title"], bahn(27), width, 2):
        d.text((tx, y), line, font=bahn(27), fill=INK)
        y += 34
    y += 8
    for line in wrap(alert["why"], ui(16), width, 5):
        d.text((tx, y), line, font=ui(16), fill=INK)
        y += 24
    if alert["fix_steps"] and y < bottom - 60:
        y += 12
        d.text((tx, y), "To fix" if alert["kind"] == "mistake" else "What to do", font=ui(16, "semibold"), fill=INK)
        y += 28
        for i, step in enumerate(alert["fix_steps"], 1):
            lines = wrap(step, ui(16), width - 26, 2)
            if y + 24 * len(lines) > bottom - 14:
                d.text((tx + 26, y), "...", font=ui(16), fill=GRAPHITE)
                break
            d.text((tx, y), f"{i}", font=bahn(17, "Bold"), fill=GRAPHITE)
            for line in lines:
                d.text((tx + 26, y), line, font=ui(16), fill=INK)
                y += 24


# ----------------------------------------------------------------------------- the timeline (bottom strip)
TL_LEFT, TL_RIGHT = 196, VW - 36
LANES = {"live": 30, "gt": 64, "alerts": 112, "real": 152}


def build_timeline(res: dict, gt_segs: list, gts: list, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """(full, future-hidden) timeline images; per frame the full one is shown left of the playhead."""
    def xs(s):
        return TL_LEFT + (TL_RIGHT - TL_LEFT) * min(max(s / duration, 0.0), 1.0)

    def base(with_live: bool) -> Image.Image:
        img = Image.new("RGB", (VW, H - VH), PAPER)
        d = ImageDraw.Draw(img)
        d.line([0, 0, VW, 0], fill=RULE, width=2)
        names = {"live": "Recognised" if res["actions"].startswith("pred") else "Input (annotated)",
                 "gt": "Annotated actions", "alerts": "Alerts", "real": "Real mistakes"}
        for k, y in LANES.items():
            d.text((36, y + 11), names[k], font=ui(15, "semibold" if k in ("live", "alerts") else "regular"),
                   fill=INK if k in ("live", "alerts") else GRAPHITE, anchor="lm")
        d.line([TL_LEFT, 96, TL_RIGHT, 96], fill=RULE, width=1)
        for s in gt_segs:
            d.rectangle([xs(s["start"]), LANES["gt"] + 4, max(xs(s["start"]) + 1, xs(s["end"])), LANES["gt"] + 18],
                        fill=verb_color(s["action"]))
        for g in gts:
            x, y = xs(g["t"]), LANES["real"] + 11
            d.polygon([(x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)], fill=INK)
        step = 60 if duration > 180 else 30
        for s in range(0, int(duration) + 1, step):
            d.line([xs(s), 186, xs(s), 192], fill=GRAPHITE, width=1)
            d.text((xs(s), 196), clock(s), font=ui(13), fill=GRAPHITE, anchor="mt")
        lx = TL_LEFT
        for color, text in ((MAGENTA, "picking up a part"), (INK, "installing"), (GRAPHITE, "removing"),
                            (FAINT, "other")):
            d.rectangle([lx, 238, lx + 18, 248], fill=color)
            d.text((lx + 26, 243), text, font=ui(13), fill=GRAPHITE, anchor="lm")
            lx += 40 + ui(13).getlength(text)
        lx += 16
        for shape, color, text in (("tri", RED, "mistake alert"), ("dot", AMBER, "early warning"),
                                   ("ring", GRAPHITE, "cannot verify"), ("dia", INK, "real mistake")):
            marker(d, shape, lx + 8, 243, color, 6)
            d.text((lx + 22, 243), text, font=ui(13), fill=GRAPHITE, anchor="lm")
            lx += 40 + ui(13).getlength(text)
        if with_live:
            for e in res["events"]:
                d.rectangle([xs(e["start"]), LANES["live"] + 2, max(xs(e["start"]) + 1, xs(e["end"])),
                             LANES["live"] + 20], fill=verb_color(e["label"]))
            for a in sorted(res["alerts"], key=lambda a: {"status": 0, "warning": 1, "mistake": 2}[a["kind"]]):
                shape = {"mistake": "tri", "warning": "dot", "status": "ring"}[a["kind"]]
                marker(d, shape, xs(a["t"]), LANES["alerts"] + 11, alert_color(a), 8)
        return img

    return np.asarray(base(True)).copy(), np.asarray(base(False)).copy()


def marker(d, shape, x, y, color, r):
    if shape == "tri":
        d.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], fill=color)
    elif shape == "dot":
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif shape == "ring":
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
    else:
        d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=color)


def timeline_at(full: np.ndarray, future: np.ndarray, t: float, duration: float) -> Image.Image:
    xp = int(TL_LEFT + (TL_RIGHT - TL_LEFT) * min(t / duration, 1.0))
    arr = future.copy()
    arr[:, :xp] = full[:, :xp]
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.line([xp, 22, xp, 180], fill=INK, width=2)
    label = clock(t)
    tw = ui(14, "semibold").getlength(label) + 14
    d.rectangle([xp - tw / 2, 2, xp + tw / 2, 21], fill=INK)
    d.text((xp, 11), label, font=ui(14, "semibold"), fill=CARD, anchor="mm")
    return img


# ----------------------------------------------------------------------------- the video layer
def corners(d, box, color, length=22, width=4):
    x0, y0, x1, y1 = box
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([x, y, x + dx * length, y], fill=color, width=width)
        d.line([x, y, x, y + dy * length], fill=color, width=width)


def chip(d, xy, text, font, fg, bg, pad=(12, 6)):
    x, y = xy
    w = font.getlength(text)
    h = font.size + 2 * pad[1]
    d.rounded_rectangle([x, y, x + w + 2 * pad[0], y + h], radius=6, fill=bg)
    d.text((x + pad[0], y + h / 2), text, font=font, fill=fg, anchor="lm")
    return x + w + 2 * pad[0], y + h


def draw_video(frame_bgr, t, event, hands, asd_row, alert, new_mistake, task, components) -> Image.Image:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (VW, VH), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sx, sy = VW / 1280, VH / 720

    if asd_row is not None:
        x, y, w, h = asd_row["box"]
        box = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
        bad = asd_row["state"] == "error_state"
        corners(d, box, RED + (255,) if bad else (255, 255, 255, 230))
        if bad:
            text = "State check: something looks mounted wrong"
        else:
            on = asd_row["state"][1:].count("1")          # parts on the base
            text = f"State check: {on} part{'s' if on != 1 else ''} on the model"
        chip(d, (box[0], max(8, box[1] - 38)), text, ui(16), CARD if bad else INK,
             RED + (235,) if bad else CARD + (215,))

    hand_boxes = []
    for hand in hands:
        pts = [(px * VW, py * VH) for px, py in hand]
        for a, b in HAND_EDGES:
            d.line([pts[a], pts[b]], fill=(255, 255, 255, 190), width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 255, 255, 230))
        xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
        hand_boxes.append((min(xs_), min(ys_), max(xs_), max(ys_)))
    if event is not None and hand_boxes and event["label"].split("_", 1)[0] in TAKE | INSTALL | {"put"}:
        box = [min(b[0] for b in hand_boxes) - 14, min(b[1] for b in hand_boxes) - 14,
               max(b[2] for b in hand_boxes) + 14, max(b[3] for b in hand_boxes) + 14]
        corners(d, box, MAGENTA + (255,), 18, 3)
        noun = event["label"].partition("_")[2]
        chip(d, (max(4, box[0]), max(118, box[1] - 38)), f"In hand: {IN_HAND.get(noun, noun.replace('_', ' '))}",
             ui(16, "semibold"), CARD, MAGENTA + (235,))

    # what the system thinks is happening now
    if event is not None:
        x1, y1 = chip(d, (24, 28), describe(event["label"]), bahn(30), CARD, INK + (220,), pad=(16, 10))
        bar_w = 160
        d.rectangle([24, y1 + 8, 24 + bar_w, y1 + 13], fill=(255, 255, 255, 110))
        d.rectangle([24, y1 + 8, 24 + bar_w * min(event["conf"], 1.0), y1 + 13], fill=(255, 255, 255, 240))
        d.text((24 + bar_w + 10, y1 + 10), f"{event['conf']:.0%} sure", font=ui(14, "semibold"),
               fill=(255, 255, 255, 240), anchor="lm")
    else:
        chip(d, (24, 28), "Watching", bahn(30, "Regular"), (255, 255, 255, 235), INK + (150,), pad=(16, 10))

    # andon strip: status of the station at a glance
    color = alert_color(alert) if alert is not None and alert["kind"] != "status" else GREEN
    d.rectangle([0, 0, VW, 8], fill=color + (255,))

    # the one moment of motion: a new mistake rises in from the bottom of the video for 6 s
    if new_mistake is not None:
        age = t - new_mistake["t"]
        ease = min(age / 0.4, 1.0)
        y = VH - 96 + int((1 - ease) * 40)
        alpha = int(240 * ease)
        c = alert_color(new_mistake)
        d.rectangle([0, y, VW, y + 72], fill=c + (alpha,))
        d.text((32, y + 36), new_mistake["title"], font=bahn(32), fill=CARD + (int(255 * ease),), anchor="lm")
        d.text((VW - 32, y + 36), "see the sheet for how to fix it", font=ui(17), fill=CARD + (int(230 * ease),),
               anchor="rm")
    return Image.alpha_composite(base, layer).convert("RGB")


# ----------------------------------------------------------------------------- rendering
def active_event(events: list, t: float):
    cur = [e for e in events if e["known_at"] <= t < e["ended_at"]]
    return cur[-1] if cur else None


def render(rid: str, run: str, cfg: dict, out_path: Path, snapshots: tuple[float, ...] = (),
           end_hold_s: float = 4.0, start: float = 0.0, stop: float | None = None) -> None:
    tasks = load_tasks(cfg["graph_path"])
    task = tasks[task_of(rid)]
    out_dir = cfg["paths"]["out"] / rid
    res = json.loads((out_dir / f"engine_{run}.json").read_text(encoding="utf-8"))
    hands = read_jsonl(out_dir / "hands.jsonl")
    asd = read_jsonl(out_dir / "asd.jsonl") if (out_dir / "asd.jsonl").is_file() else []
    if res["state"] != "asd":
        asd = []                             # oracle runs: the boxes would show the detector, not the state used
    gt_segs = gt_actions(rid, cfg["paths"]["labels"], cfg["fps"])
    gts = gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], cfg["fps"], res["duration_s"],
                      cfg["engine"]["order_grace_s"])
    duration = res["duration_s"]
    full, future = build_timeline(res, gt_segs, gts, duration)
    mistakes = [a for a in res["alerts"] if a["kind"] == "mistake"]

    cap = cv2.VideoCapture(str(cfg["paths"]["videos"] / f"{rid}.mp4"))
    first = int(round(start * cfg["fps"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = int(round((stop or duration) * cfg["fps"]))
    wanted = {round(s * cfg["fps"]) for s in snapshots}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    if not snapshots:
        import imageio_ffmpeg
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(cfg["fps"]), "-i", "-",
                                 "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    panel_cache: dict = {}
    k_asd, last_asd, i = 0, None, first
    canvas = None
    while i < last:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / cfg["fps"]
        if wanted and i > max(wanted):
            break
        if wanted and i not in wanted:
            i += 1
            continue
        while k_asd < len(asd) and asd[k_asd]["t"] <= t:
            last_asd = asd[k_asd]
            k_asd += 1
        asd_row = last_asd if last_asd and t - last_asd["t"] <= 1.5 else None
        status, verified = status_at(res["log"], t)
        flagged = {a["step"]: a for a in mistakes if a["t"] <= t and a["step"]}
        alert = current_alert(res["alerts"], t)
        recent = [a for a in mistakes if 0 <= t - a["t"] < 6]
        key = (tuple(sorted(status.items())), tuple(sorted(verified)), tuple(sorted(flagged)),
               alert_key(alert) if alert else None)
        if key not in panel_cache:
            panel_cache.clear()
            panel_cache[key] = draw_panel(task, res, rid, t, status, verified, flagged, alert, False)
        canvas = Image.new("RGB", (W, H), PAPER)
        canvas.paste(draw_video(frame, t, active_event(res["events"], t), hands[i]["hands"] if i < len(hands) else [],
                                asd_row, alert, recent[-1] if recent else None, task, cfg["components"]), (0, 0))
        canvas.paste(panel_cache[key], (PX, 0))
        canvas.paste(timeline_at(full, future, t, duration), (0, VH))
        if proc:
            proc.stdin.write(canvas.tobytes())
        else:
            canvas.save(out_path.with_name(f"{out_path.stem}_{t:.0f}s.png"))
        i += 1
        if i % 500 == 0:
            print(json.dumps({"video": rid, "frame": i, "t": round(t, 1)}), flush=True)
    cap.release()
    if proc and canvas is not None and stop is None:
        status, verified = status_at(res["log"], duration + 1)
        flagged = {a["step"]: a for a in mistakes if a["step"]}
        end = canvas.copy()
        ImageDraw.Draw(end).rectangle([0, 0, VW, 8], fill=RED if mistakes else GREEN)
        end.paste(draw_panel(task, res, rid, duration, status, verified, flagged, None, True), (PX, 0))
        end.paste(timeline_at(full, future, duration, duration), (0, VH))
        for _ in range(int(end_hold_s * cfg["fps"])):
            proc.stdin.write(end.tobytes())
    if proc:
        proc.stdin.close()
        proc.wait()
        print(json.dumps({"written": str(out_path.relative_to(ROOT))}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--run", default="pred_vjepap_asd", help="engine_<run>.json to show (e.g. pred_asd, gt_gt).")
    parser.add_argument("--snapshot", type=float, nargs="*", default=(), help="Only save PNG stills at these times.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    out = args.out or cfg["paths"]["out"] / "overlays" / f"{args.video}_{args.run}.mp4"
    render(args.video, args.run, cfg, out, tuple(args.snapshot), start=args.start, stop=args.stop)


if __name__ == "__main__":
    main()
