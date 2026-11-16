"""Overlay video with one large status card: green NEXT, amber WARNING or CHECK, red MISTAKE.

The video fills the top left. The action pill, the card, the progress ring and the timeline's labels all take the
station's colour, so its status reads at a glance. In the checklist, steps done correctly fade, so the next step stands
out. Everything drawn at time t was known at t, except the rows labelled as annotations.

    python -m assist.tools.overlay_cards --video 13_main_2_1 --run pred_vjepap_asd      # -> overlays/cards/<video>_<run>.mp4
    python -m assist.tools.overlay_cards --video 13_main_2_1 --snapshot 60 125 130      # PNG stills only
    python -m assist.tools.overlay_cards --video 13_main_2_1 --no-tracking              # without the hands and the state box
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from assist.common import ROOT, load_config, read_jsonl
from assist.engine.graph import load_tasks
from assist.evaluation.eval import gt_mistakes
from assist.industreal import gt_actions, task_of
from assist.tools.cad_state import inset, shown_state, state_track
from assist.tools.end_label_view import STRIP_Y, draw_chip, draw_strip, final_events, shown_final, with_times
from assist.tools.render_overlay import (AMBER, CARD, FAINT, GRAPHITE, H, HAND_EDGES, IN_HAND, INK, MAGENTA, PAPER, PW, PX, RED,
                                         TAKE, INSTALL, VH, VW, W, active_event, bahn, chip, clock, corners, current_alert,
                                         describe, marker, status_at, ui, verb_color, wrap)

# (light corner, deep corner) of the card, the pill's colour, the label's text colour
STATES = {"next": ((176, 219, 148), (84, 170, 52), (112, 170, 84), INK),
          "warn": ((228, 190, 120), (236, 156, 16), (214, 158, 70), INK),
          "mistake": ((222, 128, 112), (226, 52, 22), (200, 44, 44), (255, 255, 255))}
LABEL_FILL = {"next": (146, 202, 124), "warn": (255, 176, 0), "mistake": (206, 14, 14)}
FADE = 0.3                               # how much of a faded row's colour is left
TL_LEFT, TL_RIGHT, LABEL_W = 278, VW - 40, 252
LANES = {"live": 34, "gt": 88, "alerts": 148, "real": 180}
BAR_H = 26
PANEL_SHADOW, LABEL_SHADOW = 22, 16      # widths of the two soft shadows, in pixels


def _names(task, steps: list[str]) -> str:
    """Step names for the card's smaller text, without bracketed asides, so they fit."""
    names = [re.sub(r"\s*\([^()]*\)", "", task.steps[s].name) for s in steps]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _capital(text: str) -> str:
    return text[0].upper() + text[1:]


def _action(task, sid: str) -> str:
    name = task.steps[sid].name
    return _capital(name if task.steps[sid].value == 0 else f"install the {name}")


def card_text(alert: dict | None, task, nxt: list[str], ended: bool, n_mistakes: int) -> dict:
    """What the card says: its colour state, heading, title and smaller text."""
    if ended:
        if n_mistakes:
            word = "mistake" if n_mistakes == 1 else "mistakes"
            return {"state": "mistake", "heading": "TASK ENDED", "title": f"{n_mistakes} {word} flagged", "sub": ""}
        return {"state": "next", "heading": "TASK ENDED", "title": "No mistakes flagged", "sub": ""}
    if alert is None:
        if not nxt:
            return {"state": "next", "heading": "ALL STEPS DONE", "title": "", "sub": ""}
        also = f"Also allowed: {_names(task, nxt[1:3])}" if len(nxt) > 1 else ""
        return {"state": "next", "heading": "NEXT", "title": _action(task, nxt[0]), "sub": also}
    subject = _capital(alert["title"].split(": ", 1)[-1].removeprefix("Check the "))
    first = _names(task, alert.get("missing", []))
    kind = alert["type"]
    if kind == "early_warning":
        step = subject.removesuffix(" is not next")
        step = f"Removing{step[len('Remove'):]}" if step.startswith("Remove ") else step
        return {"state": "warn", "heading": "WARNING", "title": f"Wait: {step} not next", "sub": f"First do: {first}"}
    if kind == "possible_incorrect_install":
        return {"state": "warn", "heading": "CHECK", "title": subject, "sub": "The camera has not confirmed this install."}
    sub = {"wrong_order": f"Out of order. First do: {first}", "skipped_step": "Step skipped. Do it now.",
           "incorrect_install": "Installed wrong. Fix it now.", "wrong_part": "Wrong part. Remove it."}[kind]
    return {"state": "mistake", "heading": "MISTAKE", "title": subject, "sub": sub}


def row_style(sid: str, status: dict, flagged, nxt: list[str]) -> str:
    """How a checklist row looks. Only steps done with nothing wrong fade."""
    st = status.get(sid, "pending")
    if sid in flagged:
        return "flagged"
    if st == "done":
        return "faded"
    if st in ("started", "placed"):
        return "active"
    return "next" if sid in nxt else "pending"


def ring_segments(done: int, total: int) -> list[bool]:
    """One segment per step, clockwise from the top; True is done."""
    return [i < done for i in range(total)]


def _blend(color, amount: float):
    return tuple(int(p + (c - p) * amount) for c, p in zip(color, PAPER))


# ----------------------------------------------------------------------------- the right panel
def _gradient(width: int, height: int, light, deep) -> Image.Image:
    xs, ys = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    mix = np.clip(0.55 * xs + 0.75 * ys - 0.1, 0, 1)[..., None]
    return Image.fromarray((np.array(light) * (1 - mix) + np.array(deep) * mix).astype(np.uint8))


def _warning_icon(d, x: int, y: int, size: int) -> None:
    """A triangle with an exclamation mark, outlined in white."""
    d.line([(x + size // 2, y), (x + size, y + size), (x, y + size), (x + size // 2, y)], fill=(255, 255, 255), width=5,
           joint="curve")
    d.line([(x + size // 2, y + size * 0.35), (x + size // 2, y + size * 0.68)], fill=(255, 255, 255), width=6)
    d.ellipse([x + size // 2 - 4, y + size * 0.76, x + size // 2 + 4, y + size * 0.76 + 8], fill=(255, 255, 255))


def draw_card(img: Image.Image, card: dict) -> None:
    light, deep, _, _ = STATES[card["state"]]
    x0, y0, y1 = 12, 24, 442
    width, height = PW - x0, y1 - y0
    scale = 3
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, (width + 60) * scale, height * scale - 1], radius=28 * scale, fill=255)
    img.paste(_gradient(width, height, light, deep), (x0, y0), mask.resize((width, height), Image.LANCZOS))
    d = ImageDraw.Draw(img)
    white = (255, 255, 255)
    x, text_w, y = x0 + 40, PW - x0 - 60, y0 + 42
    if card["heading"] == "MISTAKE":
        _warning_icon(d, x, y + 4, 56)
        d.text((x + 72, y), card["heading"], font=bahn(66, "Bold"), fill=white)
    elif card["heading"] == "NEXT":
        d.text((x, y), card["heading"], font=bahn(66, "Bold"), fill=white)
        start = x + bahn(66, "Bold").getlength(card["heading"]) + 24
        d.line([(start, y + 36), (PW - 56, y + 36)], fill=white, width=3)
        d.polygon([(PW - 42, y + 36), (PW - 58, y + 28), (PW - 58, y + 44)], fill=white)
    else:
        size = 66 if bahn(66, "Bold").getlength(card["heading"]) <= text_w else 50
        d.text((x, y), card["heading"], font=bahn(size, "Bold"), fill=white)
    y += 88
    if card["title"]:
        for line in wrap(card["title"], bahn(46, "Bold"), text_w, 3):
            d.text((x, y), line, font=bahn(46, "Bold"), fill=white)
            y += 52
    y += 14
    if card["sub"]:
        for line in wrap(card["sub"], bahn(30, "Regular"), text_w, 3):
            if y > y1 - 46:
                break
            d.text((x, y), line, font=bahn(30, "Regular"), fill=white)
            y += 36


def draw_ring(img: Image.Image, done: int, total: int, state: str, title: str) -> None:
    deep = STATES[state][1] if state != "next" else (92, 172, 62)
    scale, r_out, r_in = 3, 88, 44
    size = (2 * r_out + 8) * scale
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(ring)
    c = size // 2
    box = [c - r_out * scale, c - r_out * scale, c + r_out * scale, c + r_out * scale]
    gap = 2.2 if total > 1 else 0
    for i, on in enumerate(ring_segments(done, total)):
        a0, a1 = -90 + 360 * i / total + gap, -90 + 360 * (i + 1) / total - gap
        d.pieslice(box, a0, a1, fill=(*deep, 255) if on else (*(58, 66, 78), 255))
    d.ellipse([c - r_in * scale, c - r_in * scale, c + r_in * scale, c + r_in * scale], fill=(*PAPER, 255))
    ring = ring.resize((size // scale, size // scale), Image.LANCZOS)
    cx, cy = 104, 582
    img.paste(ring, (cx - ring.width // 2, cy - ring.height // 2), ring)
    d = ImageDraw.Draw(img)
    big = bahn(38, "Bold")
    small = bahn(30, "Bold")
    text_w = big.getlength(str(done)) + small.getlength(f"/{total}")
    d.text((cx - text_w / 2, cy - 4), str(done), font=big, fill=INK, anchor="lm")
    d.text((cx - text_w / 2 + big.getlength(str(done)), cy + 6), f"/{total}", font=small, fill=INK, anchor="lm")
    x, y = 206, cy - 44
    for line in title.upper().split():
        d.text((x, y), line, font=bahn(38, "Bold"), fill=INK)
        y += 46


def draw_checklist(img: Image.Image, task, status: dict, verified: set, flagged, nxt: list[str]) -> None:
    d = ImageDraw.Draw(img)
    top, bottom = 702, H - 14
    n_groups, n_rows = len(task.subgoals), len(task.order)
    row = min(36, int((bottom - top - n_groups * 30) / n_rows))
    x_node, x_text, y = 44, 72, top
    rows = []
    for gi, (gname, members) in enumerate(task.subgoals):
        d.text((26, y), {"chassis": "Chassis", "front_and_wheels": "Front and wheels", "strip": "Take apart",
                         "rebuild": "Rebuild"}.get(gname, gname), font=ui(16), fill=GRAPHITE)
        y += 30
        for s in [m for m in task.order if m in members]:
            rows.append((s, y + row // 2 - 2, gi))
            y += row
    for (_, ya, ga), (_, yb, gb) in zip(rows, rows[1:]):
        if ga == gb:
            d.line([x_node, ya + 11, x_node, yb - 11], fill=_blend(FAINT, 0.6), width=2)
    for s, yc, _ in rows:
        style = row_style(s, status, flagged, nxt)
        r = 11
        box = [x_node - r, yc - r, x_node + r, yc + r]
        if style == "flagged":
            d.ellipse(box, fill=RED)
            d.text((x_node, yc), "!", font=bahn(18, "Bold"), fill=CARD, anchor="mm")
        elif style == "faded":
            d.ellipse(box, fill=_blend((31, 157, 87) if s in verified else INK, FADE))
            d.line([x_node - 5, yc, x_node - 1, yc + 4, x_node + 5, yc - 4], fill=PAPER, width=2)
        elif style == "active":
            d.ellipse(box, outline=MAGENTA, width=3)
            d.ellipse([x_node - 4, yc - 4, x_node + 4, yc + 4], fill=MAGENTA)
        elif style == "next":
            d.ellipse(box, fill=PAPER, outline=INK, width=3)
        else:
            d.ellipse(box, fill=PAPER, outline=FAINT, width=2)
        name = _capital(task.steps[s].name)
        font = ui(19, "semibold") if style in ("next", "active", "flagged") else ui(19)
        color = {"faded": _blend(INK, FADE), "pending": GRAPHITE, "flagged": INK}.get(style, INK)
        d.text((x_text, yc), name, font=font, fill=color, anchor="lm")
        if style == "next":
            d.text((PW - 24, yc), "next", font=ui(14), fill=GRAPHITE, anchor="rm")


def draw_panel(task, title: str, status: dict, verified: set, flagged, card: dict) -> Image.Image:
    img = Image.new("RGB", (PW, H), PAPER)
    draw_card(img, card)
    d = ImageDraw.Draw(img)
    d.line([(16, 478), (PW - 16, 478)], fill=(176, 180, 176), width=2)
    d.line([(16, 686), (PW - 16, 686)], fill=(176, 180, 176), width=2)
    done = sum(1 for s in task.order if status.get(s) == "done")
    draw_ring(img, done, len(task.order), card["state"], title)
    nxt = [s for s in task.order if status.get(s) != "done" and all(status.get(p) == "done" for p in task.steps[s].requires)]
    draw_checklist(img, task, status, verified, flagged, nxt)
    return img


# ----------------------------------------------------------------------------- the timeline
def build_timeline(res: dict, gt_segs: list, gts: list, duration: float) -> tuple[np.ndarray, np.ndarray]:
    def xs(s):
        return TL_LEFT + (TL_RIGHT - TL_LEFT) * min(max(s / duration, 0.0), 1.0)

    def base(with_live: bool) -> Image.Image:
        img = Image.new("RGB", (VW, H - VH), PAPER)
        d = ImageDraw.Draw(img)
        for s in gt_segs:
            d.rectangle([xs(s["start"]), LANES["gt"] - BAR_H // 2, max(xs(s["start"]) + 1, xs(s["end"])), LANES["gt"] + BAR_H // 2],
                        fill=verb_color(s["action"]))
        for g in gts:
            marker(d, "dia", xs(g["t"]), LANES["real"], INK, 6)
        step = 60 if duration > 180 else 30
        for s in range(0, int(duration) + 1, step):
            d.line([xs(s), 196, xs(s), 202], fill=GRAPHITE, width=1)
            d.text((xs(s), 205), clock(s), font=ui(16), fill=GRAPHITE, anchor="mt")
        lx = 16 + LABEL_W
        for color, text in ((MAGENTA, "picking up a part"), (INK, "installing"), (GRAPHITE, "removing"), (FAINT, "other")):
            d.rectangle([lx, 244, lx + 22, 256], fill=color)
            d.text((lx + 30, 250), text, font=ui(15), fill=GRAPHITE, anchor="lm")
            lx += 48 + ui(15).getlength(text)
        for shape, color, text in (("tri", RED, "mistake alert"), ("dot", AMBER, "early warning"),
                                   ("ring", GRAPHITE, "cannot verify"), ("dia", INK, "real mistake")):
            marker(d, shape, lx + 8, 250, color, 7)
            d.text((lx + 22, 250), text, font=ui(15), fill=GRAPHITE, anchor="lm")
            lx += 40 + ui(15).getlength(text)
        if with_live:
            for e in res["events"]:
                d.rectangle([xs(e["start"]), LANES["live"] - BAR_H // 2, max(xs(e["start"]) + 1, xs(e["end"])), LANES["live"] + BAR_H // 2],
                            fill=verb_color(e["label"]))
            for a in sorted(res["alerts"], key=lambda a: {"status": 0, "warning": 1, "mistake": 2}[a["kind"]]):
                shape = {"mistake": "tri", "warning": "dot", "status": "ring"}[a["kind"]]
                color = RED if a["kind"] == "mistake" else AMBER if a["kind"] == "warning" else GRAPHITE
                marker(d, shape, xs(a["t"]), LANES["alerts"], color, 11)
        return img

    return np.asarray(base(True)).copy(), np.asarray(base(False)).copy()


def label_column(state: str, oracle: bool, final: bool = False, row_name: str | None = None) -> Image.Image:
    """The timeline's row names on a lighter column; the rows that are the system's own take the station's colour.
    ``final`` also names the action-end label's strip; ``row_name`` renames the actions row."""
    img = Image.new("RGB", (LABEL_W, H - VH), (243, 244, 241))
    d = ImageDraw.Draw(img)
    fill, text_color = LABEL_FILL[state], STATES[state][3]
    for name, y, pill in ((row_name or ("Annotated" if oracle else "Live Actions"), LANES["live"], True),
                          ("Ground Truth", LANES["gt"], False), ("Alerts", LANES["alerts"], True)):
        if pill:
            d.rounded_rectangle([-30, y - 24, LABEL_W - 12, y + 24], radius=24, fill=fill)
        d.text((LABEL_W - 28, y), name, font=bahn(28, "SemiBold" if pill else "Light"),
               fill=text_color if pill else INK, anchor="rm")
    if final:
        d.text((LABEL_W - 28, STRIP_Y), "Final label", font=ui(14), fill=INK, anchor="rm")
    return img


@lru_cache(maxsize=None)
def shadow_mask(width: int, height: int, strength: int, dark_on_right: bool) -> Image.Image:
    """How strongly black shows through: darkest at the edge casting the shadow, easing out to nothing."""
    ramp = (np.linspace(0, 1, width) ** 2 * strength).round().astype(np.uint8)
    return Image.fromarray(np.tile(ramp if dark_on_right else ramp[::-1], (height, 1)), "L")


@lru_cache(maxsize=None)
def _black(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (0, 0, 0))


def with_shadows(canvas: Image.Image) -> Image.Image:
    """The right panel casts a soft shadow to its left, and the label column one onto the timeline."""
    canvas.paste(_black(PANEL_SHADOW, H), (PX - PANEL_SHADOW, 0), shadow_mask(PANEL_SHADOW, H, 95, True))
    canvas.paste(_black(LABEL_SHADOW, H - VH), (LABEL_W, VH), shadow_mask(LABEL_SHADOW, H - VH, 80, False))
    return canvas


def timeline_at(full, future, t: float, duration: float, labels: Image.Image) -> Image.Image:
    xp = int(TL_LEFT + (TL_RIGHT - TL_LEFT) * min(t / duration, 1.0))
    arr = future.copy()
    arr[:, :xp] = full[:, :xp]
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.line([xp, 14, xp, 194], fill=INK, width=2)
    d.line([xp - 28, 14, xp + 28, 14], fill=INK, width=1)
    img.paste(labels, (0, 0))
    return img


# ----------------------------------------------------------------------------- the video
def draw_video(frame_bgr, event, state: str, hands, asd_row) -> Image.Image:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (VW, VH), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if asd_row is not None:
        sx, sy = VW / 1280, VH / 720
        x, y, w, h = asd_row["box"]
        box = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
        bad = asd_row["state"] == "error_state"
        corners(d, box, (*RED, 255) if bad else (255, 255, 255, 230))
        if bad:
            label = "State check: something looks mounted wrong"
        else:
            on = asd_row["state"][1:].count("1")
            label = f"State check: {on} part{'s' if on != 1 else ''} on the model"
        chip(d, (box[0], max(124, box[1] - 38)), label, ui(16), CARD if bad else INK, (*RED, 235) if bad else (*CARD, 215))
    boxes = []
    for hand in hands:
        pts = [(px * VW, py * VH) for px, py in hand]
        for a, b in HAND_EDGES:
            d.line([pts[a], pts[b]], fill=(255, 255, 255, 190), width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(255, 255, 255, 230))
        boxes.append((min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)))
    if event is not None and boxes and event["label"].split("_", 1)[0] in TAKE | INSTALL | {"put"}:
        box = [min(b[0] for b in boxes) - 14, min(b[1] for b in boxes) - 14, max(b[2] for b in boxes) + 14, max(b[3] for b in boxes) + 14]
        corners(d, box, (*MAGENTA, 255), 18, 3)
        noun = event["label"].partition("_")[2]
        chip(d, (max(4, box[0]), max(124, box[1] - 38)), f"In hand: {IN_HAND.get(noun, noun.replace('_', ' '))}",
             ui(16, "semibold"), CARD, (*MAGENTA, 235))
    text = describe(event["label"]) if event is not None else "Watching"
    font = bahn(44, "Regular")
    d.rounded_rectangle([-60, 30, 56 + font.getlength(text) + 36, 116], radius=43, fill=(*STATES[state][2], 200))
    d.text((56, 73), text, font=font, fill=(255, 255, 255, 255), anchor="lm")
    return Image.alpha_composite(base, layer).convert("RGB")


# ----------------------------------------------------------------------------- rendering
def render(rid: str, run: str, cfg: dict, out_path: Path, snapshots: tuple[float, ...] = (), tracking: bool = True,
           end_hold_s: float = 4.0, start: float = 0.0, stop: float | None = None, final_alert: dict | None = None,
           final_alert_s: float = 6.0, cad: bool = False, action_end: bool = False, final_actions: bool = False) -> None:
    """``final_actions``: a replay made after every action-end label is final, which shows those labels over each whole
    action, in the pill and the actions row. Live they only exist after each action ends, so the video must say so."""
    tasks = load_tasks(cfg["graph_path"])
    task = tasks[task_of(rid)]
    out_dir = cfg["paths"]["out"] / rid
    res = json.loads((out_dir / f"engine_{run}.json").read_text(encoding="utf-8"))
    cad_track, categories = [], ()
    if cad:                                    # the state the engine was given, drawn in the video's corner
        from assist.run_engine import ASD_FILES
        categories = tuple(cfg["asd_categories"])
        cad_track = state_track(read_jsonl(out_dir / ASD_FILES[res["state"]]), list(categories))

    def add_cad(image: Image.Image, at: float) -> None:
        if cad:
            card = inset(*shown_state(cad_track, at), categories)
            image.paste(card, (VW - card.width - 16, VH - card.height - 16), card)

    end_rows = with_times(read_jsonl(out_dir / "action_end.jsonl")) if action_end else []

    def add_final_chip(image: Image.Image, at: float) -> None:
        row = shown_final(end_rows, at) if action_end else None
        if row:
            draw_chip(image, row["label"])

    def timeline(at: float, labels_image: Image.Image) -> Image.Image:
        image = timeline_at(full, future, at, duration, labels_image)
        if action_end:
            draw_strip(image, end_rows, at, duration, TL_LEFT, TL_RIGHT)
        return image

    hands = read_jsonl(out_dir / "hands.jsonl") if tracking else []
    asd = read_jsonl(out_dir / "asd.jsonl") if tracking and res["state"] == "asd" else []
    gts = gt_mistakes(rid, task, cfg["components"], cfg["paths"]["labels"], cfg["fps"], res["duration_s"],
                      cfg["engine"]["order_grace_s"])
    duration = res["duration_s"]
    shown_events = final_events(read_jsonl(out_dir / "action_end.jsonl")) if final_actions else res["events"]
    full, future = build_timeline({**res, "events": shown_events}, gt_actions(rid, cfg["paths"]["labels"], cfg["fps"]), gts,
                                  duration)
    mistakes = [a for a in res["alerts"] if a["kind"] == "mistake"]
    shown = [a for a in res["alerts"] if a["kind"] != "status"]    # "cannot verify" stays a mark on the timeline
    title = f"{'Assembly' if task.name == 'assy' else 'Maintenance'} video"
    oracle = not res["actions"].startswith("pred")
    labels = {s: label_column(s, oracle, action_end, "Final labels" if final_actions else None) for s in STATES}
    cap = cv2.VideoCapture(str(cfg["paths"]["videos"] / f"{rid}.mp4"))
    wanted = {round(s * cfg["fps"]) for s in snapshots}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    if not snapshots:
        import imageio_ffmpeg
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                                 "rgb24", "-s", f"{W}x{H}", "-r", str(cfg["fps"]), "-i", "-", "-c:v", "libx264", "-preset",
                                 "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                                stdin=subprocess.PIPE)
    panels: dict = {}
    i = int(round(start * cfg["fps"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    k_asd, last_asd, canvas, last_frame = 0, None, None, None
    last = int(round((stop or duration) * cfg["fps"]))
    while i < last:
        ok, frame = cap.read()
        if not ok or (wanted and i > max(wanted)):
            break
        if wanted and i not in wanted:
            i += 1
            continue
        t = i / cfg["fps"]
        last_frame = frame
        while k_asd < len(asd) and asd[k_asd]["t"] <= t:
            last_asd = asd[k_asd]
            k_asd += 1
        status, verified = status_at(res["log"], t)
        flagged = {a["step"] for a in mistakes if a["t"] <= t and a["step"]}
        alert = current_alert(shown, t)
        nxt = [s for s in task.order if status.get(s) != "done" and all(status.get(p) == "done" for p in task.steps[s].requires)]
        card = card_text(alert, task, nxt, False, 0)
        key = (tuple(sorted(status.items())), tuple(sorted(verified)), tuple(sorted(flagged)), tuple(card.values()))
        if key not in panels:
            panels = {key: draw_panel(task, title, status, verified, flagged, card)}
        canvas = Image.new("RGB", (W, H), PAPER)
        row = last_asd if last_asd and t - last_asd["t"] <= 1.5 else None
        canvas.paste(draw_video(frame, active_event(shown_events, t), card["state"], hands[i]["hands"] if i < len(hands) else [], row), (0, 0))
        add_cad(canvas, t)
        add_final_chip(canvas, t)
        canvas.paste(panels[key], (PX, 0))
        canvas.paste(timeline(t, labels[card["state"]]), (0, VH))
        with_shadows(canvas)
        if proc:
            proc.stdin.write(canvas.tobytes())
        else:
            canvas.save(out_path.with_name(f"{out_path.stem}_{t:.0f}s.png"))
        i += 1
        if i % 500 == 0:
            print(json.dumps({"video": rid, "frame": i, "t": round(t, 1)}), flush=True)
    cap.release()
    if proc and canvas is not None and stop is None:      # a clip cut before the real end has no end card
        status, verified = status_at(res["log"], duration + 1)
        if final_alert:                                   # an alert raised as the task ends would otherwise never show
            card = card_text(final_alert, task, [], False, 0)
            held = canvas.copy()
            held.paste(draw_video(last_frame, None, card["state"], [], None), (0, 0))
            add_cad(held, duration)
            add_final_chip(held, duration)
            held.paste(draw_panel(task, title, status, verified, {a["step"] for a in mistakes if a["step"]}, card), (PX, 0))
            held.paste(timeline(duration, labels[card["state"]]), (0, VH))
            with_shadows(held)
            for _ in range(int(final_alert_s * cfg["fps"])):
                proc.stdin.write(held.tobytes())
        card = card_text(None, task, [], True, len(mistakes))
        end = canvas.copy()
        end.paste(draw_video(last_frame, None, card["state"], [], None), (0, 0))     # the pill takes the end card's colour
        add_cad(end, duration)
        add_final_chip(end, duration)
        end.paste(draw_panel(task, title, status, verified, {a["step"] for a in mistakes if a["step"]}, card), (PX, 0))
        end.paste(timeline(duration, labels[card["state"]]), (0, VH))
        with_shadows(end)
        for _ in range(int(end_hold_s * cfg["fps"])):
            proc.stdin.write(end.tobytes())
    if proc:
        proc.stdin.close()
        proc.wait()
        print(json.dumps({"written": str(out_path.relative_to(ROOT))}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--run", default="pred_vjepap_asd", help="engine_<run>.json to show (e.g. gt_gt).")
    parser.add_argument("--snapshot", type=float, nargs="*", default=(), help="Only save PNG stills at these times.")
    parser.add_argument("--no-tracking", dest="tracking", action="store_false", help="Leave out the hands and the state box.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    out = args.out or cfg["paths"]["out"] / "overlays/cards" / f"{args.video}_{args.run}.mp4"
    render(args.video, args.run, cfg, out, tuple(args.snapshot), args.tracking)


if __name__ == "__main__":
    main()
