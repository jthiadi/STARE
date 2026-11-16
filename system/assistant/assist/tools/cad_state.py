"""A small CAD picture of the assembly state the system has reported so far, for the overlay videos.

It shows what the state input said at that moment, not what is really on the table. The dataset models correct states only,
so "error_state" has no picture: the last correct state stays up with a red label. Each file ``state<n>.fbx`` is entry ``n`` of
``asd_categories``. The files are posed differently, so every model is turned to match state 13 by its base block, which every
state contains; a part then appears in place instead of the whole car turning.

    # part_geometries.zip (17 MB), from https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224
    # unzipped so that output/assist/cad/part_geometries/state1.fbx exists
"""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from assist.common import ROOT
from assist.tools.render_overlay import CARD, INK, RED, ui
from scripts.mistakes.fbx_mesh import Piece, load

MODELS = ROOT / "output/assist/cad/part_geometries"
ERROR = "error_state"
REFERENCE = 13                          # the finished car with the long rear chassis
SHORT_CAR = 22                          # the finished car with the short one
BASE_SHAPE = np.array([1.0, 2.0, 4.0])  # the base block's sides, shortest first
SIGNS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
YAW, PITCH = 200.0, 42.0                # degrees: turned around the vertical axis, then seen from above
LIGHT = np.array([0.25, 0.45, 0.86])
SUPERSAMPLE = 3
INSET_W, INSET_H = 340, 270
PANEL_ALPHA = 175                       # of 255: how solid the panel behind the model is
PANEL_FILL = (34, 38, 44)               # dark gray, so the light title and the model read over any video


# ----------------------------------------------------------------------------- which state to show
def state_name(state) -> str | None:
    """A reading's state as a code such as ``10000000000``; readings come as a string or a list."""
    if state is None:
        return None
    return state if isinstance(state, str) else "".join(str(v) for v in state)


def model_number(state: str, categories: list[str]) -> int | None:
    """Which ``state<n>.fbx`` shows this state, or None if the dataset has no model of it."""
    if state in (ERROR, "background") or state not in categories:
        return None
    return categories.index(state)


def state_track(rows: list[dict], categories: list[str]) -> list[tuple[float, str | None, bool]]:
    """``(time, state to draw, wrong?)`` at every change, from the rows of a state file."""
    track: list[tuple[float, str | None, bool]] = []
    last = None
    for row in rows:
        name = state_name(row.get("state"))
        if name is None:
            continue
        if name == ERROR:
            entry = (row["t"], last, True)
        else:
            last = name if model_number(name, categories) is not None else None
            entry = (row["t"], last, False)
        if not track or track[-1][1:] != entry[1:]:
            track.append(entry)
    return track


def shown_state(track: list[tuple[float, str | None, bool]], t: float) -> tuple[str | None, bool]:
    """What was known at time t: never a later reading."""
    i = bisect_right([row[0] for row in track], t)
    return (None, False) if i == 0 else track[i - 1][1:]


# ----------------------------------------------------------------------------- turning the models to one pose
def _axes(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A piece's main axes (columns, longest last, right-handed), its box centre, and its box's sides, shortest first."""
    _, vectors = np.linalg.eigh(np.cov((verts - verts.mean(0)).T))
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] = -vectors[:, 0]
    along = verts @ vectors
    return vectors, vectors @ ((along.min(0) + along.max(0)) / 2), along.max(0) - along.min(0)


def _base_candidates(pieces: list[Piece]) -> list[Piece]:
    """White pieces shaped like the base block."""
    white = [p for p in pieces if p.material.startswith("White") and len(p.verts) >= 4]
    close = [p for p in white if np.abs(_axes(p.verts)[2] / _axes(p.verts)[2][0] - BASE_SHAPE).max() < 0.25]
    return close or white


def _distance(points: np.ndarray, targets: np.ndarray) -> float:
    """Mean distance from each point to its nearest target."""
    points = points[:: max(1, len(points) // 800)].astype(np.float32)
    targets = targets[:: max(1, len(targets) // 4000)].astype(np.float32)
    return float(np.mean([np.sqrt(((targets - p) ** 2).sum(1).min()) for p in points]))


def align(pieces: list[Piece], reference: list[Piece]) -> list[Piece]:
    """The model turned and moved so that its base block sits on the reference's, the way round that fits the rest best."""
    target = np.concatenate([p.verts for p in reference])
    ref_axes, ref_centre, _ = _axes(_base_candidates(reference)[0].verts)
    verts = np.concatenate([p.verts for p in pieces])
    best, best_turn, best_move = np.inf, None, None
    for base in _base_candidates(pieces):
        axes, centre, _ = _axes(base.verts)
        for signs in SIGNS:
            turn = ref_axes @ np.diag(signs) @ axes.T
            move = ref_centre - centre @ turn.T
            score = _distance(verts @ turn.T + move, target)
            if score < best:
                best, best_turn, best_move = score, turn, move
    return [Piece(p.verts @ best_turn.T + best_move, p.tris, p.color, p.material) for p in pieces]


# ----------------------------------------------------------------------------- drawing
def _camera() -> np.ndarray:
    yaw, pitch = np.radians(YAW), np.radians(PITCH)
    around = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    above = np.array([[1, 0, 0], [0, np.sin(pitch), np.cos(pitch)], [0, -np.cos(pitch), np.sin(pitch)]])
    return above @ around


def draw_model(pieces: list[Piece], size: tuple[int, int], centre: np.ndarray, span: float,
               background: tuple[int, int, int] | None) -> Image.Image:
    """Flat-shaded picture, far faces first; ``centre`` and ``span`` stay fixed so that every state lines up.
    ``background=None`` gives an RGBA picture that is clear wherever there is no model."""
    camera = _camera()
    triangles, colours = [], []
    for piece in pieces:
        corners = ((piece.verts - centre) @ camera.T)[piece.tris]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
        shade = 0.5 + 0.5 * np.abs(normal @ LIGHT)
        base = np.maximum(np.array(piece.color) ** (1 / 2.2), 0.13)    # the file's colours are linear; black keeps some shape
        colours.append(base * shade[:, None])
        triangles.append(corners)
    triangles, colours = np.concatenate(triangles), np.concatenate(colours)
    w, h = size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE
    scale = w / span
    image = Image.new("RGBA", (w, h), (0, 0, 0, 0)) if background is None else Image.new("RGB", (w, h), background)
    solid = () if background is not None else (255,)
    pen = ImageDraw.Draw(image)
    for i in np.argsort(triangles[:, :, 2].mean(1)):
        pen.polygon([(w / 2 + x * scale, h / 2 - y * scale) for x, y, _ in triangles[i]],
                    fill=tuple(int(min(c, 1) * 255) for c in colours[i]) + solid)
    return image.resize(size, Image.LANCZOS)


@lru_cache(maxsize=None)
def _reference() -> tuple[list[Piece], np.ndarray, float]:
    reference = load(MODELS / f"state{REFERENCE}.fbx")
    verts = np.concatenate([p.verts for p in reference + align(load(MODELS / f"state{SHORT_CAR}.fbx"), reference)])
    camera = _camera()
    seen = (verts - (verts.min(0) + verts.max(0)) / 2) @ camera.T
    wide, tall = (seen.max(0) - seen.min(0))[:2]
    aspect = (INSET_W - 16) / (INSET_H - 44)                # the width the picture must span so that the car also fits its height
    return reference, (verts.min(0) + verts.max(0)) / 2, float(max(wide, tall * aspect) * 1.04)


@lru_cache(maxsize=None)
def state_picture(state: str, categories: tuple[str, ...], size: tuple[int, int]) -> Image.Image:
    reference, centre, span = _reference()
    number = model_number(state, list(categories))
    pieces = reference if number == REFERENCE else align(load(MODELS / f"state{number}.fbx"), reference)
    return draw_model(pieces, size, centre, span, None)


@lru_cache(maxsize=None)
def inset(state: str | None, wrong: bool, categories: tuple[str, ...]) -> Image.Image:
    """The card for the corner of the video: a see-through dark gray panel (``PANEL_FILL``, ``PANEL_ALPHA``) under a solid model, title and label.
    It is RGBA, so it is pasted with itself as the mask."""
    card = Image.new("RGBA", (INSET_W, INSET_H), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle([0, 0, INSET_W - 1, INSET_H - 1], radius=14, fill=(*PANEL_FILL, PANEL_ALPHA))
    if state is not None:
        card.alpha_composite(state_picture(state, categories, (INSET_W - 16, INSET_H - 44)), (8, 38))
    words = Image.new("RGBA", (INSET_W, INSET_H), (0, 0, 0, 0))
    pen = ImageDraw.Draw(words)
    if state is None:
        pen.text((INSET_W / 2, INSET_H / 2), "No model for this state", font=ui(18), fill=(*CARD, 255), anchor="mm")
    pen.text((16, 20), "Recognised state", font=ui(20, "semibold"), fill=(*CARD, 255), anchor="lm")
    if wrong:
        pen.rounded_rectangle([10, INSET_H - 44, INSET_W - 10, INSET_H - 10], radius=10, fill=(*RED, 255))
        pen.text((INSET_W / 2, INSET_H - 27), "Wrong state seen", font=ui(20, "semibold"), fill=(*CARD, 255), anchor="mm")
    card.alpha_composite(words)
    return card
