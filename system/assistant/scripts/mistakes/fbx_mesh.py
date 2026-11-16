"""Read IndustReal's state models (binary FBX 7400, ``part_geometries.zip``) with no 3D library.

Each state file holds one mesh that joins every part. ``load`` returns it as numpy arrays and ``pieces`` splits it into
its connected pieces, each with its colour, so a single part can be moved.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCALAR = {"Y": "<h", "C": "<B", "I": "<i", "L": "<q", "F": "<f", "D": "<d"}
_ARRAY = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "B"}


def _node(b: bytes, p: int) -> tuple[tuple | None, int]:
    end, count, _, name_len = struct.unpack("<IIIB", b[p:p + 13])
    p += 13
    if end == 0:
        return None, p
    name = b[p:p + name_len].decode()
    p += name_len
    props: list = []
    for _ in range(count):
        kind = chr(b[p])
        p += 1
        if kind in _SCALAR:
            props.append(struct.unpack_from(_SCALAR[kind], b, p)[0])
            p += struct.calcsize(_SCALAR[kind])
        elif kind in _ARRAY:
            length, encoded, size = struct.unpack("<III", b[p:p + 12])
            p += 12
            raw = zlib.decompress(b[p:p + size]) if encoded else b[p:p + size]
            p += size
            props.append(np.frombuffer(raw, dtype="<" + _ARRAY[kind], count=length))
        elif kind in "SR":
            length = struct.unpack("<I", b[p:p + 4])[0]
            p += 4
            props.append(b[p:p + length])
            p += length
        else:
            raise ValueError(f"unknown FBX property type {kind!r}")
    kids = []
    while p < end:
        kid, p = _node(b, p)
        if kid is None:
            break
        kids.append(kid)
    return (name, props, kids), end


def _child(node: tuple, name: str) -> tuple | None:
    return next((k for k in node[2] if k[0] == name), None)


def _p70(node: tuple) -> dict[bytes, list]:
    block = _child(node, "Properties70")
    return {k[1][0]: k[1][4:] for k in block[2]} if block else {}


def triangulate(polygon_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FBX polygon indices (the last corner of a polygon is stored as ~i) -> (triangles, polygon of each triangle)."""
    tris, owner, poly, current = [], [], 0, []
    for i in polygon_index.tolist():
        current.append(i if i >= 0 else ~i)
        if i < 0:
            for k in range(1, len(current) - 1):
                tris.append((current[0], current[k], current[k + 1]))
                owner.append(poly)
            current = []
            poly += 1
    return np.array(tris, np.int64).reshape(-1, 3), np.array(owner, np.int64)


def components(n_vertices: int, tris: np.ndarray) -> np.ndarray:
    """Connected-piece label of every vertex."""
    parent = np.arange(n_vertices)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in tris.tolist():
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    roots = np.array([find(i) for i in range(n_vertices)])
    return np.unique(roots, return_inverse=True)[1]


@dataclass
class Piece:
    verts: np.ndarray      # (n, 3)
    tris: np.ndarray       # (m, 3), indices into verts
    color: tuple[float, float, float]
    material: str

    @property
    def center(self) -> np.ndarray:
        return (self.verts.min(0) + self.verts.max(0)) / 2

    @property
    def size(self) -> np.ndarray:
        return self.verts.max(0) - self.verts.min(0)


def load(path: str | Path) -> list[Piece]:
    b = Path(path).read_bytes()
    if not b.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path} is not a binary FBX file")
    p, top = 27, []
    while p < len(b):
        node, p = _node(b, p)
        if node is None:
            break
        top.append(node)
    objects = next(t for t in top if t[0] == "Objects")
    connections = next(t for t in top if t[0] == "Connections")
    geometry = next(k for k in objects[2] if k[0] == "Geometry")
    model = next(k for k in objects[2] if k[0] == "Model" and k[1][2] == b"Mesh")
    materials = {k[1][0]: k for k in objects[2] if k[0] == "Material"}
    # material slots in order: the materials connected to the mesh's model
    slots = [materials[c[1][1]] for c in connections[2] if c[1][1] in materials and c[1][2] == model[1][0]]
    verts = _child(geometry, "Vertices")[1][0].reshape(-1, 3).astype(np.float64)
    tris, owner = triangulate(_child(geometry, "PolygonVertexIndex")[1][0])
    per_poly = _child(_child(geometry, "LayerElementMaterial"), "Materials")[1][0]
    face_material = per_poly[owner] if len(per_poly) > 1 else np.zeros(len(tris), np.int64)
    scale = _p70(model).get(b"Lcl Scaling", [1.0, 1.0, 1.0])
    verts = verts * np.array(scale[:3], np.float64)
    labels = components(len(verts), tris)
    out = []
    for k in range(labels.max() + 1):
        keep = np.flatnonzero(labels == k)
        remap = -np.ones(len(verts), np.int64)
        remap[keep] = np.arange(len(keep))
        mask = labels[tris[:, 0]] == k
        if not mask.any():
            continue
        slot = slots[int(np.bincount(face_material[mask]).argmax())]
        settings = _p70(slot)
        rgb = settings.get(b"DiffuseColor", [0.5, 0.5, 0.5])[:3]
        name = slot[1][1].split(b"\x00")[0].decode()
        out.append(Piece(verts[keep], remap[tris[mask]], tuple(float(x) for x in rgb), name))
    return out
