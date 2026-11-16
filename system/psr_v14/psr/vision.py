"""One-shot visual verification of assembly states with DINOv2 (no ASD detector, no training).

Pieces
  * Localizer: patch-level nearest-prototype classifier (assembly vs. everything else).  Prototypes
    are DINOv2 patch tokens from a handful of TRAIN frames (inside / outside the train OD box).
  * View quality gate: blur + hand-skin coverage of the ROI.  Bad views are skipped ("wait").
  * Reference bank, one image per assembly state:
       cad      -> the CAD render of each of the 22 states
       exemplar -> one real TRAIN crop per state (CAD render where train has no example)
  * Sub-part comparison: for a hypothesis "before -> after", we compare the ROI's patch tokens
    only against the reference patches that differ between the before/after references
    (the region where the newly assembled part lives), plus a global term.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import CAD_DIR, CAD_STATE_CODES, SPLIT_DIRS, code, apply_steps, hamming

CACHE = Path(os.environ.get("PSR_CACHE", Path(__file__).resolve().parent.parent / "cache"))
MODEL_ID = "facebook/dinov2-small"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DinoEncoder:
    def __init__(self, device=None):
        from transformers import AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval().half()
        self.patch = 14

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, size: tuple[int, int]):
        """rgb HxWx3 uint8 -> (cls [D], patches [h, w, D]) both L2-normalised, float32 numpy."""
        W, H = size
        x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device).half()
        out = self.model(pixel_values=t).last_hidden_state[0].float()
        cls, pt = out[0], out[1:]
        pt = F.normalize(pt, dim=-1).reshape(H // self.patch, W // self.patch, -1)
        return F.normalize(cls, dim=0).cpu().numpy(), pt.cpu().numpy()


def read_rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _train_od_boxes(max_per_state=40, seed=0):
    """(frame_path, state_code, bbox) samples from TRAIN OD labels."""
    rng = random.Random(seed)
    by_state = {}
    root = SPLIT_DIRS["train"]
    for v in sorted(os.listdir(root)):
        d = json.loads((root / v / "OD_labels.json").read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i["file_name"] for i in d["images"]}
        for a in d["annotations"]:
            name = cats[a["category_id"]]
            if name in ("background", "error_state"):
                continue
            by_state.setdefault(name, []).append((root / v / "rgb" / imgs[a["image_id"]], name, a["bbox"]))
    out = {}
    for s, lst in by_state.items():
        rng.shuffle(lst)
        out[s] = lst[:max_per_state]
    return out


def crop(rgb, box, pad=0.12):
    x, y, w, h = box
    H, W = rgb.shape[:2]
    x0 = int(max(0, x - pad * w)); y0 = int(max(0, y - pad * h))
    x1 = int(min(W, x + w * (1 + pad))); y1 = int(min(H, y + h * (1 + pad)))
    return rgb[y0:y1, x0:x1]


def cad_preview(state_code: str) -> np.ndarray:
    i = CAD_STATE_CODES.index(state_code) + 1
    img = cv2.imread(str(CAD_DIR / "previews" / f"state{i}.png"), cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:  # composite on a table-like light background
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 225 * (1 - a)).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ys, xs = np.where(rgb.min(axis=2) < 215)
    if len(xs):
        rgb = rgb[max(0, ys.min() - 10):ys.max() + 10, max(0, xs.min() - 10):xs.max() + 10]
    return rgb


class VisionVerifier:
    ROI_SIZE = (224, 224)
    FRAME_SIZE = (448, 252)

    def __init__(self, bank: str = "exemplar", device=None, knn: int = 10, min_quality: float = 0.25):
        self.enc = DinoEncoder(device)
        self.bank_kind = bank
        self.knn = knn
        self.min_quality = min_quality
        CACHE.mkdir(exist_ok=True)
        self._build_localizer()
        self._build_bank()
        self.protos = None
        if bank == "proto":
            self._load_protos()
        self.prev_box = None

    def _load_protos(self, exclude_subjects=()):
        """Per-state TRAIN views (build_prototypes.py).  exclude_subjects is for leave-one-out tuning."""
        z = np.load(CACHE / "state_protos_raw.npz")
        keep = ~np.isin(z["subj"], list(exclude_subjects))
        emb = torch.from_numpy(z["emb"][keep].astype(np.float32)).to(self.enc.device)
        codes = z["code"][keep]
        self.protos = {c: emb[torch.from_numpy(codes == c).to(self.enc.device)] for c in sorted(set(codes))}

    def proto_sim(self, cls_vec, state_code):
        """kNN similarity of the ROI embedding to TRAIN views of a state (nearest available state)."""
        if state_code not in self.protos:
            c = min(self.protos, key=lambda k: hamming(k, state_code))
            pen = hamming(c, state_code)
        else:
            c, pen = state_code, 0
        q = torch.from_numpy(np.asarray(cls_vec, np.float32)).to(self.enc.device)
        s = self.protos[c] @ q
        k = min(self.knn, len(s))
        return float(s.topk(k).values.mean()), pen

    # ------------------------------------------------------------------ build (train only)
    def _build_localizer(self):
        f = CACHE / "localizer_protos.npz"
        if f.exists():
            z = np.load(f); self.fg, self.bg = z["fg"], z["bg"]; return
        samples = _train_od_boxes(max_per_state=6, seed=1)
        fg, bg = [], []
        gw, gh = self.FRAME_SIZE[0] // 14, self.FRAME_SIZE[1] // 14
        for lst in samples.values():
            for path, _, (x, y, w, h) in lst:
                rgb = read_rgb(path)
                _, pt = self.enc.encode(rgb, self.FRAME_SIZE)
                sx, sy = 1280 / gw, 720 / gh
                yy, xx = np.mgrid[0:gh, 0:gw]
                cx, cy = (xx + 0.5) * sx, (yy + 0.5) * sy
                inside = (cx > x) & (cx < x + w) & (cy > y) & (cy < y + h)
                core = (cx > x + 0.2 * w) & (cx < x + 0.8 * w) & (cy > y + 0.2 * h) & (cy < y + 0.8 * h)
                fg.append(pt[core]); bg.append(pt[~inside])
        fg = np.concatenate(fg); bg = np.concatenate(bg)
        rng = np.random.default_rng(0)
        self.fg = fg[rng.choice(len(fg), min(4000, len(fg)), replace=False)].astype(np.float16)
        self.bg = bg[rng.choice(len(bg), min(8000, len(bg)), replace=False)].astype(np.float16)
        np.savez(f, fg=self.fg, bg=self.bg)

    def _build_bank(self):
        f = CACHE / f"bank_{self.bank_kind}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            self.bank_codes = list(z["codes"]); self.bank_cls = z["cls"]; self.bank_pt = z["pt"]
            self.bank_src = list(z["src"]); return
        ex = _train_od_boxes(max_per_state=12, seed=2) if self.bank_kind == "exemplar" else {}
        codes, clss, pts, srcs = [], [], [], []
        for c in CAD_STATE_CODES:
            if c in ex:
                # one-shot: keep the single train crop closest to the mean of a few candidates
                embs = [self.enc.encode(crop(read_rgb(p), b), self.ROI_SIZE) for p, _, b in ex[c]]
                m = np.mean([e[0] for e in embs], 0)
                i = int(np.argmax([e[0] @ m for e in embs]))
                cls, pt = embs[i]; src = f"train:{ex[c][i][0].parent.parent.name}/{ex[c][i][0].name}"
            else:
                cls, pt = self.enc.encode(cad_preview(c), self.ROI_SIZE); src = "cad"
            codes.append(c); clss.append(cls); pts.append(pt); srcs.append(src)
        self.bank_codes, self.bank_cls, self.bank_pt, self.bank_src = codes, np.stack(clss), np.stack(pts), srcs
        np.savez(f, codes=codes, cls=self.bank_cls, pt=self.bank_pt, src=srcs)

    # ------------------------------------------------------------------ runtime
    def localize(self, pt_frame):
        gh, gw, D = pt_frame.shape
        p = torch.from_numpy(pt_frame.reshape(-1, D)).half().to(self.enc.device)
        fg = torch.from_numpy(self.fg).to(self.enc.device)
        bg = torch.from_numpy(self.bg).to(self.enc.device)
        s_fg = (p @ fg.T).topk(5, dim=1).values.mean(1)
        s_bg = (p @ bg.T).topk(5, dim=1).values.mean(1)
        score = (s_fg - s_bg).float().reshape(gh, gw).cpu().numpy()
        mask = (score > 0).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None, score
        best, bscore = None, -1e9
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            s = score[lab == i].sum()
            if self.prev_box is not None:   # temporal prior: prefer the blob near the previous ROI
                px, py, pw, ph = self.prev_box
                cx, cy = (x + w / 2) * 1280 / gw, (y + h / 2) * 720 / gh
                d = np.hypot(cx - (px + pw / 2), cy - (py + ph / 2)) / 640
                s *= np.exp(-d)
            if s > bscore:
                best, bscore = (x, y, w, h), s
        x, y, w, h = best
        sx, sy = 1280 / gw, 720 / gh
        box = [x * sx, y * sy, w * sx, h * sy]
        if box[2] * box[3] < 60 * 60:
            return None, score
        self.prev_box = box
        return box, score

    @staticmethod
    def view_quality(roi_rgb):
        if roi_rgb.size == 0:
            return 0.0, {}
        g = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        skin = ((hsv[..., 0] < 25) & (hsv[..., 1] > 40) & (hsv[..., 1] < 170) & (hsv[..., 2] > 60)).mean()
        q = float(np.clip(sharp / 150, 0, 1) * np.clip(1 - 2.0 * skin, 0, 1))
        return q, {"sharpness": round(float(sharp), 1), "hand_cover": round(float(skin), 3)}

    def frame_feature(self, frame, keep_rgb=False):
        rgb = read_rgb(frame) if not isinstance(frame, np.ndarray) else frame
        _, pt = self.enc.encode(rgb, self.FRAME_SIZE)
        box, heat = self.localize(pt)
        feat = {"box": box, "heat": heat, "quality": 0.0, "q_info": {}}
        if box is None:
            return feat
        roi = crop(rgb, box, pad=0.1)
        q, info = self.view_quality(roi)
        cls, rpt = self.enc.encode(roi, self.ROI_SIZE)
        sims = self.bank_cls @ cls
        feat.update(quality=q, q_info=info, cls=cls, pt=rpt, state_sims=dict(zip(self.bank_codes, sims.tolist())))
        if keep_rgb:
            feat["roi"] = roi
        return feat

    def _ref_index(self, state_code):
        """Bank entry for a (possibly non-CAD) state: exact match, else nearest by Hamming."""
        d = [hamming(state_code, c) for c in self.bank_codes]
        i = int(np.argmin(d))
        return i, d[i]

    def subpart_similarity(self, feat, i_before, i_after):
        """Compare the ROI only on the reference patches that change between before/after refs."""
        ra, rb = self.bank_pt[i_after], self.bank_pt[i_before]
        diff = 1 - (ra * rb).sum(-1)                     # where the two references disagree
        sel = diff > np.quantile(diff, 0.8)
        q = feat["pt"].reshape(-1, feat["pt"].shape[-1])
        sa = (ra[sel] @ q.T).max(1).mean()
        sb = (rb[sel] @ q.T).max(1).mean()
        return float(sa - sb)

    def group_evidence(self, feat, state, groups, min_quality=None):
        """{group: evidence} (positive favours the group) or None when the view is unusable."""
        min_quality = self.min_quality if min_quality is None else min_quality
        if feat is None or feat.get("cls") is None or feat["quality"] < min_quality:
            return None
        before = code(state)
        out = {}
        if self.protos is not None:
            sb, pb = self.proto_sim(feat["cls"], before)
            for g in groups:
                if g == ():
                    out[g] = 0.0
                    continue
                sa, pa = self.proto_sim(feat["cls"], code(apply_steps(state, g)))
                out[g] = (sa - sb) * 20 / (1 + pa + pb)
            return out
        ib, db = self._ref_index(before)
        for g in groups:
            if g == ():
                out[g] = 0.0
                continue
            after = code(apply_steps(state, g))
            ia, da = self._ref_index(after)
            if ia == ib:
                out[g] = 0.0
                continue
            glob = float(self.bank_cls[ia] @ feat["cls"] - self.bank_cls[ib] @ feat["cls"])
            sub = self.subpart_similarity(feat, ib, ia)
            out[g] = (glob + sub) * 10 / (1 + da)      # scaled; trust exact references more
        return out


class PartVerifier(VisionVerifier):
    """Runtime visual check = localizer/crop features of VisionVerifier + per-component classifiers (part_vision)."""

    def __init__(self, parts_path, device=None, min_quality: float = 0.25):
        super().__init__(bank="exemplar", device=device, min_quality=min_quality)
        from .part_vision import PartVision
        self.parts = PartVision(path=parts_path)
        self.parts.min_q = min_quality

    def group_evidence(self, feat, state, groups, min_quality=None):
        return self.parts.group_evidence(feat, state, groups)


class CadVerifier(VisionVerifier):
    """Runtime visual check = localizer crop + render-and-compare against the 3D CAD (psr.cad_match)."""

    def __init__(self, device=None, min_quality: float = 0.3, scale: float = 30.0):
        super().__init__(bank="exemplar", device=device, min_quality=min_quality)
        from .cad_match import CadMatcher
        self.cad = CadMatcher(self.enc)
        self.min_q = min_quality
        self.scale = scale
        self.last = None

    def frame_feature(self, frame, keep_rgb=True):
        return super().frame_feature(frame, keep_rgb=True)

    def cad_score(self, feat, before, after):
        r = self.cad.compare(feat["roi"], feat["cls"], before, after)
        self.last = r
        return None if r["score"] is None else r["score"] * self.scale


class LiveVerifier(VisionVerifier):
    """Runtime: localizer crop -> learned per-component verifier (frozen/v5/verifier.pt)."""

    def __init__(self, weights, min_quality: float = 0.0, device=None):
        super().__init__(bank="exemplar", device=device, min_quality=min_quality)
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from train_verifier import Verifier, to_tensor
        self._to_tensor = to_tensor
        self.ver = Verifier().to(self.enc.device)
        self.ver.load_state_dict(torch.load(weights, map_location=self.enc.device))
        self.ver.eval()
        self.min_q = min_quality
        self.parts = self

    def frame_feature(self, frame, keep_rgb=False):
        rgb = read_rgb(frame) if not isinstance(frame, np.ndarray) else frame
        _, pt = self.enc.encode(rgb, self.FRAME_SIZE)
        box, heat = self.localize(pt)
        feat = {"box": box, "heat": heat, "quality": 0.0, "q_info": {}, "cls": None}
        if box is None:
            return feat
        roi = cv2.resize(crop(rgb, box, pad=0.1), (160, 160), interpolation=cv2.INTER_AREA)
        q, info = self.view_quality(roi)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(self.ver(self._to_tensor([roi]).to(self.enc.device)).float())[0].cpu().numpy()
        feat.update(quality=q, q=q, q_info=info, cls=p)
        return feat

    def part_probs(self, p):
        return {k + 1: float(p[k]) for k in range(10)}

    def group_evidence(self, feat, state, groups, min_quality=None):
        if feat is None or feat.get("cls") is None or feat["quality"] < self.min_q:
            return None
        p = self.part_probs(feat["cls"]); out = {}
        for g in groups:
            if g == ():
                out[g] = 0.0; continue
            after = apply_steps(state, g); ev = 0.0
            for k in range(1, 11):
                if after[k] != state[k]:
                    pk = min(max(p[k], 1e-3), 1 - 1e-3); lo = float(np.log(pk / (1 - pk)))
                    ev += lo if after[k] == 1 else -lo
            out[g] = ev
        return out
