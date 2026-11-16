"""Per-component visual verifier: assembly crop -> P(component k attached on the main assembly), k = 1..10.

Training labels: assembly state from PSR_labels.csv of the TRAINING videos of the fold only; frames within
+-GUARD frames of a state change are skipped (ambiguous).  No ASD detector, boxes or ASD predictions are used.

  python train_verifier.py fold <i>     -> trains on train+val subjects outside fold i (4 folds by subject),
                                            writes out-of-fold probabilities for fold i videos
  python train_verifier.py final        -> trains on all train+val, saves cache/verifier_final.pt
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psr.core import INITIAL_STATE, apply_steps, bits, gt_groups, list_videos, load_psr_gt, procedure_type, video_dir
import os
THREE = os.environ.get("VERIF3") == "1"
BACK = os.environ.get("VBACK", "small")          # small | base
TAG = ("verif3" if THREE else "verif") + ("" if BACK == "small" else f"_{BACK}")
HID = {"small": 384, "base": 768}[BACK]
RES = int(os.environ.get("VRES", "0"))          # >0: high-res JPEG crops from cache/crops_hr at this size
if RES:
    TAG += f"_r{RES}"
from psr.vision import CACHE

GUARD, SIZE, EPOCHS, BS = 20, 224, 4, 48
if os.environ.get("VRES", "0") != "0":
    SIZE, BS = int(os.environ["VRES"]), 24
EPOCHS = int(os.environ.get("VEPOCHS", EPOCHS))
KSHOT = int(os.environ.get("VKSHOT", 0))          # >0: only K labelled crops per assembly state (few-shot)
KSTEPS = int(os.environ.get("VKSTEPS", 1500))     # few-shot: fixed number of optimiser steps
KPICK = os.environ.get("VKPICK", "view")          # view: best view quality anywhere; early: just after the state began
if KSHOT:
    TAG += f"_k{KSHOT}" + ("" if KPICK == "view" else f"_{KPICK}")
VCAD = os.environ.get("VCAD") == "1"              # add CAD renders of every state as synthetic training images
if VCAD:
    TAG += "_cad"
VZERO = os.environ.get("VZERO") == "1"            # zero-shot: CAD renders only, no labelled real crop
VCADRAND = os.environ.get("VCADRAND") == "1"      # CAD domain randomisation (real backgrounds, occluders, colour)
VSELF = os.environ.get("VSELF", "0") != "0"       # self-training on UNLABELLED training-fold crops (pseudo-labels)
SELF_MODE = int(os.environ.get("VSELF", "0") or 0)  # 1 checker median, 2 state filter, 3 filter + checker agrees
SELF_N, SELF_STEPS = int(os.environ.get("VSELF_N", 8000)), int(os.environ.get("VSELF_STEPS", 3000))
SELF_ROUNDS = int(os.environ.get("VSELF_ROUNDS", 1))
TAG += ("_zero" if VZERO else "") + ("_rand" if VCADRAND else "") + ("_self" if VSELF else "") +     (f"{SELF_MODE}" if SELF_MODE > 1 else "") + (f"r{SELF_ROUNDS}" if VSELF and SELF_ROUNDS > 1 else "")
UNFREEZE = int(os.environ.get("VUNFREEZE", 4))
if os.environ.get("VEPOCHS") or os.environ.get("VUNFREEZE"):
    TAG += f"_e{EPOCHS}u{UNFREEZE}"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ITEMS = [("train", v) for v in list_videos("train")] + [("val", v) for v in list_videos("val")]
SUBS = sorted({v[:2] for _, v in ITEMS})
FOLDS = [SUBS[i::4] for i in range(4)]


class Verifier(nn.Module):
    def __init__(self, unfreeze=None):
        unfreeze = UNFREEZE if unfreeze is None else unfreeze
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(f"facebook/dinov2-{BACK}")
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[-unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True
        self.head = nn.Sequential(nn.Linear(2 * HID, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 30 if THREE else 10))

    def forward(self, x):                       # x: B,3,H,W in [0,1]
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        h = self.backbone(pixel_values=x).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)
        return self.head(f)                      # logits for components 1..10


def raw_series(split, v, ts):
    """per-frame state with -1 (incorrect) from PSR_labels_raw.csv, forward filled; class 0 absent, 1 correct, 2 incorrect"""
    rows = []
    for line in open(video_dir(split, v) / "PSR_labels_raw.csv"):
        p = line.strip().split(",")
        if len(p) >= 12:
            rows.append((int(p[0][:-4]), [int(x) for x in p[1:12]]))
    out, near, ri, cur = [], [], 0, rows[0][1]
    changes = [f for i, (f, s) in enumerate(rows) if i and s != rows[i - 1][1]]
    for t in ts:
        while ri < len(rows) and rows[ri][0] <= t:
            cur = rows[ri][1]; ri += 1
        out.append([{0: 0, 1: 1, -1: 2}[b] for b in cur[1:]])
        near.append(min([abs(t - f) for f in changes] or [1e9]) <= GUARD)
    return np.array(out, np.int64), np.array(near)


def state_series(split, v, ts):
    if THREE:
        return raw_series(split, v, ts)
    groups = gt_groups(load_psr_gt(split, v))
    st = bits(INITIAL_STATE[procedure_type(v)]); gi = 0; out = []
    for t in ts:
        while gi < len(groups) and groups[gi][0] <= t:
            st = apply_steps(st, groups[gi][1]); gi += 1
        out.append([max(0, b) for b in st[1:]])
    near = np.array([min([abs(t - f) for f, _ in groups] or [1e9]) <= GUARD for t in ts])
    return np.array(out, np.float32), near


def _img(x):
    """decoded RGB crop from a raw array or JPEG bytes"""
    if isinstance(x, (bytes, bytearray)):
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x


_KSEL = {}


def kshot_pick(sp, v, items, t, q, y, near):
    """fixed rule (same as v7_kshot.py): per state, round-robin over the training videos (sorted), best view
    quality first, next pick from the same video >= 50 crops later; K picks per state in total"""
    key = tuple(items)
    if key not in _KSEL:
        rules = ["view", "early", "late"] if KPICK == "mix" else [KPICK]
        pools = {r: {} for r in rules}
        for sp2, v2 in sorted(items, key=lambda it: it[1]):
            z2 = np.load(CACHE / "crops" / sp2 / f"{v2}.npz")
            y2, n2 = state_series(sp2, v2, z2["t"])
            codes = ["".join(str(int(b)) for b in r) for r in y2]
            tt = z2["t"].astype(int)
            start, end, st0 = [], [None] * len(codes), 0
            for i in range(len(codes)):
                if i and codes[i] != codes[i - 1]:
                    st0 = int(tt[i])
                start.append(st0)
            nxt = None                                   # first frame of the NEXT state (None for the last state)
            for i in range(len(codes) - 1, -1, -1):
                if i < len(codes) - 1 and codes[i] != codes[i + 1]:
                    nxt = int(tt[i + 1])
                end[i] = nxt
            base = np.where(~n2)[0]
            for r in rules:
                if r == "early":                         # 20-100 frames after the state began (step just done)
                    cand = [i for i in base if 20 <= tt[i] - start[i] <= 100]
                elif r == "late":                        # 20-100 frames before the state ends (next part placed, not done)
                    cand = [i for i in base if end[i] is not None and 20 <= end[i] - tt[i] <= 100]
                else:
                    cand = base
                for i in cand:
                    pools[r].setdefault(codes[i], {}).setdefault(v2, []).append((float(z2["q"][i]), int(i)))
        sel = {}
        for r, pool in pools.items():
            for s_, by_vid in pool.items():
                vids = sorted(by_vid)
                for vv in vids:
                    by_vid[vv].sort(reverse=True)
                pick, rr = [], 0
                while len(pick) < KSHOT and any(rr < len(by_vid[vv]) for vv in vids):
                    for vv in vids:
                        if rr < len(by_vid[vv]) and len(pick) < KSHOT:
                            pick.append((vv, by_vid[vv][rr][1]))
                    rr += 50
                for vv, i in pick:
                    if i not in sel.setdefault(vv, []):
                        sel[vv].append(i)
        _KSEL[key] = sel
        print("few-shot: labelled crops", sum(len(x) for x in sel.values()), "rules", rules, flush=True)
    return _KSEL[key].get(v, [])


def load(items, with_labels=True):
    X, Y, meta = [], [], []
    for sp, v in items:
        if RES:
            z = dict(np.load(CACHE / "crops_hr" / sp / f"{v}.npz", allow_pickle=True))
            z["img"] = z["jpg"]
        else:
            z = np.load(CACHE / "crops" / sp / f"{v}.npz")
        if with_labels:
            y, near = state_series(sp, v, z["t"])
            keep = ~near
            if KSHOT:
                keep = np.zeros(len(near), bool)
                for i in kshot_pick(sp, v, items, z["t"], z["q"], y, near):
                    keep[i] = True
            X.append(z["img"][keep]); Y.append(y[keep])
        else:
            X.append(z["img"]); meta.append((sp, v, z["t"], z["q"]))
    return (np.concatenate(X), np.concatenate(Y)) if with_labels else (X, meta)


def augment(b, cad=None):
    """b: uint8 B,H,W,3 numpy -> float tensor B,3,SIZE,SIZE with light geometric/colour jitter."""
    out = []
    for j, im in enumerate(b):
        im = _img(im)
        if cad is not None and cad[j] and VCADRAND:
            im = cad_randomize(im)
        h, w = im.shape[:2]
        ang = np.random.uniform(-25, 25); sc = np.random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w / 2 + np.random.uniform(-8, 8), h / 2 + np.random.uniform(-8, 8)), ang, sc)
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        im = cv2.resize(im, (SIZE, SIZE)).astype(np.float32) / 255
        im = im * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08)
        if np.random.rand() < 0.3:
            im = cv2.GaussianBlur(im, (5, 5), np.random.uniform(0.5, 2.0))
        out.append(np.clip(im, 0, 1))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)


_BG = None


def cad_randomize(im):
    """CAD render (RGB uint8, flat background 214,214,209) -> real background patch, random skin-coloured
    occluders, per-channel colour gain and sensor noise.  Backgrounds come from training frames away from the
    assembly (cache/v7/bg_pool.npz)."""
    global _BG
    if _BG is None:
        _BG = np.load(CACHE / "v7" / "bg_pool.npz", allow_pickle=True)["jpg"]
    im = im.copy()
    bg = (np.abs(im.astype(np.int16) - np.array([214, 214, 209], np.int16)).max(-1) < 14)
    b = cv2.resize(_img(_BG[np.random.randint(len(_BG))]), im.shape[1::-1])
    im[bg] = b[bg]
    h, w = im.shape[:2]
    for _ in range(np.random.choice([0, 1, 1, 2])):                 # hand-like occluders
        tone = np.random.uniform(0.25, 1.0)                       # dark to light skin
        c = np.array([tone * 230, tone * 175, tone * 150]) + np.random.uniform(-15, 15, 3)
        cv2.ellipse(im, (int(np.random.uniform(0, w)), int(np.random.uniform(0, h))),
                    (int(np.random.uniform(0.08, 0.22) * w), int(np.random.uniform(0.05, 0.12) * h)),
                    float(np.random.uniform(0, 180)), 0, 360, c.tolist(), -1)
    im = im.astype(np.float32) * np.random.uniform(0.9, 1.1, 3) + np.random.normal(0, 6, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def to_tensor(b):
    return torch.from_numpy(np.stack([cv2.resize(_img(im), (SIZE, SIZE)) for im in b]).astype(np.float32) / 255).permute(0, 3, 1, 2)


def train(items, seed=0, extra=None, steps_override=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if VZERO:
        X, Y = np.empty(0, dtype=object), np.zeros((0, 10), np.float32)
    else:
        X, Y = load(items)
    isc = np.zeros(len(X), bool)
    if VCAD:                                   # synthetic CAD images; real crops repeated to ~half of each batch
        zc = np.load(CACHE / "v7" / "cad_imgs.npz", allow_pickle=True)
        Yc = np.array([[int(c) for c in code[1:]] for code in zc["code"]], np.float32)
        reps = max(1, len(Yc) // max(1, len(X)))
        X = np.concatenate([X] * reps + [zc["jpg"]]); Y = np.concatenate([Y] * reps + [Yc])
        isc = np.r_[np.zeros(len(X) - len(Yc), bool), np.ones(len(Yc), bool)]
    if extra is not None:                      # pseudo-labelled unlabelled crops (self-training)
        X = np.concatenate([X, extra[0]]); Y = np.concatenate([Y, extra[1]]); isc = np.r_[isc, np.zeros(len(extra[1]), bool)]
        print("with CAD:", len(Yc), "renders, real repeated", reps, "x", flush=True)
    if THREE:
        cnt = np.stack([(Y == c).sum(0) for c in range(3)], 1).astype(np.float32) + 1   # 10 x 3
        cw = torch.from_numpy((cnt.sum(1, keepdims=True) / cnt) ** 0.5).cuda()          # sqrt inverse frequency
        cw = cw.clamp(max=20.0)
        print(f"train frames {len(X)}  incorrect frames per component {(Y == 2).sum(0).tolist()}", flush=True)
    else:
        print(f"train frames {len(X)}  positive rate per component {Y.mean(0).round(2).tolist()}", flush=True)
        pos = torch.from_numpy(Y.mean(0)).clamp(0.05, 0.95)
        pw = ((1 - pos) / pos).sqrt().cuda()                       # mild re-balancing
    m = Verifier().cuda()
    opt = torch.optim.AdamW([{"params": [p for p in m.backbone.parameters() if p.requires_grad], "lr": 2e-5},
                             {"params": m.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    if KSHOT or VZERO:                           # few-shot / zero-shot: fixed number of steps
        global EPOCHS
        EPOCHS = max(1, -(-(steps_override or KSTEPS) * BS // len(X)))
    steps = EPOCHS * (len(X) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    m.train(); it = 0; t0 = time.time()
    for ep in range(EPOCHS):
        perm = np.random.permutation(len(X)); tot = 0
        for k in range(0, len(perm) - BS + 1, BS):
            idx = perm[k:k + BS]
            x = augment(X[idx], isc[idx]).cuda(non_blocking=True); y = torch.from_numpy(Y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                if THREE:
                    lg = m(x).float().view(-1, 10, 3)
                    loss = sum(F.cross_entropy(lg[:, k], y[:, k], weight=cw[k]) for k in range(10)) / 10
                else:
                    loss = F.binary_cross_entropy_with_logits(m(x).float(), y, pos_weight=pw)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); it += 1
        print(f"epoch {ep} loss {tot / max(1, len(perm) // BS):.4f}  {time.time() - t0:.0f}s", flush=True)
    return m.eval()


def self_train(m, tr_items, seed):
    """pseudo-label the UNLABELLED crops of the training videos with model m (temporal median over +-7 crops,
    keep crops where every part is confidently on or off), then retrain with real + CAD + pseudo-labels.
    No ground truth is read here."""
    Xs, meta = load(tr_items, with_labels=False)
    PX, PY = [], []
    P = [predict(m, imgs) for imgs in Xs]
    if SELF_MODE >= 2:
        from v7_pseudo import filter_pseudo
        fp = filter_pseudo(tr_items, {mt[1]: (mt[2], p, mt[3]) for mt, p in zip(meta, P)})
    for imgs, (sp, v, t, q), p in zip(Xs, meta, P):
        sm = np.stack([np.median(p[max(0, i - 7):i + 8], 0) for i in range(len(p))])
        if SELF_MODE >= 2:
            lab, ok = fp[v]
            if SELF_MODE == 3:
                ok = ok & (np.abs(sm - lab) < 0.5).all(1)
            PX.append(imgs[ok]); PY.append(lab[ok])
        else:
            ok = ((sm > 0.9) | (sm < 0.1)).all(1)
            PX.append(imgs[ok]); PY.append((sm[ok] > 0.5).astype(np.float32))
    PX, PY = np.concatenate(PX), np.concatenate(PY)
    sel = np.random.default_rng(seed).choice(len(PX), min(SELF_N, len(PX)), replace=False)
    print("self-training: pseudo-labelled", len(PX), "kept", len(sel), flush=True)
    del m; torch.cuda.empty_cache()
    return train(tr_items, seed=seed + 100, extra=(PX[sel], PY[sel]), steps_override=SELF_STEPS)


@torch.no_grad()
def predict(m, imgs, bs=128):
    out = []
    for k in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(to_tensor(imgs[k:k + bs]).cuda()).float()
            out.append((torch.softmax(lg.view(-1, 10, 3), -1).reshape(-1, 30) if THREE else torch.sigmoid(lg)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 30 if THREE else 10), np.float32)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fold":
        i = int(sys.argv[2]); held = set(FOLDS[i])
        tr_items = [it for it in ITEMS if it[1][:2] not in held]
        m = train(tr_items, seed=i)
        if VSELF:
            for rnd in range(SELF_ROUNDS):
                m = self_train(m, tr_items, i + 1000 * rnd)
        X, meta = load([it for it in ITEMS if it[1][:2] in held], with_labels=False)
        od = CACHE / f"{TAG}_oof"; od.mkdir(exist_ok=True)
        from sklearn.metrics import roc_auc_score
        aucs = []
        for imgs, (sp, v, t, q) in zip(X, meta):
            p = predict(m, imgs)
            np.savez(od / f"{v}.npz", t=t, p=p.astype(np.float16), q=q)
            y, near = state_series(sp, v, t); aucs.append((y[~near], p[~near]))
        Yh = np.concatenate([a for a, _ in aucs]); Ph = np.concatenate([b for _, b in aucs])
        if THREE:
            Ph = Ph.reshape(-1, 10, 3); res = {}
            for k in range(10):
                y = Yh[:, k]
                on = y > 0
                if on.any() and len(set(y[on])) == 2:   # correct vs incorrect among attached frames
                    res[f"{k + 1}_inc"] = round(roc_auc_score(y[on] == 2, Ph[on, k, 2]), 3)
                att = Ph[:, k, 1] + Ph[:, k, 2]
                if 0 < on.mean() < 1:
                    res[f"{k + 1}_on"] = round(roc_auc_score(on, att), 3)
        else:
            res = {k + 1: round(roc_auc_score(Yh[:, k], Ph[:, k]), 3) for k in range(10) if 0 < Yh[:, k].mean() < 1}
        print("held-out subjects", sorted(held), "AUC per component", res, flush=True)
        (od / f"fold{i}_auc.json").write_text(json.dumps(res))
    elif mode == "final":
        m = train(ITEMS, seed=7)
        torch.save(m.state_dict(), CACHE / f"{TAG}_final.pt")
        print("saved", flush=True)
