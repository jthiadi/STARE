"""STARE-lite on MECCANO, procedure step recognition (PREREGISTRATION.md). One module, one subcommand per stage:

  python metrics/meccano/stare_lite.py tcn          # GPU: causal TCN on the frozen features -> work/ar/<vid>.npz
  python metrics/meccano/stare_lite.py steps        # CPU: triggers + step model (NB tables + LR) from train -> work/steps/
  python metrics/meccano/stare_lite.py locproto     # GPU: localizer prototypes from train state boxes -> work/vision/
  python metrics/meccano/stare_lite.py crops        # GPU: train crops (every 12th frame) -> work/vision/train_crops.npz
  python metrics/meccano/stare_lite.py verifier     # GPU: DINOv2-base 17-part verifier -> work/vision/verifier.pt
  python metrics/meccano/stare_lite.py vprobs k/n   # GPU: even frames of val + test: localize, quality, verifier -> work/vprobs/
  python metrics/meccano/stare_lite.py run          # CPU: STARE filter on val + test, scored with STORM-PSR's evaluator
Everything at frame t uses frames <= t only. Labels are read for training (train split) and for scoring only."""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYS = ROOT / "system"
sys.path[:0] = [str(SYS / "psr_v14"), str(SYS / "assistant")]
MEC = Path(os.environ.get("MECCANO_ROOT", ROOT / "data/MECCANO"))
VIDEOS = Path(os.environ.get("MECCANO_VIDEOS", MEC / "MECCANO_RGB_Videos"))
PSR = ROOT / "third_party/PSR-annotations/x/MECCANO_PSR_Annotations"
ASD = ROOT / "third_party/PSR-annotations/xa/MECCANO_ASD_Annotations"
PROC = ROOT / "third_party/STORM-PSR/evaluation/utils/procedure_info_MECCANO.json"
WORK = Path(os.environ.get("MECCANO_WORK", HERE / "work"))
FPS, K = 12, 17
MATCH_BEFORE, MATCH_AFTER = 84, 18            # IndustReal's 70 / 15 frames at 10 fps, as 12 fps
D0 = 14                                        # IndustReal's 12 frames (1.2 s) at 12 fps
SPLITS = {sp: sorted(os.listdir(PSR / sp)) for sp in ("train", "val", "test")}
SPLIT_OF = {v: sp for sp, vs in SPLITS.items() for v in vs}
PROC_INFO = json.loads(PROC.read_text())
STEP_DESC = {p["id"]: p["description"] for p in PROC_INFO}


# ---------------------------------------------------------------------------------------------------- data
def n_frames(v):
    import cv2
    c = cv2.VideoCapture(str(VIDEOS / f"{v}.mp4")); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return n


def actions_table():
    names, segs = {}, defaultdict(list)
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(MEC / "annotations" / f"MECCANO_{sp}_actions.csv", encoding="utf-8")):
            names[int(r["action_id"])] = r["action_name"]
            segs[r["video_id"]].append((r["action_name"], int(r["start_frame"][:-4]), int(r["end_frame"][:-4])))
    return [names[i] for i in sorted(names)], segs


def gt_steps(v):                                 # scoring / training only
    return [{"frame": int(r[0][:-4]), "id": int(r[1]), "description": r[2]}
            for r in csv.reader(open(PSR / SPLIT_OF[v] / v / "PSR_labels.csv", encoding="utf-8"))]


def gt_states(v):                                # train only: piecewise-constant state per frame
    rows = [(int(r[0][:-4]), [int(x) for x in r[1:]]) for r in csv.reader(open(PSR / SPLIT_OF[v] / v / "PSR_labels_raw.csv"))]
    return sorted(rows)


GENERIC = {"check_booklet", "browse_booklet"}


# ---------------------------------------------------------------------------------------------------- 1. action model
def load_x(v):
    d = np.load(WORK / "feats/dino" / f"{v}.npz")["feats"].astype(np.float32)
    s = np.load(WORK / "feats/ssv2" / f"{v}.npz")
    x = np.concatenate([d.reshape(len(d), -1), s["feats"].reshape(len(d), -1).astype(np.float32),
                        s["video_logits"].reshape(len(d), -1).astype(np.float32)], 1)
    return x


def frame_labels(v, n, names, segs):
    idx = {a: i + 1 for i, a in enumerate(names)}
    y = np.zeros(n, np.int64)
    for a, s, e in segs[v]:
        y[s:min(e, n - 1) + 1] = idx[a]
    return y


def cmd_tcn():
    import torch
    import torch.nn.functional as F
    from scripts.segmentation.train_ar_object_tcn import TCN
    names, segs = actions_table()
    data = {}
    for v in SPLIT_OF:
        x = load_x(v); data[v] = (x, frame_labels(v, len(x), names, segs))
    mu = np.concatenate([data[v][0] for v in SPLITS["train"]]).mean(0)
    sd = np.concatenate([data[v][0] for v in SPLITS["train"]]).std(0) + 1e-6
    dev = "cuda"
    T = {v: (torch.from_numpy(((x - mu) / sd).T[None]).float().to(dev), torch.from_numpy(y).to(dev)) for v, (x, y) in data.items()}
    torch.manual_seed(0)
    m = TCN(len(mu), len(names) + 1, hidden=192, layers=10, stages=4, causal=True).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-4)
    best, best_ep, best_sd = -1, -1, None
    for ep in range(60):
        m.train()
        for v in np.random.default_rng(ep).permutation(SPLITS["train"]):
            x, y = T[v]
            loss = sum(F.cross_entropy(o[0].T, y) for o in m(x))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            ok = tot = 0
            for v in SPLITS["val"]:
                x, y = T[v]; p = m(x)[-1][0].argmax(0)
                ok += int((p == y).sum()); tot += len(y)
        acc = ok / tot
        if acc > best:
            best, best_ep, best_sd = acc, ep, {k: t.detach().clone() for k, t in m.state_dict().items()}
        print(json.dumps({"epoch": ep, "val_frame_acc": round(acc, 4)}), flush=True)
    m.load_state_dict(best_sd); m.eval()
    (WORK / "ar").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_sd, "mu": mu, "sd": sd, "names": names, "epoch": best_ep}, WORK / "ar/tcn.pt")
    with torch.no_grad():
        for v in SPLIT_OF:
            p = torch.softmax(m(T[v][0])[-1][0], 0).T.cpu().numpy()
            np.savez_compressed(WORK / "ar" / f"{v}.npz", top=p.argmax(1).astype(np.int16), conf=p.max(1).astype(np.float16))
    print(json.dumps({"chosen_epoch": best_ep, "val_frame_acc": round(best, 4)}), flush=True)


def ar_names_per_frame(v, names, oracle=False, segs=None, n=None):
    if oracle:                                   # training of the step model: annotated actions (as IndustReal)
        y = frame_labels(v, n, names, segs)
    else:
        y = np.load(WORK / "ar" / f"{v}.npz")["top"].astype(int)
    return [[] if c == 0 else [names[c - 1]] for c in y]


# ---------------------------------------------------------------------------------------------------- 2. triggers + step model
def choose_triggers(names, segs):
    inwin = lambda e, f: e - MATCH_BEFORE <= f <= e + MATCH_AFTER
    comps = {v: sorted({g["frame"] for g in gt_steps(v)}) for v in SPLITS["train"]}
    occ, hit = Counter(), Counter()
    for v in comps:
        for a, s, e in segs[v]:
            occ[a] += 1; hit[a] += any(inwin(e, f) for f in comps[v])
    rank = sorted((a for a in occ if occ[a] >= 10), key=lambda a: (-hit[a] / occ[a], a))
    total = sum(len(c) for c in comps.values()); chosen = []
    for a in rank:
        chosen.append(a)
        cov = sum(any(inwin(e, f) for b, s, e in segs[v] if b in chosen) for v in comps for f in comps[v]) / total
        if cov >= 0.8:
            break
    return chosen, round(cov, 3)


def patch_psr(triggers):
    import psr.events as EV
    import psr.filter_tracker as FT
    import psr.core as CORE
    EV.TRIGGERS = set(triggers); EV.GENERIC = GENERIC; FT.GENERIC = GENERIC
    FT.PROC_INFO = PROC_INFO; FT.STEP_DESC = STEP_DESC; FT.INITIAL_STATE = {"assy": "0" * K}
    CORE.PROC_INFO = PROC_INFO; CORE.STEP_DESC = STEP_DESC


def gt_groups(v):
    by = defaultdict(list)
    for g in gt_steps(v):
        by[g["frame"]].append(g["id"])
    return [(f, tuple(sorted(x))) for f, x in sorted(by.items())]


def cmd_steps():
    from psr.core import apply_steps, code
    from psr.events import EventExtractor
    names, segs = actions_table()
    trig, cov = choose_triggers(names, segs)
    patch_psr(trig)
    samples, recall = [], [0, 0]
    for v in SPLITS["train"]:
        n = n_frames(v)
        obs = ar_names_per_frame(v, names, oracle=True, segs=segs, n=n)
        ex, trs = EventExtractor(), []
        for t in range(n):
            trs += ex.update(t, obs[t])
        groups = gt_groups(v)
        pairs = sorted((abs(tr.frame - gf), gi, ti) for gi, (gf, g) in enumerate(groups) for ti, tr in enumerate(trs)
                       if -MATCH_AFTER <= tr.frame - gf <= MATCH_BEFORE)
        lab, used = [None] * len(trs), set()
        for _, gi, ti in pairs:
            if gi not in used and lab[ti] is None:
                lab[ti] = gi; used.add(gi)
        recall[0] += len(used); recall[1] += len(groups)
        state = tuple([0] * K); gi_done = 0; since = set(); ntrig = 0; last = 0
        for ti, tr in enumerate(trs):
            ntrig += 1
            while gi_done < len(groups) and groups[gi_done][0] < tr.frame - MATCH_BEFORE and gi_done not in used:
                state = apply_steps(state, groups[gi_done][1]); gi_done += 1
            since |= set(tr.window)
            g = groups[lab[ti]][1] if lab[ti] is not None else ()
            samples.append({"proc": "assy", "state": code(state), "trigger": tr.name, "window": sorted(tr.window),
                            "since": sorted(since), "ntrig": min(ntrig, 3), "ntrig_raw": ntrig, "duration": tr.duration,
                            "elapsed": tr.frame - last, "group": list(g)})
            if lab[ti] is not None:
                state = apply_steps(state, g); gi_done = max(gi_done, lab[ti] + 1); since = set(); ntrig = 0; last = tr.frame
    key = lambda g: ",".join(map(str, g))
    vocab = sorted({tuple(s["group"]) for s in samples})
    tables = {
        "groups": [list(g) for g in vocab],
        "actions": sorted({a for s in samples for a in s["window"] + s["since"]}),
        "triggers": sorted({s["trigger"] for s in samples}),
        "n_group": {key(g): c for g, c in Counter(tuple(s["group"]) for s in samples).items()},
        "n_group_proc": {f"assy|{key(g)}": c for g, c in Counter(tuple(s["group"]) for s in samples).items()},
        "n_group_proc_state": {f"assy|{st}|{key(g)}": c for (st, g), c in Counter((s["state"], tuple(s["group"])) for s in samples).items()},
        "n_proc_state": {f"assy|{st}": c for st, c in Counter(s["state"] for s in samples).items()},
        "n_trig": {f"{key(g)}|{t}": c for (g, t), c in Counter((tuple(s["group"]), s["trigger"]) for s in samples).items()},
        "n_act": {f"{key(g)}|{a}": c for (g, a), c in Counter((tuple(s["group"]), a) for s in samples for a in s["window"]).items()},
        "n_ntrig": {f"{key(g)}|{n}": c for (g, n), c in Counter((tuple(s["group"]), s["ntrig"]) for s in samples).items()},
        "n_since": {f"{key(g)}|{a}": c for (g, a), c in Counter((tuple(s["group"]), a) for s in samples for a in s["since"]).items()},
    }
    from psr.lr_model import LRGroupModel
    lr = LRGroupModel(samples=samples, C=0.5)
    out = WORK / "steps"; out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(tables))
    lr.save(out / "lr.pkl")
    info = {"triggers": trig, "train_completion_coverage": cov, "samples": len(samples),
            "train_groups_with_trigger": f"{recall[0]}/{recall[1]}", "groups": len(vocab)}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


# ---------------------------------------------------------------------------------------------------- 3. vision
FRAME_W, FRAME_H = 1280, 720                     # frames resized to the IndustReal geometry the localizer code uses


FRAMES = Path(os.environ.get("MECCANO_FRAMES", MEC / "frames_960"))   # every frame as a 960 x 540 JPEG (<vid>.bin + .idx.npy)


def frames(v, step=1):
    import cv2
    if (FRAMES / f"{v}.bin").exists():          # the frame cache: same frames as the video, no 1080p decoding
        idx = np.load(FRAMES / f"{v}.idx.npy"); data = np.memmap(FRAMES / f"{v}.bin", np.uint8, "r")
        for t in range(0, len(idx) - 1, step):
            bgr = cv2.imdecode(np.asarray(data[idx[t]:idx[t + 1]]), cv2.IMREAD_COLOR)
            yield t, cv2.cvtColor(cv2.resize(bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return
    c = cv2.VideoCapture(str(VIDEOS / f"{v}.mp4")); t = 0
    while True:
        if t % step == 0:
            ok, bgr = c.read()
            if not ok:
                break
            yield t, cv2.cvtColor(cv2.resize(bgr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        elif not c.grab():
            break
        t += 1
    c.release()


def asd_boxes(v):
    """train state boxes: frame -> (x, y, w, h) in 1280 x 720 pixels (YOLO-normalised centre format)."""
    out = {}
    f = ASD / "train" / f"{v}_GT.csv"
    if not f.exists():
        return out
    for r in csv.reader(open(f)):
        fr, cx, cy, w, h = int(r[0]), *map(float, r[3:7])
        out[fr] = ((cx - w / 2) * FRAME_W, (cy - h / 2) * FRAME_H, w * FRAME_W, h * FRAME_H)
    return out


def cmd_locproto():
    from psr.vision import DinoEncoder, VisionVerifier
    enc = DinoEncoder("cuda"); FW, FH = VisionVerifier.FRAME_SIZE
    gw, gh = FW // 14, FH // 14
    rng = np.random.default_rng(1); fg, bg = [], []
    for v in SPLITS["train"]:
        boxes = asd_boxes(v)
        pick = set(rng.choice(sorted(boxes), min(30, len(boxes)), replace=False).tolist()) if boxes else set()
        for t, rgb in frames(v):
            if t not in pick:
                continue
            x, y, w, h = boxes[t]
            _, pt = enc.encode(rgb, (FW, FH))
            sx, sy = FRAME_W / gw, FRAME_H / gh
            yy, xx = np.mgrid[0:gh, 0:gw]; cx, cy = (xx + 0.5) * sx, (yy + 0.5) * sy
            inside = (cx > x) & (cx < x + w) & (cy > y) & (cy < y + h)
            core = (cx > x + 0.2 * w) & (cx < x + 0.8 * w) & (cy > y + 0.2 * h) & (cy < y + 0.8 * h)
            fg.append(pt[core]); bg.append(pt[~inside])
    fg, bg = np.concatenate(fg), np.concatenate(bg)
    out = WORK / "vision"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "localizer_protos.npz", fg=fg[rng.choice(len(fg), min(4000, len(fg)), replace=False)].astype(np.float16),
             bg=bg[rng.choice(len(bg), min(8000, len(bg)), replace=False)].astype(np.float16))
    print(json.dumps({"fg": len(fg), "bg": len(bg)}), flush=True)


class Localizer:
    """STARE's localizer (psr/vision.py VisionVerifier.localize), prototypes from MECCANO train boxes."""

    def __init__(self):
        import torch
        from psr.vision import DinoEncoder, VisionVerifier
        self.enc = DinoEncoder("cuda"); self.size = VisionVerifier.FRAME_SIZE; self.torch = torch
        z = np.load(WORK / "vision/localizer_protos.npz")
        self.fg = torch.from_numpy(z["fg"]).cuda(); self.bg = torch.from_numpy(z["bg"]).cuda(); self.prev = None

    def __call__(self, rgb):
        import cv2
        _, pt = self.enc.encode(rgb, self.size)
        gh, gw, D = pt.shape
        p = self.torch.from_numpy(pt.reshape(-1, D)).half().cuda()
        score = ((p @ self.fg.T).topk(5, dim=1).values.mean(1) - (p @ self.bg.T).topk(5, dim=1).values.mean(1)).float()
        score = score.reshape(gh, gw).cpu().numpy()
        n, lab, stats, _ = cv2.connectedComponentsWithStats((score > 0).astype(np.uint8), 8)
        if n <= 1:
            return None
        best, bs = None, -1e9
        for i in range(1, n):
            x, y, w, h, _ = stats[i]; s = score[lab == i].sum()
            if self.prev is not None:
                px, py, pw, ph = self.prev
                d = np.hypot((x + w / 2) * FRAME_W / gw - (px + pw / 2), (y + h / 2) * FRAME_H / gh - (py + ph / 2)) / 640
                s *= np.exp(-d)
            if s > bs:
                best, bs = (x, y, w, h), s
        x, y, w, h = best
        box = [x * FRAME_W / gw, y * FRAME_H / gh, w * FRAME_W / gw, h * FRAME_H / gh]
        if box[2] * box[3] < 60 * 60:
            return None
        self.prev = box
        return box


def crop336(rgb, box):
    import cv2
    from psr.vision import crop
    return cv2.resize(crop(rgb, box, pad=0.1), (336, 336), interpolation=cv2.INTER_AREA)


def state_at(rows, t):
    s = [0] * K
    for f, st in rows:
        if f <= t:
            s = st
    return s


def cmd_crops():
    import cv2
    loc = Localizer(); X, Y = [], []
    for v in SPLITS["train"]:
        rows = gt_states(v); changes = [f for f, _ in rows]; loc.prev = None
        for t, rgb in frames(v, step=2):
            box = loc(rgb)
            if box is None or t % 12 or any(abs(t - c) <= 24 for c in changes):
                continue
            X.append(cv2.imencode(".jpg", cv2.cvtColor(crop336(rgb, box), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
            Y.append([1.0 if x != 0 else 0.0 for x in state_at(rows, t)])
        print(json.dumps({"crops_after": v, "n": len(X)}), flush=True)
    np.savez(WORK / "vision/train_crops.npz", jpg=np.array(X, dtype=object), y=np.array(Y, np.float32))


def verifier_model():
    import torch.nn as nn
    from transformers import AutoModel

    class V(nn.Module):                         # STARE's verifier (train_verifier.py), 17 parts
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            for p in self.backbone.parameters():
                p.requires_grad = False
            for blk in self.backbone.encoder.layer[-4:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
            self.head = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, K))

        def forward(self, x):
            import torch
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            h = self.backbone(pixel_values=(x - mean) / std).last_hidden_state
            return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
    return V()


def _batch(jpgs, aug=False, rng=None):
    import cv2
    import torch
    ims = []
    for b in jpgs:
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if aug:
            im = np.clip(im * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
            if rng.random() < 0.5:
                im = im[:, ::-1]
        ims.append(im)
    return torch.from_numpy(np.ascontiguousarray(np.stack(ims).transpose(0, 3, 1, 2)))


def cmd_verifier():
    import torch
    z = np.load(WORK / "vision/train_crops.npz", allow_pickle=True); X, Y = z["jpg"], z["y"]
    torch.manual_seed(0); rng = np.random.default_rng(0)
    m = verifier_model().cuda()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5 * 5, weight_decay=0.05)
    steps = 4 * (len(X) // 24); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=steps)
    scaler = torch.cuda.amp.GradScaler(); it = 0
    for ep in range(4):
        perm = rng.permutation(len(X))
        for i in range(len(X) // 24):
            idx = perm[i * 24:(i + 1) * 24]
            xb = _batch(X[idx], True, rng).cuda(); yb = torch.from_numpy(Y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xb), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); it += 1
            if it % 200 == 0:
                print(json.dumps({"epoch": ep, "step": it, "loss": round(float(loss), 4)}), flush=True)
    torch.save(m.state_dict(), WORK / "vision/verifier.pt")
    print("VERIFIER DONE", flush=True)


def cmd_vprobs(shard):
    import torch
    from psr.vision import VisionVerifier
    k, n = (int(x) for x in shard.split("/"))
    loc = Localizer(); m = verifier_model().cuda().eval(); m.load_state_dict(torch.load(WORK / "vision/verifier.pt"))
    out = WORK / "vprobs"; out.mkdir(parents=True, exist_ok=True)
    for v in (SPLITS["val"] + SPLITS["test"])[k::n]:
        if (out / f"{v}.npz").exists():
            continue
        loc.prev = None; T, P, Q = [], [], []
        for t, rgb in frames(v, step=2):            # even frames, as STARE
            box = loc(rgb)
            if box is None:
                continue
            c = crop336(rgb, box)
            q, _ = VisionVerifier.view_quality(c)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(m(torch.from_numpy(c.astype(np.float32).transpose(2, 0, 1)[None] / 255).cuda()))[0]
            T.append(t); P.append(p.float().cpu().numpy()); Q.append(q)
        np.savez_compressed(out / f"{v}.npz", t=np.array(T), p=np.array(P, np.float16), q=np.array(Q, np.float32))
        print(json.dumps({"vprobs": v, "readings": len(T)}), flush=True)


# ---------------------------------------------------------------------------------------------------- 4. tracker + scoring
def make_tracker_class():
    from psr.v11 import V11Tracker

    class LiteTracker(V11Tracker):
        """STARE's filter (v11 = v9 filter + short-trigger weighting) for K = 17 components, all tracked."""

        def _emit(self, lp, w=1.0):
            tau = self.f.tau * w
            for s in list(self.belief):
                self.belief[s] += tau * sum(lp[k][{0: 0, 1: 1, -1: 2}[s[k]]] for k in range(K))
            self._norm()

        def _confirmed(self, s_new):
            changed = [k for k in range(K) if s_new[k] != self.reported[k]]
            recent = self.hist[-self.f.conf_n:]
            if not changed or len(recent) < self.f.conf_min:
                return False, 0.0
            vals = []
            for lp in recent:
                tot = 0.0
                for k in changed:
                    pc = min(max(math.exp(lp[k][{0: 0, 1: 1, -1: 2}[s_new[k]]]), 1e-3), 1 - 1e-3)
                    tot += math.log(pc / (1 - pc))
                vals.append(tot / len(changed))
            m = sum(vals) / len(vals)
            return m >= self.f.conf_thr, m

        def _recover(self, t, lp):
            """filter_tracker._recover for K components (visual recovery, option rec_n)."""
            if not hasattr(self, "contra_k"):
                self.contra_k = [0] * K
            for k in range(K):
                on = lp[k][1] - lp[k][0]; cur = self.reported[k]
                against = (on >= self.f.rec_thr and cur != 1) or (on <= -self.f.rec_thr and cur == 1)
                self.contra_k[k] = self.contra_k[k] + 1 if against else 0
            cands = [k for k in range(K) if self.contra_k[k] >= self.f.rec_n]
            if not cands:
                return []
            k = max(cands, key=lambda c: self.contra_k[c]); self.contra_k[k] = 0
            new = list(self.reported); new[k] = 1 if self.reported[k] != 1 else 0; new = tuple(new)
            steps = self._steps_between(self.reported, new)
            if not steps or (self.f.rec_expected and not all(s in self.expected for s in steps)):
                return []
            self.reported = new; self.belief = {new: 0.0}
            self.since = set(); self.ntrig = 0; self.last_commit = t
            entries = [{"frame": t, "id": s, "description": STEP_DESC[s], "conf": 1} for s in steps if s % 3 != 1]
            self.y_hat += entries
            return entries

        def update(self, t, ar_names, feat=None):
            self.since |= {n for n in ar_names if n not in GENERIC}
            for trig in self.events.update(t, ar_names):
                self.ntrig += 1; self.last_trigger = t
                self._trigger(t, trig, {"duration": trig.duration, "elapsed": t - self.last_commit, "tsince": trig.tsince})
            self._drift(self.f.eps)                       # drift prior (0 in the frozen IndustReal settings)
            if feat is None:
                return []
            lp = {k: tuple(math.log(min(max(x, 1e-4), 1.0)) for x in (1 - feat["p"][k], feat["p"][k], 1e-3)) for k in range(K)}
            self.hist.append(lp); self.hist = self.hist[-50:]
            w = min(1.0, max(self.f.q_min, feat["q"] / self.f.q_ref)) if self.f.q_ref > 0 else 1.0
            self._emit(lp, w)
            out = self._decide(t)
            if not out and self.f.rec_n:
                out = self._recover(t, lp)
            return out
    return LiteTracker


def chosen_settings():
    """The reporting settings selected on MECCANO validation (results/retune.json, PREREGISTRATION_retune.md)."""
    c = json.loads((HERE / "results/retune.json").read_text())["chosen"]["config"]
    return {**c, "rec_thr": 3.0, "rec_expected": True}


def cmd_run(splits=("val", "test")):
    override = os.environ.get("MECCANO_FILTER_OVERRIDE")
    res = evaluate(json.loads(override) if override else chosen_settings(), splits)
    out = Path(os.environ.get("MECCANO_RESULTS", HERE / "results"))     # reproduce_all.py points this elsewhere
    out.mkdir(parents=True, exist_ok=True)
    (out / "meccano_psr.json").write_text(json.dumps(res, indent=1))


def evaluate(override, splits):
    sys.path.insert(0, str(ROOT / "third_party/STORM-PSR/evaluation"))
    from utils import utils as UT                                       # STORM-PSR's evaluator, unchanged
    from psr.blend import Blend
    from psr.filter_tracker import FilterConfig
    from psr.lr_model import LRGroupModel
    from psr.tracker import GroupModel, TrackerConfig
    names, _ = actions_table()
    info = json.loads((WORK / "steps/info.json").read_text())
    patch_psr(info["triggers"])
    frozen = json.loads((SYS / "psr_v14/model/config.json").read_text())   # IndustReal's frozen settings, unchanged
    sm = frozen["step_model"]
    model = Blend(GroupModel(json.loads((WORK / "steps/tables.json").read_text())), LRGroupModel(path=WORK / "steps/lr.pkl"), sm["w_lr"])
    tcfg = TrackerConfig(w_action=sm["w_action"], w_since=sm["w_since"], null_bias=sm["null_bias"])
    fcfg = FilterConfig(**{**frozen["filter"], **override})
    Tracker = make_tracker_class()
    res = {"settings": {"filter": vars(fcfg), "step_model": sm, "d0_frames": D0, **info}}
    for sp in splits:
        agg, rows = UT.initiate_metrics(), []
        for v in SPLITS[sp]:
            z = np.load(WORK / "vprobs" / f"{v}.npz")
            feats = {int(t): {"p": p.astype(np.float32), "q": float(q)} for t, p, q in zip(z["t"], z["p"], z["q"])}
            obs = ar_names_per_frame(v, names)
            tr = Tracker("assy", fcfg, model, tcfg, d0=D0)
            for t in range(len(obs)):
                tr.update(t, obs[t], feats.get(t))
            gt = UT.load_psr_labels(PSR / sp / v / "PSR_labels.csv")
            met, det, log, delays = UT.determine_performance(gt, tr.y_hat, PROC_INFO, FPS=FPS)
            UT.update_metrics(agg, met)                                  # STORM-PSR's aggregation
            rows.append({"video": v, "pos": round(float(met["pos"]), 3), "f1": round(float(met["f1"]), 3),
                         "delay_s": round(float(met["avg_delay"]) / FPS, 1),
                         "steps": [(e["frame"], e["description"]) for e in tr.y_hat]})
        res[sp] = {"POS": round(float(UT.mean_list(agg["pos"])), 3), "F1": round(float(UT.mean_list(agg["f1"])), 3),
                   "delay_s": round(float(UT.mean_list(agg["avg_delay"])) / FPS, 1),
                   "TP": int(agg["system_TPs"]), "FP": int(agg["system_FPs"]), "FN": int(agg["system_FNs"]), "videos": rows}
        print(sp, json.dumps({k: res[sp][k] for k in ("POS", "F1", "delay_s")}), flush=True)
    return res


RETUNE_BASE = {"conf_thr": 5.0, "eps": 0.0, "rec_n": 0, "tau": 0.05}
RETUNE_GRID = dict(conf_thr=[1.0, 3.0, 5.0], eps=[0.0, 1e-4, 1e-3], rec_n=[0, 12], tau=[0.05, 0.1])


def _retune_one(c):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        r = evaluate({**c, "rec_thr": 3.0, "rec_expected": True}, ("val",))["val"]
    return {"config": c, "criterion": round(r["POS"] + r["F1"] - 0.02 * r["delay_s"], 4),
            "val": {k: r[k] for k in ("POS", "F1", "delay_s")}, "changed": sum(c[k] != RETUNE_BASE[k] for k in RETUNE_BASE)}


def cmd_retune():
    """Validation-only grid (PREREGISTRATION_retune.md, amended grid), 6 processes; writes results/retune.json."""
    import itertools
    from multiprocessing import get_context
    cfgs = [dict(zip(RETUNE_GRID, v)) for v in itertools.product(*RETUNE_GRID.values())]
    with get_context("fork").Pool(6) as pool:
        rows = []
        for r in pool.imap_unordered(_retune_one, cfgs):
            rows.append(r); print(json.dumps(r), flush=True)
    best = max(rows, key=lambda r: (round(r["criterion"], 6), -r["changed"], r["config"]["conf_thr"]))
    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / "retune.json").write_text(json.dumps({"chosen": best, "rows": sorted(rows, key=lambda r: -r["criterion"])}, indent=1))
    print("CHOSEN", json.dumps(best), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    {"tcn": cmd_tcn, "steps": cmd_steps, "locproto": cmd_locproto, "crops": cmd_crops, "verifier": cmd_verifier,
     "run": cmd_run, "run_val": lambda: cmd_run(("val",)), "run_test": lambda: cmd_run(("test",)),
     "retune": cmd_retune}.get(c, lambda: cmd_vprobs(sys.argv[2]))()
