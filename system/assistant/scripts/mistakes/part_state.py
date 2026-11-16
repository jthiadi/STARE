"""Spike: a fine-tuned backbone reads each part's state on the assembly crop (absent / mounted / mounted wrong).

The released state detector (YOLO) names the whole 11-part state as one class and almost never reports a wrongly
mounted part. Here every state-detector box is cropped (a square around it, 15 % margin) and a fine-tuned
backbone answers, for each of the 11 parts, absent / mounted / mounted wrong (PSR value 0 / 1 / -1):

  dino    DINOv2-small, all layers fine-tuned, the current frame only (336 px)
  vjepa   V-JEPA 2 ViT-L, last 12 layers fine-tuned, 16 frames ending at the current frame (every 2nd frame,
          3.2 s, causal), 256 px

Training: the 12,483 annotated assembly boxes of the 36 train videos (``OD_labels.json``), each labelled with the
annotated state at that frame (``PSR_labels_raw.csv``, which has -1 for wrongly mounted parts). The epoch is chosen
on val (exact-state accuracy on every 4th detection). Prediction: every box of our detector on val and of the
released detector on test, so each video gets ``asd_part.jsonl`` (same rows, per-part state, ``--state asdpart``).

Stop rule, fixed before any result: on the 16 val videos the chosen setting must find more wrongly mounted parts
than our detector alone (``pred_asdours``), with no more false alarms per hour and no lower step-recognition F1.
Otherwise the spike stops and test is not scored.

    python -m scripts.mistakes.part_state --train --model dino                          # GPU server
    python -m scripts.mistakes.part_state --train --model vjepa --batch 4 --accum 4     # GPU server
    python -m scripts.mistakes.part_state --select                                      # CPU: val + stop rule
    python -m scripts.mistakes.part_state --test                                        # CPU: only if val passed
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from assist.common import load_config, read_jsonl, write_json, write_jsonl
from assist.industreal import SPLITS, psr_states, task_of
from monitor.settings import ROOT

OUT = ROOT / "output/assist/part_state"
LABELS = ROOT / "data/evaluation/official"
CACHE = ROOT / "output/vjepa_frames_960"                    # every frame at 960 x 540 (vjepa2_finetune --cache)
SCALE = 960 / 1280                                          # detector boxes are in 1280 x 720 pixels
MARGIN = 0.15
ASD = {"val": "asd_ours.jsonl", "test": "asd.jsonl"}       # state-detector boxes per split (ours on val)
STATE = {"val": "asdours", "test": "asd"}
VALUES = (0, 1, -1)                                         # class index -> PSR value
PARTS = 11
CLIP, STRIDE = 16, 2
THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 1.01)                     # P(wrong) above which a part is called wrong; 1.01 never


# ---------------------------------------------------------------------------------------------- labels
def state_at(changes: list, initial: list[int], frame: int) -> list[int]:
    """The annotated state at ``frame``: the last change at or before it, else the task's initial state."""
    state = list(initial)
    for f, s in changes:
        if f > frame:
            break
        state = s
    return list(state)


def to_classes(state: list[int]) -> list[int]:
    return [VALUES.index(v) for v in state]


def to_state(p: np.ndarray, thr: float) -> tuple[list[int], float]:
    """(11, 3) probabilities -> (state values, confidence). A part is wrong when P(wrong) > ``thr``, else the
    likelier of absent / mounted; the confidence is the mean probability of the chosen answers."""
    state, conf = [], []
    for q in p:
        k = 2 if q[2] > thr else int(np.argmax(q[:2]))
        state.append(VALUES[k])
        conf.append(float(q[k]) if k < 2 else float(q[2]))
    return state, float(np.mean(conf))


def square(box: list[float], frame_w: int = 960, frame_h: int = 540) -> tuple[int, int, int, int]:
    """Detector box (x, y, w, h in 1280 x 720) -> a square crop (x1, y1, x2, y2) in cache pixels."""
    x, y, w, h = (v * SCALE for v in box)
    side = int(min(frame_h, (1 + 2 * MARGIN) * max(w, h)))
    cx = int(np.clip(x + w / 2, side / 2, frame_w - side / 2))
    cy = int(np.clip(y + h / 2, side / 2, frame_h - side / 2))
    return cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side


def _initial(rid: str, tasks) -> list[int]:
    return list(tasks[task_of(rid)].initial_state)


def train_items(tasks) -> list[tuple]:
    """(rid, frame, box, 11 classes) for every annotated assembly box of the train videos."""
    items = []
    for rid in SPLITS["train"]:
        d = json.loads((LABELS / rid / "OD_labels.json").read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in d["categories"]}
        frames = {im["id"]: int(im["file_name"].split(".")[0]) for im in d["images"]}
        changes, init = psr_states(rid, LABELS), _initial(rid, tasks)
        for a in d["annotations"]:
            if names[a["category_id"]].isdigit():
                f = frames[a["image_id"]]
                items.append((rid, f, a["bbox"], to_classes(state_at(changes, init, f))))
    return items


def eval_items(split: str, tasks, every: int = 1) -> list[tuple]:
    """(rid, frame, box, 11 classes) for every ``every``-th state-detector row of the split."""
    items = []
    for rid in SPLITS[split]:
        changes, init = psr_states(rid, LABELS), _initial(rid, tasks)
        for r in read_jsonl(ROOT / "output/assist" / rid / ASD[split])[::every]:
            items.append((rid, r["f"], r["box"], to_classes(state_at(changes, init, r["f"]))))
    return items


# ---------------------------------------------------------------------------------------------- GPU (server)
def _torch_parts(model_name: str):
    import cv2
    import torch
    from torch import nn

    from scripts.features.finetune_videomae_ar import Frames

    class Crops(torch.utils.data.Dataset):
        """(frames x size x size x 3 uint8, 11 classes): the square around the box, the same square on every
        frame of the clip, with a small random shift / zoom when training."""

        def __init__(self, items, n_frames: int, size: int, train: bool):
            self.items, self.n, self.size, self.train = items, n_frames, size, train

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            rid, f, box, classes = self.items[i]
            video = Frames(rid, CACHE)
            f = min(f, len(video) - 1)
            frames = video.get([max(0, f - STRIDE * (self.n - 1 - k)) for k in range(self.n)])
            x1, y1, x2, y2 = square(box)
            if self.train:
                side = x2 - x1
                new = int(side * np.random.uniform(0.9, 1.0))
                x1 = int(np.clip(x1 + np.random.randint(-side // 20, side // 20 + 1) + (side - new) // 2, 0, 960 - new))
                y1 = int(np.clip(y1 + np.random.randint(-side // 20, side // 20 + 1) + (side - new) // 2, 0, 540 - new))
                x2, y2 = x1 + new, y1 + new
            clip = np.stack([cv2.resize(fr[y1:y2, x1:x2], (self.size, self.size), interpolation=cv2.INTER_AREA)
                             for fr in frames])
            return torch.from_numpy(clip), torch.tensor(classes)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class Dino(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import AutoModel

            from monitor.settings import DINO_V2_DIR
            self.enc = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True)
            self.head = nn.Linear(2 * self.enc.config.hidden_size, PARTS * 3)

        def forward(self, clips):
            x = clips[:, -1].permute(0, 3, 1, 2).float() / 255.0
            h = self.enc(pixel_values=(x - mean.to(x.device)) / std.to(x.device)).last_hidden_state
            return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)).view(-1, PARTS, 3)

    class Vjepa(nn.Module):
        def __init__(self, layers: int, size: int):
            super().__init__()
            from scripts.features.vjepa2_finetune import Finetuned
            self.net = Finetuned(layers, size, PARTS * 3)

        def forward(self, clips):
            return self.net(clips).view(-1, PARTS, 3)

    return torch, nn, Crops, (Dino if model_name == "dino" else Vjepa)


def train(args) -> None:
    torch, nn, Crops, Model = _torch_parts(args.model)
    from assist.engine.graph import load_tasks
    tasks = load_tasks(load_config()["graph_path"])
    torch.manual_seed(0)
    np.random.seed(0)
    n_frames, size = (1, 336) if args.model == "dino" else (CLIP, 256)
    model = (Model() if args.model == "dino" else Model(args.layers, size)).cuda()
    tr = train_items(tasks)
    wrong = np.array([2 in it[3] for it in tr])
    weights = np.where(wrong, args.wrong_oversample, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(weights, args.per_epoch, replacement=True)
    loader = torch.utils.data.DataLoader(Crops(tr, n_frames, size, True), batch_size=args.batch, sampler=sampler,
                                         num_workers=args.workers, drop_last=True, persistent_workers=True)
    val = eval_items("val", tasks, every=4)
    val_y = np.array([it[3] for it in val])
    enc = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n and "probe" not in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and ("head" in n or "probe" in n)]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.lr * 20}], weight_decay=0.05)
    steps = args.epochs * (len(loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr, args.lr * 20], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    class_w = torch.tensor([1.0, 1.0, args.wrong_weight]).cuda()
    print(json.dumps({"model": args.model, "train_items": len(tr), "with_wrong_part": int(wrong.sum()),
                      "val_items": len(val)}), flush=True)
    log, best = [], {"exact": -1.0}
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        for i, (clips, y) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(clips.cuda())
                loss = nn.functional.cross_entropy(logits.reshape(-1, 3).float(), y.cuda().reshape(-1),
                                                   weight=class_w, label_smoothing=0.05) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        p = predict(model, val, n_frames, size, args, Crops, torch)
        exact = float((p.argmax(2) == val_y).all(1).mean() * 100)
        log.append({"epoch": epoch + 1, "loss": round(loss.item() * args.accum, 4), "val_exact_state": round(exact, 2),
                    "val_part_acc": round(float((p.argmax(2) == val_y).mean() * 100), 2),
                    "minutes": round((time.time() - start) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if exact > best["exact"]:
            best = {"epoch": epoch + 1, "exact": exact,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best.pop("state"))
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ("val", "test"):
        items = eval_items(split, tasks)
        p = predict(model, items, n_frames, size, args, Crops, torch).astype(np.float16)
        rids = np.array([it[0] for it in items])
        for rid in SPLITS[split]:
            arrays[f"{split}/{rid}"] = p[rids == rid]
    np.savez(OUT / f"probs_{args.model}.npz", **arrays)
    write_json(OUT / f"train_{args.model}.json", {"model": args.model, "chosen_epoch": best["epoch"],
                                                  "args": vars(args), "log": log})


def predict(model, items, n_frames, size, args, Crops, torch) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(Crops(items, n_frames, size, False), batch_size=args.batch * 2,
                                         num_workers=args.workers)
    out = []
    with torch.inference_mode():
        for clips, _ in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.softmax(model(clips.cuda()).float(), 2).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, PARTS, 3), np.float32)


# ---------------------------------------------------------------------------------------------- CPU (laptop)
def probs(split: str, model: str) -> dict[str, np.ndarray]:
    """rid -> (rows, 11, 3); ``both`` averages the two models."""
    if model == "both":
        a, b = probs(split, "dino"), probs(split, "vjepa")
        return {rid: (a[rid] + b[rid]) / 2 for rid in a}
    z = np.load(OUT / f"probs_{model}.npz")
    return {rid: z[f"{split}/{rid}"].astype(np.float32) for rid in SPLITS[split]}


def write_source(split: str, p: dict, thr: float) -> None:
    for rid in SPLITS[split]:
        rows = read_jsonl(ROOT / "output/assist" / rid / ASD[split])
        out = []
        for r, q in zip(rows, p[rid], strict=True):
            state, conf = to_state(q, thr)
            out.append({"f": r["f"], "t": r["t"], "state": state, "conf": round(conf, 3), "box": r["box"]})
        write_jsonl(ROOT / "output/assist" / rid / "asd_part.jsonl", out)


def evaluate_run(split: str, state: str) -> dict:
    """Engine (predicted actions + ``state``) on the split: mistakes and step recognition."""
    from assist.engine.graph import load_tasks
    from assist.evaluation.eval import evaluate
    from assist.evaluation.psr_eval import Data, engine_steps, score
    from assist.run_engine import run

    cfg = load_config()
    tasks = load_tasks(cfg["graph_path"])
    results = {}
    for rid in SPLITS[split]:
        results[rid] = run(rid, "pred", state, cfg, tasks)
        write_json(cfg["paths"]["out"] / rid / f"engine_pred_{state}.json", results[rid])
    r = evaluate(f"pred_{state}", SPLITS[split], cfg, tasks)["rows"]
    data = Data(cfg, "ours" if split == "val" else "release")
    psr = score(SPLITS[split], data, lambda rid, v: engine_steps(results[rid], cfg["components"], data.proc_info,
                                                                 data.graph[v["task"]]))["summary"]["all"]
    return {"found": r["all"]["tp"], "mistakes": r["all"]["gt"], "alerts": r["all"]["alerts"],
            "install_found": r["install"]["tp"], "install": r["install"]["gt"],
            "false_alarms_per_h": round(r["all"]["false_alarms_per_h"], 1),
            "psr_f1": round(psr["f1"], 3), "psr_pos": round(psr["pos"], 3), "psr_delay_s": round(psr["delay_s"], 1)}


def select() -> None:
    base = evaluate_run("val", STATE["val"])
    print(json.dumps({"baseline": base}), flush=True)
    grid = []
    models = [m for m in ("dino", "vjepa") if (OUT / f"probs_{m}.npz").is_file()]
    models += ["both"] if len(models) == 2 else []
    for model, thr in itertools.product(models, THRESHOLDS):
        write_source("val", probs("val", model), thr)
        grid.append({"model": model, "thr": thr, **evaluate_run("val", "asdpart")})
        print(json.dumps(grid[-1]), flush=True)
    passed = [g for g in grid if g["install_found"] > base["install_found"]
              and g["false_alarms_per_h"] <= base["false_alarms_per_h"] and g["psr_f1"] >= base["psr_f1"]]
    best = max(passed, key=lambda g: (g["found"], g["psr_f1"] + g["psr_pos"], -g["false_alarms_per_h"])) \
        if passed else None
    report = {"rule": "val: more wrongly mounted parts found than our detector alone, false alarms / h and "
                      "step-recognition F1 not worse", "baseline_val": base, "passed": best is not None,
              "chosen": best, "grid": grid}
    write_json(OUT / "select_val.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))


def test() -> None:
    report = json.loads((OUT / "select_val.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise SystemExit("val did not pass the stop rule: test is not scored")
    c = report["chosen"]
    write_source("test", probs("test", c["model"]), c["thr"])
    result = {"chosen_on_val": c, "baseline_test": evaluate_run("test", STATE["test"]),
              "part_state_test": evaluate_run("test", "asdpart")}
    write_json(OUT / "test.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", choices=("dino", "vjepa"), default="dino")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--per-epoch", type=int, default=4000, help="Training crops drawn per epoch.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone; the new head gets 20x.")
    parser.add_argument("--layers", type=int, default=12, help="V-JEPA 2: fine-tuned last layers (of 24).")
    parser.add_argument("--wrong-oversample", type=float, default=4.0, help="Sampling weight of crops with a wrong part.")
    parser.add_argument("--wrong-weight", type=float, default=3.0, help="Loss weight of the 'mounted wrong' class.")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.train:
        train(args)
    if args.select:
        select()
    if args.test:
        test()


if __name__ == "__main__":
    main()
