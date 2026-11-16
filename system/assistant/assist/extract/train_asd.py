"""Our own assembly state detector (ASD) for IndustReal: YOLOv8-M trained on the real train videos.

The released IndustReal ASD predictions cover the test videos only, so when this was built nothing could be tuned
on val. (IndustReal's weights turned out to be published too; ``--predict --weights`` runs them.) This trains the same kind of model (YOLOv8-M from COCO, the official recipe) on the
36 train videos' box labels, with one change: on frames where the official step labels say a part is
mounted incorrectly (state -1), the box gets the ``error_state`` class. In the released labels only states
outside the 22-state list are ``error_state``; frames with a wrongly mounted part often carry a look-alike
correct state instead.

Steps (each resumable, run on the GPU server):
    python -m assist.extract.train_asd --build          # frames + YOLO labels for train / val
    python -m assist.extract.train_asd --train
    python -m assist.extract.train_asd --predict        # every frame of the val + test videos -> CSV
Predictions use the released CSV format (``clip,framenr,bb_class,bb_conf,bb_x,bb_y,bb_w,bb_h``, box centre +
size in pixels), so every reader of the released predictions works unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from assist.common import ROOT, load_config
from assist.industreal import SPLITS, psr_states

OUT = ROOT / "output/assist/asd_ours"
DATA = Path(os.environ.get("ASD_DATA", OUT / "yolo"))   # e.g. a RAM disk (/dev/shm/...) for fast I/O
ERROR_CLASS = 23                                 # index of "error_state" in the released class list


def frame_labels(rid: str, labels: Path) -> dict[int, tuple[int, list[float]]]:
    """frame -> (class index, [x, y, w, h] pixels) from OD_labels.json, with incorrect-install relabelling."""
    od = json.loads((labels / rid / "OD_labels.json").read_text(encoding="utf-8"))
    frames = {im["id"]: int(Path(im["file_name"]).stem) for im in od["images"]}
    changes = psr_states(rid, labels)
    out = {}
    for a in od["annotations"]:
        f = frames[a["image_id"]]
        state = [s for fr, s in changes if fr <= f]
        cls = a["category_id"]                   # indexes asd_categories directly (OD "categories" ids are off by one)
        if state and -1 in state[-1] and cls != 0:
            cls = ERROR_CLASS
        out[f] = (cls, a["bbox"])
    return out


def build_video(rid: str, split: str, cfg: dict) -> str:
    img_dir, lab_dir = DATA / "images" / split, DATA / "labels" / split
    done = img_dir / f".{rid}.done"
    if done.is_file():
        return rid
    wanted = frame_labels(rid, cfg["paths"]["labels"])
    cap = cv2.VideoCapture(str(cfg["paths"]["videos"] / f"{rid}.mp4"))
    w, h = int(cap.get(3)), int(cap.get(4))
    i = 0
    while wanted and i <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            cls, (x, y, bw, bh) = wanted[i]
            cv2.imwrite(str(img_dir / f"{rid}_{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lab_dir / f"{rid}_{i:06d}.txt").write_text(
                f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")
        i += 1
    cap.release()
    done.touch()
    print(json.dumps({"built": rid, "split": split, "frames": len(wanted)}), flush=True)
    return rid


def build(cfg: dict, workers: int) -> None:
    names = cfg["asd_categories"]
    jobs = []
    for split in ("train", "val"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs += [(rid, split) for rid in SPLITS[split]]
    with ProcessPoolExecutor(workers) as pool:       # one video per worker; decoding is the bottleneck
        list(pool.map(build_video, [j[0] for j in jobs], [j[1] for j in jobs], [cfg] * len(jobs)))
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(names)), encoding="utf-8")


def train(args) -> None:
    from ultralytics import YOLO
    devices = [int(d) for d in args.devices.split(",")]
    last = OUT / "runs/yolov8m_real/weights/last.pt"
    if args.resume and last.is_file():           # continue an interrupted run (same settings, e.g. after a reboot)
        YOLO(str(last)).train(resume=True, device=devices if len(devices) > 1 else devices[0],
                              batch=16 * len(devices), workers=6)
        return
    model = YOLO(str(ROOT.parent / "models/yolov8m.pt"))
    model.train(data=str(DATA / "data.yaml"), epochs=args.epochs, imgsz=640, batch=16 * len(devices),
                device=devices if len(devices) > 1 else devices[0], workers=6, cache=False,   # shared server
                project=str(OUT / "runs"), name="yolov8m_real", exist_ok=True, patience=8, seed=0, plots=False)


def predict(cfg: dict, args) -> None:
    from ultralytics import YOLO
    weights = args.weights or OUT / "runs/yolov8m_real/weights/best.pt"
    model = YOLO(str(weights))
    pred_dir = Path(args.predict_out) if args.predict_out else OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rid in args.videos or SPLITS["val"] + SPLITS["test"]:
        dest = pred_dir / f"{rid}_results_pred.csv"
        if dest.is_file():
            continue
        cap = cv2.VideoCapture(str(cfg["paths"]["videos"] / f"{rid}.mp4"))
        rows, batch, idx, i = [], [], [], 0
        while True:
            ok, frame = cap.read()
            if ok:
                batch.append(frame)
                idx.append(i)
                i += 1
            if batch and (len(batch) == 32 or not ok):
                for f, res in zip(idx, model.predict(batch, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False,
                                                     half=True, max_det=3)):
                    for (cx, cy, bw, bh), c, k in zip(res.boxes.xywh.tolist(), res.boxes.conf.tolist(),
                                                      res.boxes.cls.tolist()):
                        rows.append([rid, f, int(k), round(c, 4), round(cx, 1), round(cy, 1), round(bw, 1),
                                     round(bh, 1)])
                batch, idx = [], []
            if not ok:
                break
        cap.release()
        tmp = dest.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip", "framenr", "bb_class", "bb_conf", "bb_x", "bb_y", "bb_w", "bb_h"])
            w.writerows(rows)
        tmp.replace(dest)
        print(json.dumps({"predicted": rid, "frames": i, "detections": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--devices", default="0", help='GPU ids, e.g. "0,1".')
    parser.add_argument("--resume", action="store_true", help="Continue the interrupted training run.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--weights", help="--predict: use these weights (e.g. IndustReal's released detector).")
    parser.add_argument("--predict-out", help="--predict: folder for the CSVs (default: our detector's folder).")
    parser.add_argument("--videos", nargs="+", help="--predict: these recordings (default: val + test).")
    parser.add_argument("--imgsz", type=int, default=640, help="--predict: image size.")
    args = parser.parse_args()
    cfg = load_config()
    if args.build:
        build(cfg, args.workers)
    if args.train:
        train(args)
    if args.predict:
        predict(cfg, args)


if __name__ == "__main__":
    main()
