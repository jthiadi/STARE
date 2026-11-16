"""Frame-wise IndustReal actions (verb_noun, 10 fps) for any egocentric assembly video.

    python -m scripts.predict_actions --video my_video.mp4 [--overlay] [--sam]

Pipeline: video (resampled to 10 fps) -> DINOv2-small (3 views + hi-res hands) + VideoMAE-SSv2 motion
+ the two fine-tuned DINOv2 hand-crop backbones + the two IndustReal-fine-tuned VideoMAE backbones
-> verb TCN ensemble x noun TCN ensemble ->
Viterbi decoding over the 74 official actions + background (settings in release/decoder.json).

Writes to output/predict/<video name>/:
    actions.csv      one row per frame: time, action, verb, noun, confidence
    segments.csv     one row per predicted action segment
    predictions.npz  pred / conf per frame plus the verb and noun probabilities
    <video>_actions[_sam].mp4  with --overlay / --sam
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from monitor.settings import DINO_V2_DIR, ROOT

FPS = 10.0


def _log(**kw) -> None:
    print(json.dumps(kw), flush=True)


def ensure_10fps(video: Path, work: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(fps - FPS) < 0.5:
        return video
    import imageio_ffmpeg

    target = work / f"{video.stem}_10fps.mp4"
    if not target.is_file():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", str(target)], check=True)
    _log(step="resampled to 10 fps", source_fps=round(fps, 2), video=str(target))
    return target


def dino_features(video: Path, feat_dirs: dict[str, Path], name: str) -> None:
    from transformers import AutoModel

    from scripts.features.extract_ar_frame_features import HIRES_VIEWS, VIEWS, extract

    jobs = [("ar_frame_features", VIEWS, False), ("ar_frame_features_hires", HIRES_VIEWS, True)]
    todo = [j for j in jobs if not (feat_dirs[j[0]] / f"{name}.npz").is_file()]
    if not todo:
        return
    model = AutoModel.from_pretrained(DINO_V2_DIR, local_files_only=True, use_safetensors=True)
    model = model.to("cuda").half().eval()
    for folder, views, grid in todo:
        start = time.time()
        feats = extract(model, video, 16, views=views, grid=grid)
        np.savez(feat_dirs[folder] / f"{name}.npz", feats=feats)
        _log(step=folder, frames=int(len(feats)), seconds=round(time.time() - start, 1))
    del model
    torch.cuda.empty_cache()


def videomae_features(video: Path, out: Path) -> None:
    if out.is_file():
        return
    from scripts.features.extract_videomae_features import extract, load_model

    start = time.time()
    model = load_model()
    feats, logits = extract(model, video, stride=4, batch=4)
    np.savez(out, feats=feats[:, None, :], video_logits=logits)
    _log(step="videomae_ssv2_features", frames=int(len(feats)), seconds=round(time.time() - start, 1))
    del model
    torch.cuda.empty_cache()


def finetuned_features(video: Path, release: Path, out: Path, fold: int) -> None:
    """Same input as training: hands crop -> 405x270 JPEG (q92) -> 252x378, then the fine-tuned backbone."""
    if out.is_file():
        return
    from scripts.features.dump_ar_crops import CROP_SIZE, HANDS_BOX
    from scripts.features.finetune_dino_ar import INPUT_HW, VERBS, Net, _to_tensor, embed_batch

    start = time.time()
    meta = json.loads((release / "finetune/meta.json").read_text(encoding="utf-8"))
    model = Net(len(meta["objects"]), len(VERBS)).to("cuda")
    model.load_state_dict(torch.load(release / f"finetune/fold{fold}.pt", map_location="cuda", weights_only=True))
    model.eval()
    cap = cv2.VideoCapture(str(video))
    parts, batch = [], []
    while True:
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            crop = cv2.resize(frame[int(HANDS_BOX[1] * h):int(HANDS_BOX[3] * h),
                                    int(HANDS_BOX[0] * w):int(HANDS_BOX[2] * w)], CROP_SIZE,
                              interpolation=cv2.INTER_AREA)
            crop = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], cv2.IMREAD_COLOR)
            img = cv2.resize(crop, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
            batch.append(_to_tensor(img))
        if batch and (len(batch) == 32 or not ok):
            parts.append(embed_batch(model, torch.from_numpy(np.stack(batch)).to("cuda")))
            batch = []
        if not ok:
            break
    cap.release()
    np.savez(out, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    _log(step=f"finetuned_fold{fold}", frames=sum(len(p["feats"]) for p in parts),
         seconds=round(time.time() - start, 1))
    del model
    torch.cuda.empty_cache()


def vmae_finetuned_features(video: Path, release: Path, work: Path, out: Path, fold: int) -> None:
    """Fine-tuned VideoMAE backbone ``fold`` on the same cached 224 x 224 JPEG frames used in training."""
    if out.is_file():
        return
    from scripts.features.finetune_videomae_ar import build_cache, extract, load_model
    from scripts.data.tas_labels import load_mapping

    start = time.time()
    work.mkdir(parents=True, exist_ok=True)
    build_cache(out.stem, video, work)
    model = load_model(len(load_mapping()), release / f"vmae_finetune/fold{fold}.pt")
    feats, logits = extract(model, out.stem, 4, 8, work)
    np.savez(out, feats=feats[:, None, :], video_logits=logits)
    _log(step=f"vmae_finetuned_fold{fold}", frames=int(len(feats)), seconds=round(time.time() - start, 1))
    del model
    torch.cuda.empty_cache()


def needs_vmae_finetuned(release: Path) -> bool:
    return any(torch.load(release / f"{t}_model.pt", map_location="cpu", weights_only=True)["features"]
               .get("crossfit_sets", 1) > 1 for t in ("verb", "noun"))


@torch.inference_mode()
def tcn_probs(bundle_path: Path, name: str, feat_dirs: dict[str, Path]) -> np.ndarray:
    """Mean class probabilities over every TCN in the bundle and both fine-tuned feature variants
    (variant k: the fold-k fine-tuned DINOv2, plus the fold-k fine-tuned VideoMAE when the bundle uses it)."""
    from scripts.segmentation.train_ar_object_tcn import TCN, load_features, predict_proba

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    spec = bundle["features"]
    sets = spec.get("crossfit_sets", 1)
    variants = [[feat_dirs[f"finetuned_fold{k}"]] + ([feat_dirs[f"vmae_finetuned_fold{k}"]] if sets > 1 else [])
                for k in (0, 1)]
    model = TCN(bundle["dim"], len(bundle["classes"]), hidden=bundle["hidden"], layers=bundle["layers"],
                stages=bundle["stages"], aux=bundle["aux"]).to("cuda")
    total, count = None, 0
    for ft in variants if spec["finetuned"] else [[]]:
        extra = [feat_dirs[e] for e in spec["extra"]] + ft
        feats = load_features(name, feat_dirs[spec["base"]], spec["progress"], extra)
        assert feats.shape[1] == bundle["dim"], (feats.shape, bundle["dim"])
        for state in bundle["models"]:
            model.load_state_dict(state)
            prob = predict_proba(model, feats, "cuda")
            total = prob if total is None else total[:len(prob)] + prob[:len(total)]
            count += 1
    return total / count


def write_outputs(out: Path, actions: list[str], pred: np.ndarray, conf: np.ndarray) -> None:
    from scripts.segmentation.evaluate_ar_object_segments import gt_noun_from_action
    from scripts.segmentation.tas_metrics import segments

    with (out / "actions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "action", "verb", "noun", "confidence"])
        for i, (p, c) in enumerate(zip(pred, conf)):
            a = actions[int(p)]
            bg = int(p) == 0
            writer.writerow([i, round(i / FPS, 1), a, "" if bg else a.split("_", 1)[0],
                             "" if bg else gt_noun_from_action(a), round(float(c), 3)])
    with (out / "segments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_s", "end_s", "action", "mean_confidence"])
        for label, s, e in segments(pred):
            if label:
                writer.writerow([round(s / FPS, 1), round(e / FPS, 1), actions[label],
                                 round(float(conf[s:e].mean()), 3)])


def predict(video_path: Path, release: Path, out: Path) -> dict:
    """Features -> verb/noun TCNs -> decoded actions for one video; writes predictions.npz and the CSVs to ``out``.

    Returns the 10 fps video path, the action list, the joint (T, actions) scores and the decoded pred/conf.
    """
    name = video_path.stem
    feat_dirs = {d: out / "features" / d for d in
                 ("ar_frame_features", "ar_frame_features_hires", "videomae_ssv2_features",
                  "finetuned_fold0", "finetuned_fold1", "vmae_finetuned_fold0", "vmae_finetuned_fold1")}
    for d in feat_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    video = ensure_10fps(video_path, out)

    dino_features(video, feat_dirs, name)
    videomae_features(video, feat_dirs["videomae_ssv2_features"] / f"{name}.npz")
    for fold in (0, 1):
        finetuned_features(video, release, feat_dirs[f"finetuned_fold{fold}"] / f"{name}.npz", fold)
    if needs_vmae_finetuned(release):
        for fold in (0, 1):
            vmae_finetuned_features(video, release, out / "features/vmae_frames",
                                    feat_dirs[f"vmae_finetuned_fold{fold}"] / f"{name}.npz", fold)

    from scripts.segmentation.combine_verb_noun_actions import decode, joint_scores

    decoder = json.loads((release / "decoder.json").read_text(encoding="utf-8"))
    pv = tcn_probs(release / "verb_model.pt", name, feat_dirs)
    pn = tcn_probs(release / "noun_model.pt", name, feat_dirs)
    n = min(len(pv), len(pn))
    actions = decoder["actions"]
    scores = joint_scores(pv[:n], pn[:n], decoder["verbs"], decoder["nouns"], actions)
    pred, conf = decode(scores, decoder["switch"], decoder["bias"], np.array(decoder["transitions"]),
                        decoder["trans_weight"])
    np.savez(out / "predictions.npz", actions=np.array(actions), pred=pred, conf=conf,
             verb_probs=pv[:n], noun_probs=pn[:n], verbs=np.array(decoder["verbs"]),
             nouns=np.array(decoder["nouns"]))
    write_outputs(out, actions, pred, conf)
    _log(step="predictions", frames=int(n), segments=str(out / "segments.csv"))
    return {"video": video, "actions": actions, "scores": scores, "pred": pred, "conf": conf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=ROOT / "release",
                        help="Folder with verb_model.pt, noun_model.pt, decoder.json, finetune/ (GitHub release).")
    parser.add_argument("--output", type=Path, default=ROOT / "output/predict")
    parser.add_argument("--overlay", action="store_true", help="Render an overlay video of the predictions.")
    parser.add_argument("--sam", action="store_true",
                        help="Also run SAM 3.1 (slow: ~10 s per frame on a 6 GB GPU) and draw its masks.")
    args = parser.parse_args()
    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    assert torch.cuda.is_available(), "a CUDA GPU is required"

    name = args.video.stem
    out = args.output / name
    result = predict(args.video, args.release, out)
    video, actions, pred, conf = result["video"], result["actions"], result["pred"], result["conf"]

    sam_dir = None
    if args.sam:
        sam_dir = out / "sam31"
        subprocess.run([sys.executable, "-u", "-m", "scripts.kit_sop.run_kit_card_detect", "--video", str(video),
                        "--output", str(sam_dir), "--resume", "--batch", "1",
                        "--checkpoint", str(ROOT / "models/sam3.1/sam3.1_detector_fp16.safetensors")],
                       check=True, cwd=ROOT, env=dict(os.environ, SAM3_CUDNN_BENCHMARK="0"))
    if args.overlay or args.sam:
        from scripts.segmentation.render_action_overlay import render

        target = out / f"{name}_actions{'_sam' if sam_dir else ''}.mp4"
        render(video, [actions[int(p)] for p in pred], conf, target, "Verb+Noun TCN", sam_dir=sam_dir)
        _log(step="overlay", video=str(target))


if __name__ == "__main__":
    main()
