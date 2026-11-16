"""Official IndustReal protocol for the DINOv2 + TCN object (noun) model.

Train: the 36 official train recordings (DINOv2 fine-tune + TCN).
Val:   the 16 official val recordings (choose ensemble weights + decoding).
Test:  the 32 official test recordings (scored once).
Official 18-noun vocabulary. Resumable: finished steps are skipped.
"""
from __future__ import annotations

import json
import sys

from monitor.settings import ROOT
from scripts.segmentation.run_subject_holdout_eval import _run

TRAIN = ["01_assy_0_1", "01_assy_1_1", "01_main_0_1", "02_assy_0_1", "02_assy_1_2", "02_main_0_1",
         "04_assy_0_1", "04_assy_2_1", "04_main_0_1", "06_assy_0_1", "06_assy_1_4", "06_main_0_1",
         "07_assy_0_1", "07_assy_2_3", "07_main_0_1", "11_assy_0_1", "11_assy_3_3", "11_main_0_1",
         "15_assy_0_1", "15_main_3_1", "15_main_3_2", "16_assy_0_1", "16_main_0_1", "16_main_3_3",
         "21_assy_0_1", "21_main_0_1", "22_assy_0_1", "22_assy_2_3", "22_main_0_1", "25_assy_0_1",
         "25_assy_2_1", "25_main_0_1", "27_assy_0_1", "27_main_0_1", "27_main_1_3", "27_main_3_1"]
VAL = ["05_assy_0_1", "05_assy_2_2", "05_main_0_1", "14_assy_0_1", "14_main_0_1", "14_main_2_2",
       "14_main_2_3", "20_assy_0_1", "20_assy_3_6", "20_main_0_1", "24_assy_0_1", "24_assy_2_4",
       "24_main_0_1", "26_assy_0_1", "26_assy_1_5", "26_main_0_1"]
TEST = ["03_assy_0_1", "03_assy_1_3", "03_main_0_1", "08_assy_0_1", "08_assy_2_4", "08_main_0_1",
        "09_assy_0_1", "09_assy_3_1", "09_main_0_1", "10_assy_0_1", "10_assy_3_2", "10_main_0_1",
        "10_main_3_4", "12_assy_0_1", "12_assy_3_4", "12_main_0_1", "13_assy_0_1", "13_main_0_1",
        "13_main_1_3", "13_main_2_1", "17_assy_0_1", "17_assy_1_5", "17_main_0_1", "18_assy_0_1",
        "18_assy_2_5", "18_main_0_1", "19_assy_0_1", "19_assy_3_5", "19_main_0_1", "23_assy_0_1",
        "23_assy_1_2", "23_main_0_1"]
ARCHIVES = {"train": ["train_p1.zip", "train_p2.zip", "train_p3.zip", "train_p4.zip"],
            "val": ["val_p1.zip", "val_p2.zip"],
            "test": ["test_p1.zip", "test_p2.zip", "test_p3.zip"]}
RUNS = {"tcn_v7": ["--verb-aux"], "tcn_v8": []}


def _subject(rid: str) -> str:
    return rid.split("_", 1)[0]


def main() -> None:
    py = sys.executable
    out = ROOT / "output/official_split_eval"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    held = VAL + TEST
    assert not {_subject(r) for r in TRAIN} & {_subject(r) for r in held}

    videos = ROOT / "data/generated/industreal/videos"
    for split, rids in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        for rid in rids:
            if not (videos / f"{rid}.mp4").is_file():
                _run([py, "-u", "-m", "scripts.data.download_industreal", "--recording", rid], log)
            if not (ROOT / "data/evaluation/official" / rid / "AR_labels.csv").is_file():
                _run([py, "-u", "-m", "scripts.data.download_eval_labels", "--recording", rid,
                      "--archive", *ARCHIVES[split]], log)
    print(json.dumps({"step": "download", "done": True}), flush=True)

    _run([py, "-u", "-m", "scripts.features.dump_ar_crops"], log)
    _run([py, "-u", "-m", "scripts.features.extract_ar_frame_features"], log)
    _run([py, "-u", "-m", "scripts.features.extract_ar_frame_features", "--hires",
          "--output", str(ROOT / "output/ar_frame_features_hires")], log)
    print(json.dumps({"step": "base features", "done": True}), flush=True)

    ft = out / "finetune"
    pool = sorted({_subject(r) for r in TRAIN})
    for fold in (0, 1):
        if (ft / f"fold{fold}.extracted").is_file():
            continue
        resume = ["--extract-only"] if (ft / f"fold{fold}.trained").is_file() else []
        _run([py, "-u", "-m", "scripts.features.finetune_dino_ar", "--fold", str(fold),
              "--test-recordings", *held, "--pool-subjects", *pool, "--subject-split",
              "--extract-needed", "--full-nouns", "--batch", "32", "--lr", "4e-5", "--workers", "0",
              "--output", str(ft), *resume], log)
        print(json.dumps({"step": f"finetune fold {fold}", "done": True}), flush=True)

    done = []
    for name, extra in RUNS.items():
        if not (out / name / "ar_object_paper_metrics.json").is_file():
            _run([py, "-u", "-m", "scripts.segmentation.train_ar_object_tcn",
                  "--train-recordings", *TRAIN, "--test-recordings", *held,
                  "--extra-features", str(ROOT / "output/ar_frame_features_hires"),
                  "--crossfit", str(ft / "feats_fold0"), str(ft / "feats_fold1"),
                  "--crossfit-meta", str(ft / "meta.json"),
                  "--progress", "--cosine", "--warp", "--epochs", "80", "--seeds", "3",
                  "--hidden", "192", "--layers", "10", "--stages", "4", "--folds", "6",
                  "--subject-folds", "--host-features", "--full-nouns",
                  *extra, "--output", str(out / name)], log)
        done.append(name)
        _run([py, "-u", "-m", "scripts.segmentation.evaluate_unseen_split", *[str(out / n) for n in done],
              "--val", *VAL, "--test", *TEST, "--full-nouns",
              "--output", str(out / f"result_{'_'.join(done)}")], log)
        print(json.dumps({"step": name, "done": True}), flush=True)


if __name__ == "__main__":
    main()
