"""Run the full v13 inference pipeline from the video frames on a split, then evaluate it.

    python run_pipeline.py --split test --ar-dir <dir with <video>/ar.jsonl>

Steps (inputs only until the evaluation step; no labels are read):
  1. extract_crops.py     : DINOv2-small localizer finds the assembly in every 2nd frame -> cache/crops/<split>
  2. extract_crops_hr.py  : the same boxes cropped at 336 px                              -> cache/crops_hr/<split>
  3. predict_verifier.py  : both frozen DINOv2-base verifiers                             -> cache/<tag>_<split>
  4. eval_v13.py          : v13 causal state filter + mistake alerts + official evaluator, predicted AR and GT AR (oracle)
Verifier weights are expected in weights/ (not in git; see README).
Steps already done (output exists) are skipped.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(args, **env):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True, env={**os.environ, **env})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ar-dir", help="folder with <video>/ar.jsonl (predicted AR); omit to run only the GT-AR oracle")
    a = ap.parse_args()
    run(["extract_crops.py", a.split])
    run(["extract_crops_hr.py", a.split, "336"])
    w = HERE / "weights"
    # the env variables reproduce the training configuration of each verifier (backbone, resolution, tag)
    run(["predict_verifier.py", a.split, str(w / "verif_base_r336.pt")], VBACK="base", VRES="336")
    run(["predict_verifier.py", a.split, str(w / "verif_base_r336_e6u6.pt")], VBACK="base", VRES="336",
        VEPOCHS="6", VUNFREEZE="6")
    if a.ar_dir:
        run(["eval_v13.py", "--split", a.split, "--ar", "predicted", "--ar-dir", a.ar_dir])
    run(["eval_v13.py", "--split", a.split, "--ar", "gt"])


if __name__ == "__main__":
    main()
