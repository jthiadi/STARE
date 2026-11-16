"""Download the pretrained backbones into models/.

    python -m scripts.data.download_models          # DINOv2-small + VideoMAE-SSv2 (needed for predictions)
    python -m scripts.data.download_models --sam    # + SAM 3.1 (gated: request access to facebook/sam3.1 on
                                               #   Hugging Face, then `huggingface-cli login`)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from huggingface_hub import hf_hub_download, snapshot_download

from monitor.settings import DINO_V2_DIR, DINO_V2_ID, DINO_V2_REVISION, ROOT
from scripts.features.extract_videomae_features import MODEL_DIR, download

SAM_DIR = ROOT / "models/sam3.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam", action="store_true", help="Also fetch SAM 3.1 and export its fp16 detector.")
    args = parser.parse_args()

    if not (DINO_V2_DIR / "model.safetensors").is_file():
        snapshot_download(DINO_V2_ID, revision=DINO_V2_REVISION, local_dir=DINO_V2_DIR,
                          allow_patterns=["*.json", "*.safetensors"])
    print(f"DINOv2-small:  {DINO_V2_DIR}", flush=True)
    download()
    print(f"VideoMAE-SSv2: {MODEL_DIR}", flush=True)
    if args.sam:
        if not (SAM_DIR / "sam3.1_multiplex.pt").is_file():
            hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt", local_dir=SAM_DIR)
        if not (SAM_DIR / "sam3.1_detector_fp16.safetensors").is_file():
            subprocess.run([sys.executable, "-m", "scripts.kit_sop.export_sam31_detector"], check=True, cwd=ROOT)
        print(f"SAM 3.1:       {SAM_DIR}", flush=True)


if __name__ == "__main__":
    main()
