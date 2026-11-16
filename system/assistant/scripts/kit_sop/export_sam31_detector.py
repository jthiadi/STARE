"""Export the image-detector weights of the SAM 3.1 multiplex checkpoint as fp16 safetensors.

The full .pt (3.3 GB, detector + tracker) must be read whole into RAM on every load; the
detector-only fp16 file is about half the size and loads through the validated safetensors path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from monitor.settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/sam3.1/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models/sam3.1/sam3.1_detector_fp16.safetensors")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    state = {}
    for key in [k for k in ckpt if "detector" in k]:
        value = ckpt.pop(key)
        state[key.replace("detector.", "")] = (value.half() if value.is_floating_point() else value).contiguous()
    del ckpt
    save_file(state, str(args.output))
    print(args.output, f"{args.output.stat().st_size / 2**30:.2f} GB", len(state), "tensors")


if __name__ == "__main__":
    main()
