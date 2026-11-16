"""Pack the files that are too large for the Git repository into two archives, for upload to a model / dataset host.

  stare_outputs.tar  saved model outputs: everything metrics/reproduce_all.py reads (CPU reproduction of Tables 1-8)
  stare_weights.tar  trained weights: what the live system (system/assistant/live) and the MECCANO runs load on the GPU

Each archive keeps STARE-relative paths and contains MANIFEST.sha256. Unpack with tools/unpack_artifacts.py.
Absolute paths of the machines the files were made on are cut back to repository-relative ones inside the archive (the
files on disk are not changed); if any private string is left afterwards, nothing is written.
Not packed: the datasets (IndustReal, MECCANO), the public backbones (scripts/data/download_models.py fetches them) and
third-party code and checkpoints (tools/fetch_third_party.sh).

    python tools/pack_artifacts.py                 # list what would be packed, with sizes (nothing written)
    python tools/pack_artifacts.py --write <dir>   # write <dir>/stare_outputs.tar and <dir>/stare_weights.tar
    PACK_PRIVATE_TERMS=name1,name2 python tools/pack_artifacts.py --write <dir>   # also refuse files containing these
"""
import argparse
import hashlib
import io
import os
import re
import tarfile
import zipfile
from pathlib import Path

STARE = Path(__file__).resolve().parents[1]
A = "system/assistant"
OUTPUTS = [
    f"{A}/output/assist/**/*",                                   # per-recording live outputs (actions, final labels, hands)
    f"{A}/output/mistakes/**/*",
    f"{A}/output/official_split_eval/vjepa2/clip_probs_*.npz",   # saved per-segment V-JEPA 2 outputs (Tables 1-2)
    f"{A}/output/official_split_eval/vjepa2/*.json",
    f"{A}/output/meccano_action_end/*.json",                    # MECCANO action recognition (Table 3)
    f"{A}/output/meccano_action_end/vmae_probs.npz",
    f"{A}/output/meccano_action_end/vjepa/clip_probs_*.npz",
    f"{A}/output/meccano_action_end/vjepa/*.json",
    f"{A}/output/meccano_action_end/tcn/**/*",
    f"{A}/output/meccano_action_end/frames/*.idx.npy",
    "metrics/e2e/work/val_ensemble/**/*",                        # validation ensemble labels (Tables 4, 6, 7)
    "metrics/e2e/work/latency/**/*",
    "metrics/meccano/work/ar/*.npz",                             # MECCANO step recognition (Table 5)
    "metrics/meccano/work/vprobs/**/*",
    "metrics/meccano/work/steps/**/*",
    "third_party/PROVIA/runs/industreal/**/*_candidates.npz",    # our PROVIA / baseline runs: alarm candidates (Table 8)
    "system/STORM-PSR-IndustReal/**/*",                          # released IndustReal ASD outputs + procedure info
]
WEIGHTS = [
    "system/weights/*.pt",                                       # two part verifiers + the official IndustReal ASD
    f"{A}/release_vjepa/vmae_full.pt",
    f"{A}/release_vjepa/settings.json",
    f"{A}/release_vjepa/action_order_counts.npz",
    f"{A}/release_vjepa_prefix/*",
    f"{A}/output/official_split_eval/vjepa2/vjft_area_k12.pt",   # final-label V-JEPA 2 members + action-beginning run
    f"{A}/output/official_split_eval/vjepa2/vjft_hand_k12.pt",
    f"{A}/output/official_split_eval/vjepa2/vjft_hand_k12_s1.pt",
    f"{A}/output/official_split_eval/vjepa2/vjft_hand_k12_p10.pt",
    f"{A}/output/official_split_eval/tcn_verb_causal/**/*",      # causal verb / noun TCN ensembles
    f"{A}/output/official_split_eval/tcn_noun_causal/**/*",
    f"{A}/output/official_split_eval/finetune/**/*",             # cross-fitted DINOv2 hand-crop backbones
    f"{A}/output/official_split_eval/vmae_finetune/fold*/*",    # cross-fitted VideoMAE backbones
    f"{A}/output/meccano_action_end/vjepa/*.pt",                 # MECCANO V-JEPA 2 runs
    "metrics/meccano/work/ar/tcn.pt",                            # MECCANO TCN and part verifier
    "metrics/meccano/work/vision/verifier.pt",
    "metrics/meccano/work/vision/localizer_protos.npz",
]
EXCLUDE = [f"{A}/output/assist/cad/part_geometries/*"]                # IndustReal CAD files: from the dataset page
TEXT = {".json", ".jsonl", ".txt", ".csv", ".md", ".yaml"}
SEP = rb"(?:\\\\|\\|/)+"                                          # JSON-escaped backslashes, backslash or slash
ABS = re.compile(rb"(?:[A-Za-z]:" + SEP + rb"|/)[^\s\"']*?(?:SpecialImplementation|STARE|v14_integration|industreal_extra)" + SEP)
# Home folders always count as private; add your own user names / server folders with PACK_PRIVATE_TERMS=a,b,c
EXTRA = [re.escape(t.strip()).encode() for t in os.environ.get("PACK_PRIVATE_TERMS", "").split(",") if t.strip()]
PRIVATE = re.compile(b"|".join([rb"[A-Za-z]:" + SEP + rb"Users", rb"/home/[a-z]", *EXTRA]), re.I)


def files(patterns: list[str]) -> list[Path]:
    out = set()
    for pat in patterns:
        out.update(p for p in STARE.glob(pat) if p.is_file() and "__pycache__" not in p.parts)
    out -= {p for pat in EXCLUDE for p in STARE.glob(pat)}
    return sorted(out)


def content(p: Path) -> bytes:
    data = p.read_bytes()
    return ABS.sub(b"", data) if p.suffix in TEXT else data


def private(p: Path, data: bytes) -> bool:
    if p.suffix in (".npz", ".pt") and zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = b"".join(z.read(n) for n in z.namelist())
    return bool(PRIVATE.search(data))


def check(fs: list[Path]) -> list[str]:
    """Files that would still carry a private string after the path clean-up."""
    return [p.relative_to(STARE).as_posix() for p in fs if private(p, content(p))]


def write(name: str, fs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    with tarfile.open(out_dir / name, "w") as tar:
        for p in fs:
            data, rel = content(p), p.relative_to(STARE).as_posix()
            info = tarfile.TarInfo(rel)
            info.size, info.mtime = len(data), int(p.stat().st_mtime)
            tar.addfile(info, io.BytesIO(data))
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}\n")
        manifest = "".join(lines).encode()
        info = tarfile.TarInfo("MANIFEST.sha256")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    print(f"wrote {out_dir / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", type=Path, help="folder to write the archives to (default: only list)")
    a = ap.parse_args()
    for name, pats in (("stare_outputs.tar", OUTPUTS), ("stare_weights.tar", WEIGHTS)):
        fs = files(pats)
        print(f"{name}: {len(fs)} files, {sum(p.stat().st_size for p in fs) / 1e9:.2f} GB")
        for pat in pats:
            if not any(True for _ in STARE.glob(pat)):
                print(f"  WARNING no file matches {pat}")
        left = check(fs)
        for rel in left:
            print(f"  PRIVATE STRING LEFT in {rel}")
        if a.write:
            if left:
                raise SystemExit(f"{name}: not written, {len(left)} file(s) still carry private strings")
            write(name, fs, a.write)


if __name__ == "__main__":
    main()
