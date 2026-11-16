"""Unpack stare_outputs.tar / stare_weights.tar (made by tools/pack_artifacts.py) into this STARE folder and check every
file against the archive's MANIFEST.sha256.

    python tools/unpack_artifacts.py <archive.tar> [<archive.tar> ...]
"""
import hashlib
import sys
import tarfile
from pathlib import Path

STARE = Path(__file__).resolve().parents[1]


def main() -> None:
    for archive in map(Path, sys.argv[1:]):
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            for m in members:
                dest = (STARE / m.name).resolve()
                if STARE not in dest.parents:
                    raise SystemExit(f"{archive}: refusing to write outside STARE: {m.name}")
            manifest = tar.extractfile("MANIFEST.sha256").read().decode().splitlines()
            tar.extractall(STARE, members=[m for m in members if m.name != "MANIFEST.sha256"])
        bad = []
        for line in manifest:
            digest, rel = line.split("  ", 1)
            h = hashlib.sha256((STARE / rel).read_bytes()).hexdigest()
            if h != digest:
                bad.append(rel)
        print(f"{archive.name}: {len(manifest)} files, {'all checksums OK' if not bad else f'{len(bad)} BAD: {bad[:5]}'}")
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
