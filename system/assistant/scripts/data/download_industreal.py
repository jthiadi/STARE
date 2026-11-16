"""Fetch ORIGINAL RGB videos from the official ZIP using HTTP ranges: one recording (--recording), or all (--all).

Avoids downloading ~5 GB merely to test one recording. ZIP CRC is checked by
zipfile; files remain separate from synthetic UI fixtures and model predictions.
"""
import argparse
import io
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://data.4tu.nl/file/b008dd74-020d-4ea4-a8ba-7bb60769d224/cf21602e-4424-42a4-b05d-9897957082ad"
RETRIES = 6                                     # per range request; waits 1, 2, 4, 8, 16 s between attempts


class RemoteZip(io.RawIOBase):
    def __init__(self, url=URL):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Invalid archive seek")
        return self.position

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if count <= 0:
            return b""
        if count > 200_000_000:
            raise ValueError("Refusing an unexpectedly large archive read.")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{self.position+count-1}"})
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    if response.status != 206:
                        raise ValueError("Download host does not support partial requests. No complete archive was downloaded.")
                    data = response.read(count)
                if len(data) != count:
                    raise OSError(f"short read: {len(data)} of {count} bytes")
                break
            except OSError as error:             # HTTP 5xx / 429, dropped connection, timeout: retry; other 4xx: give up
                code = getattr(error, "code", None)
                if (code is not None and code < 500 and code != 429) or attempt == RETRIES - 1:
                    raise
                print(f"  {error}; retrying in {2 ** attempt} s", flush=True)
                time.sleep(2 ** attempt)
        self.position += count
        return data


def fetch(archive, info, folder, resume=False):
    """Copy one video out of the remote ZIP. With resume, skip a finished file and restart a leftover .partial."""
    target = folder / Path(info.filename).name
    if target.exists():
        if resume:
            print(f"{target.name} exists, skipped", flush=True)
            return
        raise ValueError(f"Already exists; not overwritten: {target}")
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        if not resume:
            raise ValueError(f"A previous partial file exists. Inspect it before retrying: {partial}")
        partial.unlink()
    print(f"Downloading original IndustReal video {info.filename} ({info.file_size / 1e6:.1f} MB)", flush=True)
    with archive.open(info) as source, partial.open("xb") as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    partial.replace(target)
    (folder / (target.name + ".source.json")).write_text(json.dumps({
        "dataset": "IndustReal", "source": URL, "archive_member": info.filename,
        "crc32": f"{info.CRC:08x}", "bytes": info.file_size, "purpose": "raw video, not labelled predictions"
    }, indent=2), encoding="utf-8")
    print(target, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true", help="every video in the archive (~5 GB); finished files are skipped")
    parser.add_argument("--recording", default="03_assy_0_1")
    args = parser.parse_args()
    with zipfile.ZipFile(RemoteZip()) as archive:
        videos = [i for i in archive.infolist() if Path(i.filename).suffix.lower() in (".mp4", ".avi", ".mkv", ".mov") and not i.is_dir()]
        if args.list:
            print(json.dumps([{"name": i.filename, "bytes": i.file_size} for i in videos], indent=2))
            return
        folder = ROOT / "data/generated/industreal/videos"
        folder.mkdir(parents=True, exist_ok=True)
        if args.all:
            for info in videos:
                fetch(archive, info, folder, resume=True)
            return
        chosen = [i for i in videos if Path(i.filename).stem == args.recording]
        if len(chosen) != 1:
            raise ValueError("Recording not uniquely found. Run with --list first.")
        fetch(archive, chosen[0], folder)


if __name__ == "__main__":
    main()
