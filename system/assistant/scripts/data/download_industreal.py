"""Fetch one ORIGINAL RGB video from the official ZIP using HTTP ranges.

Avoids downloading ~5 GB merely to test one recording. ZIP CRC is checked by
zipfile; files remain separate from synthetic UI fixtures and model predictions.
"""
import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://data.4tu.nl/file/b008dd74-020d-4ea4-a8ba-7bb60769d224/cf21602e-4424-42a4-b05d-9897957082ad"


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
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status != 206:
                raise ValueError("Download host does not support partial requests. No complete archive was downloaded.")
            data = response.read(count)
        if len(data) != count:
            raise ValueError("Incomplete archive download")
        self.position += count
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--recording", default="03_assy_0_1")
    args = parser.parse_args()
    with zipfile.ZipFile(RemoteZip()) as archive:
        videos = [i for i in archive.infolist() if Path(i.filename).suffix.lower() in (".mp4", ".avi", ".mkv", ".mov") and not i.is_dir()]
        if args.list:
            print(json.dumps([{"name": i.filename, "bytes": i.file_size} for i in videos], indent=2))
            return
        chosen = [i for i in videos if Path(i.filename).stem == args.recording]
        if len(chosen) != 1:
            raise ValueError("Recording not uniquely found. Run with --list first.")
        info = chosen[0]
        folder = ROOT / "data/generated/industreal/videos"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(info.filename).name
        if target.exists():
            raise ValueError(f"Already exists; not overwritten: {target}")
        partial = target.with_suffix(target.suffix + ".partial")
        if partial.exists():
            raise ValueError(f"A previous partial file exists. Inspect it before retrying: {partial}")
        print(f"Downloading original IndustReal video {info.filename} ({info.file_size / 1e6:.1f} MB)", flush=True)
        with archive.open(info) as source, partial.open("xb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        partial.replace(target)
        (folder / (target.name + ".source.json")).write_text(json.dumps({
            "dataset": "IndustReal", "source": URL, "archive_member": info.filename,
            "crc32": f"{info.CRC:08x}", "bytes": info.file_size, "purpose": "raw video, not labelled predictions"
        }, indent=2), encoding="utf-8")
        print(target, flush=True)


if __name__ == "__main__":
    main()
