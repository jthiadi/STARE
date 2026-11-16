"""Serve the card-UI demo videos with a recording picker (supports seeking).  python serve.py [--port 8030]
Then on your PC: ssh -N -L 8030:127.0.0.1:8030 <server>  and open http://localhost:8030"""
import argparse, functools, json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):                                   # HTTP range requests, so the video can be seeked
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        a, b = rng.replace("bytes=", "").split("-")
        start = int(a) if a else max(0, size - int(b)); end = int(b) if a and b else size - 1
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, src, dst):
        left = getattr(self, "_left", None)
        if left is None:
            return super().copyfile(src, dst)
        while left > 0:
            buf = src.read(min(1 << 16, left))
            if not buf:
                break
            dst.write(buf); left -= len(buf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8030); a = ap.parse_args()
    vids = sorted(p.stem for p in (HERE / "videos").glob("*.mp4"))
    (HERE / "videos" / "list.json").write_text(json.dumps(vids))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(HERE)))
    print(f"card demo: {len(vids)} recordings on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
