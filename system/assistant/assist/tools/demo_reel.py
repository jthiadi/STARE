"""Demo reel: title cards in the overlay's design + the key moments of the overlay videos, as one H.264 file.

Clips are rendered straight from the engine output (same renderer as the full overlays), so the reel never
drifts from them. Edit ``REEL`` to change what is shown.

    python -m assist.tools.demo_reel            # reuses clips already rendered in overlays/_reel
    python -m assist.tools.demo_reel --force    # re-render every clip
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from assist.common import ROOT, load_config
from assist.tools.render_overlay import (GRAPHITE, GREEN, H, INK, PAPER, RED, RULE, W, bahn, render, ui,
                                         wrap)

# (kind, ...): ("card", title, lines, seconds) or ("clip", recording, run, start_s, stop_s)
REEL = [
    ("card", "Assembly assistant", ["Watches the worker through a head camera, knows the procedure, and says what is "
                                    "wrong and how to fix it, while it happens.",
                                    "Everything on screen is computed online: at each moment only the video up to "
                                    "that moment is used."], 6),
    ("card", "1. A maintenance job done in the wrong order",
     ["Recording 13_main_2_1, live system (recognised actions + state detector).",
      "The worker re-inserts the pins without swapping the rear chassis first."], 4),
    ("clip", "13_main_2_1", "pred_vjepap_asd", 84, 136),
    ("clip", "13_main_2_1", "pred_vjepap_asd", 158, 172),
    ("clip", "13_main_2_1", "pred_vjepap_asd", 226, None),
    ("card", "2. A correct assembly",
     ["Recording 10_assy_0_1, live system.",
      "The checklist fills in step by step and no mistake is raised."], 4),
    ("clip", "10_assy_0_1", "pred_vjepap_asd", 0, 30),
    ("clip", "10_assy_0_1", "pred_vjepap_asd", 200, None),
    ("card", "3. Parts mounted wrong",
     ["Recording 18_assy_2_5, oracle run: annotated actions and assembly state.",
      "The released state detector misses these two mistakes; with a correct state the engine flags each "
      "within a second."], 5),
    ("clip", "18_assy_2_5", "gt_gt", 318, 342),
    ("clip", "18_assy_2_5", "gt_gt", 368, 392),
]


FOOTER = "IndustReal test recordings. Engine, explanations and overlay: assist/ in this repository."


def card(title: str, lines: list[str], seconds: float, out: Path, fps: float, footer: str = FOOTER) -> None:
    import imageio_ffmpeg

    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    x = 220
    d.rectangle([x - 40, 300, x - 32, 760], fill=GREEN if title.startswith("Assembly") else
                RED if "wrong" in title else INK)
    d.text((x, 300), title, font=bahn(72), fill=INK)
    y = 420
    for k, line in enumerate(lines):          # first paragraph in ink, the rest in graphite
        for part in wrap(line, ui(30), 1300, 3):
            d.text((x, y), part, font=ui(30), fill=INK if k == 0 else GRAPHITE)
            y += 46
        y += 18
    d.line([x, 820, W - 220, 820], fill=RULE, width=2)
    d.text((x, 840), footer, font=ui(20), fill=GRAPHITE)
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", str(out)],
                            stdin=subprocess.PIPE)
    for _ in range(int(seconds * fps)):
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> None:
    import imageio_ffmpeg

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-render clips that already exist.")
    args = parser.parse_args()
    cfg = load_config()
    work = cfg["paths"]["out"] / "overlays" / "_reel"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, item in enumerate(REEL):
        # the run is in the clip's name, so changing a clip's run never reuses the old file
        out = work / (f"{i:02d}.mp4" if item[0] == "card" else f"{i:02d}_{item[2]}_{item[3]}.mp4")
        if item[0] == "card":
            card(item[1], item[2], item[3], out, cfg["fps"])
        elif args.force or not out.is_file():
            _, rid, run, start, stop = item
            render(rid, run, cfg, out, start=start, stop=stop, end_hold_s=3.0)
        parts.append(out)
        print(f"part {i + 1}/{len(REEL)} done", flush=True)
    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    final = cfg["paths"]["out"] / "overlays" / "demo_reel.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    print(f"wrote {final.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
