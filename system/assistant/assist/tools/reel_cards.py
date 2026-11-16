"""The demo reel in the card design, with the alert sounds. The same title cards and clips as ``demo_reel.py``.

    python -m assist.tools.reel_cards            # reuses clips already rendered in overlays/cards/_reel
    python -m assist.tools.reel_cards --force    # re-render every clip   -> overlays/cards/demo_reel.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import wave

import numpy as np

from assist.common import ROOT, load_config
from assist.tools.add_sound import RATE, VOICES, mix, plan, say_all, tone_mistake, tone_warning
from assist.tools.demo_reel import REEL, card

END_HOLD_S = 7.0                         # at a clip's real end, long enough for the last sentence


def clip_sounds(alerts: list[dict], start: float, stop: float | None, offset: float, clip_seconds: float,
                lengths: dict[str, float]) -> tuple[list, list]:
    """Tones and speech of one clip, on the reel's clock: only alerts inside it, and speech stops when it ends."""
    end_t = np.inf if stop is None else stop
    inside = [{**a, "t": offset + a["t"] - start} for a in alerts if start <= a["t"] < end_t]
    tones, speech = plan(inside, lengths)
    end = offset + clip_seconds
    return tones, [(t, text, min(seconds, end - t)) for t, text, seconds in speech if t < end]


def main() -> None:
    import imageio_ffmpeg

    from assist.dashboard import data
    from assist.tools.overlay_cards import render

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-render clips that already exist.")
    parser.add_argument("--voice", choices=list(VOICES), default="David")
    args = parser.parse_args()
    cfg = load_config()
    overlays = cfg["paths"]["out"] / "overlays"
    work = overlays / "cards/_reel"
    work.mkdir(parents=True, exist_ok=True)
    parts, offset, tones, speech = [], 0.0, [], []
    voiced = {}
    for i, item in enumerate(REEL):
        out = work / (f"{i:02d}.mp4" if item[0] == "card" else f"{i:02d}_{item[2]}_{item[3]}.mp4")
        if item[0] == "card":
            card(item[1], item[2], item[3], out, cfg["fps"])
        else:
            _, rid, run, start, stop = item
            if args.force or not out.is_file():
                render(rid, run, cfg, out, start=start, stop=stop, end_hold_s=END_HOLD_S)
        _, seconds = imageio_ffmpeg.count_frames_and_secs(str(out))
        if item[0] == "clip":
            alerts = data.station(rid, run)["alerts"]
            voiced.update(say_all({a["spoken"] for a in alerts if a.get("spoken")}, args.voice, overlays / "_sound"))
            t, s = clip_sounds(alerts, start, stop, offset, seconds, {k: len(v) / RATE for k, v in voiced.items()})
            tones += t
            speech += s
        parts.append(out)
        offset += seconds
        print(f"part {i + 1}/{len(REEL)} done, reel at {offset:.1f} s", flush=True)
    made = {"warning": tone_warning(), "mistake": tone_mistake()}
    track = mix(offset, [(t, made[k]) for t, k in tones] + [(t, voiced[text][:int(n * RATE)]) for t, text, n in speech])
    audio = work / "reel.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    final = overlays / "cards/demo_reel.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(final)], check=True)
    print(f"wrote {final.relative_to(ROOT)}: {offset:.1f} s, tones {[(round(t, 1), k) for t, k in tones]}, "
          f"{len(speech)} sentences")


if __name__ == "__main__":
    main()
