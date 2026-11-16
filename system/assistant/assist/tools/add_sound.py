"""Add the alert sounds to an overlay video, by the same rules as the dashboard's Sound button.

Early warning: two rising notes and one short line. Mistake: three low, harsh falling notes, then a firm command. The
video picture is copied as it is, unless the last sentence runs past its end; then the last frame is held until it ends.
The speech comes from the voices installed in Windows, processed to carry over noise.

    python -m assist.tools.add_sound --video 13_main_2_1 --run pred_vjepap_asd --voice David
                                                    # -> output/assist/overlays/13_main_2_1_pred_vjepap_asd_sound.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 22050
GROUP_S = 2.0                                  # alerts this soon after a mistake belong to the same moment
SPEECH_DELAY = {"warning": 0.35, "mistake": 0.6}   # seconds from the tone to the sentence
VOICES = {"David": "Microsoft David Desktop", "Zira": "Microsoft Zira Desktop"}
SPEECH_RATE = -1                               # Windows speech speed, -10 to 10: a little slower is clearer in noise
END_MARGIN_S = 0.5


def plan(alerts: list[dict], lengths: dict[str, float]) -> tuple[list, list]:
    """(tones as (time, kind), speech as (time, text, seconds to play)). See ``static/sound.js``: the same rules."""
    sounding = [a for a in alerts if a.get("spoken") and a.get("outcome") != "duplicate"]
    tones, speech = [], []
    last_mistake, speaking_until = -np.inf, -np.inf
    for a in sorted(sounding, key=lambda a: (a["t"], a["kind"] != "mistake")):
        t = a["t"]
        if t - last_mistake < GROUP_S:
            continue
        kind = "mistake" if a["kind"] == "mistake" else "warning"
        tones.append((t, kind))
        if kind == "mistake":
            last_mistake = t
            if speech and speech[-1][0] + speech[-1][2] > t:     # the newest mistake cuts the old sentence
                start, text, _ = speech.pop()
                if t > start:
                    speech.append((start, text, t - start))
        elif t < speaking_until:                                 # never over a sentence that is still playing
            continue
        start = t + SPEECH_DELAY[kind]
        speech.append((start, a["spoken"], lengths[a["spoken"]]))
        speaking_until = start + lengths[a["spoken"]]
    return tones, speech


def _note(frequency: float, seconds: float, volume: float, harsh: bool = False) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    wave_ = np.sin(2 * np.pi * frequency * t)
    if harsh:
        wave_ = np.tanh(4 * wave_) / np.tanh(4.0)
    edge = min(int(0.012 * RATE), len(t) // 2)
    envelope = np.ones(len(t))
    envelope[:edge] = np.linspace(0, 1, edge)
    envelope[-edge:] = np.linspace(1, 0, edge)
    return (volume * wave_ * envelope).astype(np.float32)


def _join(notes: list[np.ndarray], gap_s: float) -> np.ndarray:
    gap = np.zeros(int(gap_s * RATE), np.float32)
    parts = []
    for n in notes:
        parts += [n, gap]
    return np.concatenate(parts[:-1])


def tone_warning() -> np.ndarray:
    """Two rising notes: 'heads up'."""
    return _join([_note(659.25, 0.11, 0.45), _note(880.0, 0.16, 0.45)], 0.02)


def tone_mistake() -> np.ndarray:
    """Three low, harsh falling notes: 'stop'."""
    return _join([_note(440.0, 0.13, 0.8, True), _note(440.0, 0.13, 0.8, True), _note(330.0, 0.19, 0.8, True)], 0.05)


def firm(speech: np.ndarray) -> np.ndarray:
    """Make a voice carry over noise: remove the rumble, lift the 2-4 kHz range where consonants live, and bring quiet
    syllables up towards loud ones, as loud as it can go without clipping."""
    if not np.abs(speech).max():
        return speech
    spectrum = np.fft.rfft(speech)
    frequency = np.fft.rfftfreq(len(speech), 1 / RATE)
    spectrum[frequency < 150] = 0
    spectrum[(frequency >= 2000) & (frequency <= 4000)] *= 1.6
    clear = np.fft.irfft(spectrum, len(speech))
    level = np.abs(clear).max()
    if not level:
        return speech
    squeezed = np.tanh(2.5 * clear / level)
    return (0.95 * squeezed / np.abs(squeezed).max()).astype(np.float32)


def mix(seconds: float, sounds: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """One track of at least ``seconds``, long enough for every (time, sound); scaled down if it would clip."""
    end = max([int(seconds * RATE)] + [int(t * RATE) + len(s) for t, s in sounds])
    out = np.zeros(end, np.float32)
    for t, s in sounds:
        k = int(t * RATE)
        out[k:k + len(s)] += s
    peak = float(np.abs(out).max()) if len(out) else 0.0
    return out / peak if peak > 1 else out


def say_all(texts: set[str], voice: str, folder: Path) -> dict[str, np.ndarray]:
    """Each sentence as sound, from the Windows voice; kept in ``folder`` so a second run is instant."""
    folder.mkdir(parents=True, exist_ok=True)
    jobs = {t: folder / f"{hashlib.sha1(f'{voice}|{SPEECH_RATE}|{t}'.encode()).hexdigest()[:16]}.wav" for t in texts}
    todo = [{"text": t, "path": str(p)} for t, p in jobs.items() if not p.is_file()]
    if todo:
        listing = folder / "todo.json"
        listing.write_text(json.dumps(todo), encoding="utf-8")
        script = ("Add-Type -AssemblyName System.Speech;"
                  f"$items = Get-Content -Raw -Encoding UTF8 '{listing}' | ConvertFrom-Json;"
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                  f"$s.SelectVoice('{VOICES[voice]}'); $s.Rate = {SPEECH_RATE}; $s.Volume = 100;"
                  f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({RATE}, 'Sixteen', 'Mono');"
                  "foreach ($i in $items) { $s.SetOutputToWaveFile($i.path, $f); $s.Speak($i.text) };"
                  "$s.SetOutputToNull()")
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    out = {}
    for text, path in jobs.items():
        with wave.open(str(path)) as w:
            out[text] = firm(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768)
    return out


def build_track(alerts: list[dict], video_seconds: float, voice: str, folder: Path) -> tuple[np.ndarray, list, list]:
    voiced = say_all({a["spoken"] for a in alerts if a.get("spoken")}, voice, folder)
    tones, speech = plan(alerts, {t: len(s) / RATE for t, s in voiced.items()})
    made = {"warning": tone_warning(), "mistake": tone_mistake()}
    sounds = [(t, made[kind]) for t, kind in tones]
    sounds += [(t, voiced[text][:int(seconds * RATE)]) for t, text, seconds in speech]
    return mix(video_seconds, sounds), tones, speech


def write_video(video: Path, track: np.ndarray, out: Path, folder: Path) -> float:
    """The video with this track; returns how many seconds the last frame had to be held."""
    import imageio_ffmpeg

    audio = folder / f"{out.stem}.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    _, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    hold = max(0.0, len(track) / RATE + END_MARGIN_S - seconds) if len(track) / RATE > seconds else 0.0
    picture = (["-vf", f"tpad=stop_mode=clone:stop_duration={hold:.2f}", "-c:v", "libx264", "-preset", "medium",
                "-crf", "21", "-pix_fmt", "yuv420p"] if hold else ["-c:v", "copy"])
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                    *picture, "-c:a", "aac", "-b:a", "128k", "-map", "0:v:0", "-map", "1:a:0", "-movflags", "+faststart",
                    str(out)], check=True)
    return hold


def main() -> None:
    from assist.common import load_config
    from assist.dashboard import data

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--run", default=data.LIVE_RUN)
    parser.add_argument("--voice", choices=list(VOICES), default="David")
    parser.add_argument("--design", default="", help='Subfolder of overlays holding the video, e.g. "cards".')
    args = parser.parse_args()
    overlays = load_config()["paths"]["out"] / "overlays"
    video = overlays / args.design / f"{args.video}_{args.run}.mp4"
    out = overlays / args.design / f"{args.video}_{args.run}_sound.mp4"
    folder = overlays / "_sound"
    import imageio_ffmpeg
    _, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    track, tones, speech = build_track(data.station(args.video, args.run)["alerts"], seconds, args.voice, folder)
    hold = write_video(video, track, out, folder)
    print(json.dumps({"out": str(out), "voice": args.voice, "tones": tones, "sentences": len(speech),
                      "last_frame_held_s": round(hold, 2)}))


if __name__ == "__main__":
    main()
