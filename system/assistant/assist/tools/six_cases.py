"""One clip per kind of alert, with sound, every one from a live run. Each card says where the clip comes from.

Chosen by matching alerts to the labelled mistakes. Four cases have a correct live alert on the test videos. The live system
has no correct "installed wrong" on test or val (test: 5 raised, all false alarms), and no test video has a real wrong part,
so those two come from training videos. There the state detector has seen the video, so they show the alert working, not
how well the system finds new mistakes. Their actions come from the TCN alone, predicted by folds that did not train on the
video; the V-JEPA 2 model is not run on training videos. Inputs for a training video, after running the released detector's
weights on it (``assist.extract.train_asd --predict``):

    python -m assist.extract.run_asd --source weights --split train --video 11_assy_3_3
    python -m assist.extract.run_ar --split train --video 11_assy_3_3
    python -m assist.extract.run_hands --split train --video 11_assy_3_3 --compute
    python -m assist.run_engine --video 11_assy_3_3 --actions pred --state asd --split train
    python -m assist.tools.six_cases           # reuses rendered clips -> overlays/cards/six_cases.mp4 and six_cases_<n>_*.mp4
    python -m assist.tools.six_cases --force   # re-render every clip

With ``--v9``, the clips come from our engine with v9's steps as the state (``scripts.mistakes.compare_v9``), in new files.
Only three cases: on test that run raised no "installed wrong" or "wrong part", and its one "check the install" on 23_assy_1_2
names another part, at another time.

    python -m assist.tools.six_cases --v9      # -> overlays/cards/six_cases_v9.mp4 and six_cases_v9_<n>_*.mp4

With ``--cad``, all six come with a CAD picture of the recognised state in the corner (``assist.tools.cad_state``), in new
files: cases 1, 2 and 6 from that v9 run, cases 3, 4 and 5 from the released detector.

    python -m assist.tools.six_cases --cad     # -> overlays/cards/six_cases_cad.mp4 and six_cases_cad_<n>_*.mp4

With ``--final``, the same six also show the action-end label (``assist.extract.action_end``), in new files. It needs each video's
``action_end.jsonl`` (``assist.extract.run_action_end``).

    python -m assist.tools.six_cases --final   # -> overlays/cards/six_cases_final.mp4 and six_cases_final_<n>_*.mp4

With ``--replay``, a replay made after every final label: the action pill and the actions row (renamed "Final labels") show each
action's final label over its whole span. Live those labels only exist after each action ends, so every title card says so. The
alerts, checklist and CAD picture stay those of the live run.

    python -m assist.tools.six_cases --replay  # -> overlays/cards/six_cases_replay.mp4 and six_cases_replay_<n>_*.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import wave

import numpy as np

from assist.common import ROOT, load_config
from assist.tools.add_sound import RATE, VOICES, mix, say_all, tone_mistake, tone_warning
from assist.tools.demo_reel import card
from assist.tools.reel_cards import END_HOLD_S, clip_sounds

LIVE = "pred_vjepap_asd"
TRAIN_LIVE = "pred_asd"
CARD_S = 6
TEST_FOOTER = "IndustReal test recording. Engine, explanations and overlay: assist/ in this repository."
TRAIN_FOOTER = "IndustReal training recording. Engine, explanations and overlay: assist/ in this repository."

CASES = [
    {"title": "1. Skipped step", "alert": "skipped_step", "t": 239.8, "rid": "13_main_2_1", "run": LIVE,
     "start": 226, "stop": None, "footer_split": "test",
     "lines": ["Live system, recording 13_main_2_1. The task ends and the short rear chassis was never installed.",
               "Correct according to the labels. A wrong order alert fired at the same moment."]},
    {"title": "2. Wrong order", "alert": "wrong_order", "t": 128.5, "rid": "13_main_2_1", "run": LIVE,
     "start": 121, "stop": 137, "footer_split": "test",
     "lines": ["Live system, recording 13_main_2_1. The front pin goes in before the short rear chassis.",
               "Correct according to the labels, 4 s after it happened."]},
    {"title": "3. Installed wrong", "alert": "incorrect_install", "t": 139.0, "rid": "11_assy_3_3", "run": TRAIN_LIVE,
     "start": 129, "stop": 152, "footer_split": "train",
     "lines": ["Live system without V-JEPA 2, training recording 11_assy_3_3. The rear wheel is labelled wrong; the alert "
               "comes 3 s later, on the right part.",
               "The state detector trained on this video. On test this alert was raised 5 times, all false alarms."]},
    {"title": "4. Check the install", "alert": "possible_incorrect_install", "t": 311.7, "rid": "23_assy_1_2",
     "run": LIVE, "start": 298, "stop": 322, "footer_split": "test",
     "lines": ["Live system, recording 23_assy_1_2. The front wheel was installed wrong and the camera never confirms it.",
               "Correct according to the labels. The rear wheel check already on screen was a false alarm."]},
    {"title": "5. Wrong part", "alert": "wrong_part", "t": 246.0, "rid": "22_assy_2_3", "run": TRAIN_LIVE,
     "start": 238, "stop": 255, "footer_split": "train",
     "lines": ["Live system without V-JEPA 2, training recording 22_assy_2_3. The short chassis is on a model that does "
               "not use it. Correct, but over 2 min after it went on.",
               "The state detector trained on this video. No test video has a real wrong part."]},
    {"title": "6. Early warning", "alert": "early_warning", "t": 90.1, "rid": "13_main_2_1", "run": LIVE,
     "start": 84, "stop": 104, "footer_split": "test",
     "lines": ["Live system, recording 13_main_2_1. The worker picks up the front pin while the short chassis is still due.",
               "The real mistake follows 34 s later."]},
]

V9_RUN = "pred_vjepap_asdv9"
V9_CASES = [
    {"title": "1. Skipped step", "alert": "skipped_step", "t": 239.8, "rid": "13_main_2_1", "run": V9_RUN,
     "start": 226, "stop": None, "footer_split": "test",
     "lines": ["Our engine with v9's steps, recording 13_main_2_1. The task ends and the short rear chassis was never installed.",
               "Correct according to the labels. A wrong order alert fired at the same moment."]},
    {"title": "2. Wrong order", "alert": "wrong_order", "t": 130.0, "rid": "13_main_2_1", "run": V9_RUN,
     "start": 121, "stop": 137, "footer_split": "test",
     "lines": ["Our engine with v9's steps, recording 13_main_2_1. The front pin goes in before the short rear chassis.",
               "Correct according to the labels, 5.7 s after it happened."]},
    {"title": "6. Early warning", "alert": "early_warning", "t": 90.1, "rid": "13_main_2_1", "run": V9_RUN,
     "start": 84, "stop": 104, "footer_split": "test",
     "lines": ["Our engine with v9's steps, recording 13_main_2_1. The worker picks up the front pin while the short chassis "
               "is still due.",
               "The real mistake follows 34 s later."]},
]
CAD_CASES = [V9_CASES[0], V9_CASES[1], CASES[2], CASES[3], CASES[4], V9_CASES[2]]


def _with_final_label(case: dict) -> dict:
    """On a training video the final label's models trained on that video too, so the card must say so."""
    lines = [line.replace("The state detector trained on this video.",
                          "The final label's models and the state detector trained on this video.") for line in case["lines"]]
    return {**case, "lines": lines}


FINAL_CASES = [_with_final_label(case) for case in CAD_CASES]
REPLAY_CASES = FINAL_CASES
REPLAY_FOOTER = "Action labels are the final labels, filled in after each action ends."


def missing_alert(alerts: list[dict], kind: str, t: float, start: float, stop: float | None) -> bool:
    """True unless the clip shows an alert of this kind at this time."""
    end = np.inf if stop is None else stop
    return not any(a["type"] == kind and abs(a["t"] - t) < 0.2 and start <= a["t"] < end for a in alerts)


def shown_alert(alerts: list[dict], kind: str, t: float) -> dict:
    return next(a for a in alerts if a["type"] == kind and abs(a["t"] - t) < 0.2)


def spoken_alerts(alerts: list[dict], kind: str, t: float) -> list[dict]:
    """Two alerts can fire together and only one sentence plays: make it the one this clip is about."""
    return [a for a in alerts if a["type"] == kind or abs(a["t"] - t) >= 0.2]


def clip_name(tag: str, case: dict) -> str:
    """One case alone, for slides."""
    return f"{tag}_{case['title'][0]}_{case['alert']}.mp4"


def main() -> None:
    import imageio_ffmpeg

    from assist.dashboard import data
    from assist.tools.overlay_cards import render

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-render clips that already exist.")
    parser.add_argument("--voice", choices=list(VOICES), default="David")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--v9", action="store_true", help="Clips from our engine with v9's steps, in new files.")
    choice.add_argument("--cad", action="store_true", help="All six with a CAD picture of the recognised state, in new files.")
    choice.add_argument("--final", action="store_true", help="All six with the CAD picture and the action-end label, in new files.")
    choice.add_argument("--replay", action="store_true", help="All six as a replay made after every final label: the action pill "
                                                              "and row show final labels over each whole action; new files.")
    args = parser.parse_args()
    cases = REPLAY_CASES if args.replay else FINAL_CASES if args.final else CAD_CASES if args.cad else V9_CASES if args.v9 else CASES
    tag = ("six_cases_replay" if args.replay else "six_cases_final" if args.final else "six_cases_cad" if args.cad
           else "six_cases_v9" if args.v9 else "six_cases")
    cfg = load_config()
    overlays = cfg["paths"]["out"] / "overlays"
    work = overlays / "cards" / {"six_cases": "_six", "six_cases_v9": "_six_v9", "six_cases_cad": "_six_cad",
                                 "six_cases_final": "_six_final", "six_cases_replay": "_six_replay"}[tag]
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    parts, offset, tones, speech, voiced, spans = [], 0.0, [], [], {}, []
    for i, case in enumerate(cases):
        alerts = [{**a, "spoken": data.spoken_text(a)} for a in data.score(case["rid"], case["run"])["alerts"]]
        if missing_alert(alerts, case["alert"], case["t"], case["start"], case["stop"]):
            raise ValueError(f"{case['title']}: no {case['alert']} at {case['t']} s in {case['rid']} {case['run']}")
        title = work / f"{i}_card.mp4"
        footer = TRAIN_FOOTER if case["footer_split"] == "train" else TEST_FOOTER
        if args.replay:
            footer = f"{footer} {REPLAY_FOOTER}"
        card(case["title"], case["lines"], CARD_S, title, cfg["fps"], footer=footer)
        clip = work / f"{i}_clip.mp4"
        if args.force or not clip.is_file():
            at_end = case["stop"] is None                 # it fires as the video ends: hold it, then the end card
            render(case["rid"], case["run"], cfg, clip, start=case["start"],
                   stop=case["stop"], end_hold_s=2.0 if at_end else END_HOLD_S,
                   final_alert=shown_alert(alerts, case["alert"], case["t"]) if at_end else None,
                   cad=args.cad or args.final or args.replay, action_end=args.final, final_actions=args.replay)
        _, card_s = imageio_ffmpeg.count_frames_and_secs(str(title))
        _, clip_s = imageio_ffmpeg.count_frames_and_secs(str(clip))
        offset += card_s
        voiced.update(say_all({a["spoken"] for a in alerts if a.get("spoken")}, args.voice, overlays / "_speech"))
        t, s = clip_sounds(spoken_alerts(alerts, case["alert"], case["t"]), case["start"], case["stop"], offset, clip_s,
                           {k: len(v) / RATE for k, v in voiced.items()})
        tones += t
        speech += s
        spans.append((offset, clip_s))
        offset += clip_s
        parts += [title, clip]
        print(f"case {i + 1}/{len(cases)} done, reel at {offset:.1f} s", flush=True)
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
    final = overlays / f"cards/{tag}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags",
                    "+faststart", str(final)], check=True)
    print(f"wrote {final.relative_to(ROOT)}: {offset:.1f} s", flush=True)
    for case, (begin, seconds) in zip(cases, spans):          # each case alone, for slides, without its title card
        name = clip_name(tag, case)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{begin:.3f}", "-t", f"{seconds:.3f}", "-i", str(final),
                        "-c", "copy", "-movflags", "+faststart", str(overlays / "cards" / name)], check=True)
        print(f"wrote {name}: {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    main()
