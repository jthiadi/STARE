// Sound for the station replay, off until the Sound button is pressed.
// Early warning: two rising notes and one short line. Mistake: three low, harsh falling notes, then a firm command. The
// browser makes both, with no audio files. A sound plays when playback reaches an alert, at the same time the screen
// shows it. assist/tools/add_sound.py follows the same rules for the overlay videos, where the voice can also be
// processed to carry over noise; a browser's speech output cannot.
const MAX_STEP_S = 1;                    // a bigger jump between two looks at the clock is a seek, not playback
const GROUP_S = 2;                       // alerts this soon after a mistake belong to the same moment
const VOICE_KEY = "assist.voice";

function englishVoices(voices) {
  return voices.filter(v => /^en[-_]/i.test(v.lang)).sort((a, b) => a.name.localeCompare(b.name));
}

function chooseVoice(voices, name) {     // the saved voice, else the lower, firmer one if this computer has it
  return voices.find(v => v.name === name) || voices.find(v => /David/.test(v.name)) || null;
}

function dueAlerts(alerts, from, to) {
  if (to < from || to - from > MAX_STEP_S) return [];
  return alerts.filter(a => a.spoken && a.outcome !== "duplicate" && from < a.t && a.t <= to);
}

function choose(due, t, lastMistake) {
  const since = t - lastMistake;         // negative after rewinding past that mistake: it no longer applies
  if (!due.length || (since >= 0 && since < GROUP_S)) return null;
  return due.find(a => a.kind === "mistake") || due[0];
}

function makeSound() {
  let context = null, lastT = 0, lastMistake = -Infinity, waiting = null, voiceName = null;
  try { voiceName = localStorage.getItem(VOICE_KEY); } catch (e) { /* the page works without remembering it */ }

  function tone(frequency, start, seconds, volume, shape) {
    const osc = context.createOscillator(), gain = context.createGain();
    const at = context.currentTime + start;
    osc.type = shape;
    osc.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(volume, at + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + seconds);
    osc.connect(gain).connect(context.destination);
    osc.start(at);
    osc.stop(at + seconds + 0.02);
  }

  function warningTone() {                             // two rising notes: "heads up"
    tone(659.25, 0, 0.11, 0.25, "sine");
    tone(880, 0.13, 0.16, 0.25, "sine");
  }

  function mistakeTone() {                             // three low, harsh falling notes: "stop"
    tone(440, 0, 0.13, 0.35, "square");
    tone(440, 0.18, 0.13, 0.35, "square");
    tone(330, 0.36, 0.19, 0.35, "square");
  }

  function say(text, delay) {
    if (!window.speechSynthesis) return;
    const line = new SpeechSynthesisUtterance(text);
    const voice = chooseVoice(speechSynthesis.getVoices(), voiceName);
    if (voice) line.voice = voice;
    line.lang = voice ? voice.lang : "en-US";
    line.volume = 1;
    line.rate = 0.92;                                  // a little slower and lower: firm, and clearer in noise
    line.pitch = 0.85;
    clearTimeout(waiting);
    waiting = setTimeout(() => speechSynthesis.speak(line), delay);
  }

  return {
    enabled: false,
    voice: () => voiceName,
    setVoice(name) {
      voiceName = name || null;
      try { localStorage.setItem(VOICE_KEY, name); } catch (e) { /* see above */ }
      this.quiet();
      say("This is how the alerts will sound.", 0);    // hear it when choosing
    },
    toggle() {
      this.enabled = !this.enabled;
      if (this.enabled) {
        context = context || new AudioContext();       // browsers only allow this after a click
        context.resume();
      } else {
        this.quiet();
      }
      return this.enabled;
    },
    quiet() {
      clearTimeout(waiting);                           // also a sentence that was about to start
      if (window.speechSynthesis) speechSynthesis.cancel();
    },
    // called on every drawn frame; ``playing`` is false while paused or scrubbing
    tick(alerts, t, playing) {
      const from = lastT;
      lastT = t;
      if (!this.enabled || !playing) return;
      const a = choose(dueAlerts(alerts, from, t), t, lastMistake);
      if (!a) return;
      if (a.kind === "mistake") {
        lastMistake = t;
        this.quiet();                                  // the newest mistake matters more than the old sentence
        mistakeTone();
        say(a.spoken, 600);
      } else {
        warningTone();
        if (!window.speechSynthesis || !speechSynthesis.speaking) say(a.spoken, 350);  // never over a mistake's reason
      }
    },
  };
}

if (typeof module !== "undefined") module.exports = { dueAlerts, choose, englishVoices, chooseVoice };
