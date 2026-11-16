// Station detail: the video, and everything the engine knew at the current video time.
// Actions appear at known_at, alerts and checklist changes at t, so nothing shows before the engine had it.
const RID = params.get("rid");
const HOLD_S = 20;                       // an alert stays on the card this long
const video = document.getElementById("video");
const sound = makeSound();
let S = null;                            // the station response
let lastCard = null, lastChecklist = null;

async function load() {
  S = await api(`/api/station/${RID}?run=${RUN}`);
  document.title = `${RID} replay`;
  document.getElementById("rid").textContent = RID;
  document.getElementById("task").textContent = S.task_title;
  document.getElementById("duration").textContent = `of ${clock(S.duration_s)}`;
  if (!S.has_video) {
    document.getElementById("player").innerHTML = '<div class="no-video">Video not converted. Run ' +
      "<code>python -m assist.tools.transcode_videos</code> and reload.</div>";
  } else if (!video.src) {
    video.src = `/video/${RID}`;
  }
  lastCard = lastChecklist = null;
  scoreLine();
  logTable();
  draw();
}

async function nav() {
  const rids = (await api(`/api/stations?run=${RUN}`)).stations.map(s => s.rid);
  const i = rids.indexOf(RID);
  const set = (id, rid) => {
    const a = document.getElementById(id);
    a.href = rid ? withRun(`/station.html?rid=${rid}`) : "#";
    a.style.visibility = rid ? "visible" : "hidden";
  };
  set("prev", rids[i - 1]);
  set("next", rids[i + 1]);
}

const now = () => S.has_video ? video.currentTime : 0;

function draw() {
  if (!S) return;
  const t = now();
  document.getElementById("clock").textContent = clock(t);
  // at the very end the player has already stopped, but an alert can fall exactly there
  sound.tick(S.alerts, t + (video.ended ? 0.05 : 0), S.has_video && (video.ended || (!video.paused && !video.seeking)));
  currentAction(t);
  alertCard(t);
  checklist(t);
  playhead(t);
  document.querySelectorAll("#log tbody tr").forEach(tr => tr.classList.toggle("future", +tr.dataset.t > t));
}

function currentAction(t) {
  const box = document.getElementById("now");
  if (!box) return;
  const e = S.events.find(e => e.known_at <= t && t < (e.ended_at ?? e.end));
  box.hidden = !e;
  if (!e) return;
  box.className = "now " + verbKind(e.label);
  box.querySelector(".what").textContent = describe(e.label);
  box.querySelector(".conf").textContent = S.actions === "gt" ? "annotated" : `${Math.round(100 * e.conf)}% sure`;
}

function stepStates(t) {
  const status = {}, verified = new Set(), changed = {};
  for (const e of S.log) {
    if (e.t > t) continue;
    if (e.status === "verified") verified.add(e.step);
    else status[e.step] = e.status;
    changed[e.step] = e.t;
  }
  const flagged = {};
  for (const a of S.alerts) {
    if (a.t <= t && a.kind === "mistake" && a.step && !(changed[a.step] > a.t)) flagged[a.step] = a;
  }
  return { status, verified, flagged };
}

function checklist(t) {
  const { status, verified, flagged } = stepStates(t);
  const next = S.steps.find(s => (status[s.id] || "pending") !== "done");
  const key = JSON.stringify([status, [...verified], Object.keys(flagged), next?.id]);
  if (key === lastChecklist) return;
  lastChecklist = key;
  const tag = { started: "started", placed: "placed, waiting for the camera", done: "done", pending: "" };
  let html = "", sub = null;
  for (const s of S.steps) {
    if (s.subgoal !== sub) {
      sub = s.subgoal;
      html += `<li class="subgoal">Stage ${sub + 1}</li>`;
    }
    const st = status[s.id] || "pending";
    const cls = [st, flagged[s.id] ? "flagged" : "", s === next ? "next" : ""].join(" ");
    const text = flagged[s.id] ? TYPE_NAME[flagged[s.id].type].toLowerCase()
               : st === "done" && verified.has(s.id) ? "done, seen by the camera" : tag[st];
    html += `<li class="${cls}"><span class="mark"></span><span class="name">${esc(s.name[0].toUpperCase() + s.name.slice(1))}</span>` +
            `<span class="tag">${esc(text)}</span></li>`;
  }
  document.getElementById("checklist").innerHTML = html;
}

function alertCard(t) {
  const shown = S.alerts.filter(a => a.t <= t && t - a.t <= HOLD_S && a.outcome !== "duplicate");
  const a = shown[shown.length - 1];
  const card = document.getElementById("alert");
  const key = a ? `${a.t}${a.type}` : "ok";
  if (key === lastCard) return;
  const fresh = lastCard !== null && a;
  lastCard = key;
  if (!a) {
    const { status } = stepStates(t);
    const next = S.steps.find(s => (status[s.id] || "pending") !== "done");
    card.className = "alert ok";
    card.innerHTML = `<div class="kind">On track</div><h2>${next ? "Next: " + esc(next.name) : "All steps done"}</h2>` +
                     `<p class="muted">No alert in the last ${HOLD_S} seconds.</p>`;
    return;
  }
  card.className = `alert ${a.severity}${fresh ? " fresh" : ""}`;
  const steps = (a.fix_steps || []).map(s => `<li>${esc(s)}</li>`).join("");
  let truth = "";
  if (a.kind === "mistake") {
    const m = S.mistakes.find(m => m.caught && m.alert_t === a.t);
    truth = a.outcome === "caught" ? `Annotation check: a real mistake${m ? `, annotated at ${clock(m.t)}` : ""}.`
          : "Annotation check: no such mistake in the annotation (false alarm).";
  }
  card.innerHTML = `<div class="kind">${esc(TYPE_NAME[a.type])} at ${clock(a.t)}</div>` +
    `<h2>${esc(a.title || TYPE_NAME[a.type])}</h2><p>${esc(a.why || a.reason || "")}</p>` +
    (steps ? `<ol>${steps}</ol>` : "") + (truth ? `<div class="truth">${truth}</div>` : "");
}

function scoreLine() {
  const s = S.summary;
  document.getElementById("score").textContent = s.n_mistakes || s.n_alerts
    ? `${s.n_mistakes} annotated mistakes: ${s.caught} caught, ${s.missed} missed. ${s.false_alarms} false alarms.`
    : "No annotated mistakes and no mistake alerts in this recording.";
}

function logTable() {
  const rows = S.alerts.filter(a => a.outcome !== "duplicate").map(a => ({
    t: a.t, color: SEVERITY_COLOR[a.severity], name: TYPE_NAME[a.type], text: a.title || a.reason,
    check: a.outcome ? `<span class="pill ${a.outcome}">${OUTCOME_NAME[a.outcome]}</span>` : "" }));
  for (const m of S.mistakes.filter(m => !m.caught)) {
    rows.push({ t: m.t, color: "var(--red)", name: TYPE_NAME[m.type], text: `Annotated mistake: ${m.step_name}`,
                check: '<span class="pill missed">missed</span>' });
  }
  rows.sort((a, b) => a.t - b.t);
  document.querySelector("#log tbody").innerHTML = rows.map(r =>
    `<tr data-t="${r.t}"><td class="num">${clock(r.t)}</td><td><span class="dot" style="--c:${r.color}"></span>${esc(r.name)}</td>` +
    `<td>${esc(r.text)}</td><td>${r.check}</td></tr>`).join("") ||
    '<tr><td colspan="4" class="muted">No alerts.</td></tr>';
  document.querySelectorAll("#log tbody tr[data-t]").forEach(tr => tr.onclick = () => seek(+tr.dataset.t - 3));
}

// ------------------------------------------------------------------ timeline
const LEFT = 120, LANES = { actions: 14, alerts: 46, mistakes: 78 }, AXIS = 112;
let X = null;

function timeline() {
  const svg = document.getElementById("timeline");
  const w = svg.clientWidth || 1000, d = S.duration_s;
  X = t => LEFT + (w - LEFT - 10) * t / d;
  let out = "";
  for (const [lane, y] of Object.entries(LANES)) {
    const name = { actions: "Actions", alerts: "Alerts", mistakes: "Annotated mistakes" }[lane];
    out += `<text x="0" y="${y + 4}">${name}</text><line x1="${LEFT}" x2="${w - 10}" y1="${y}" y2="${y}" stroke="var(--paper)" stroke-width="18"/>`;
  }
  for (const e of S.events) {
    out += `<rect x="${X(e.start)}" y="${LANES.actions - 8}" width="${Math.max(1.5, X(e.end) - X(e.start))}" height="16"` +
           ` fill="${VERB_COLOR[verbKind(e.label)]}"><title>${esc(describe(e.label))} (${clock(e.start)})</title></rect>`;
  }
  for (const a of S.alerts.filter(a => a.outcome !== "duplicate")) {
    const x = X(a.t), y = LANES.alerts;
    out += `<path d="M${x} ${y - 8} L${x + 7} ${y + 7} L${x - 7} ${y + 7} Z" fill="${SEVERITY_COLOR[a.severity]}">` +
           `<title>${esc(a.title || TYPE_NAME[a.type])} (${clock(a.t)})</title></path>`;
  }
  for (const m of S.mistakes) {
    const x = X(m.t), y = LANES.mistakes;
    out += `<rect x="${x - 6}" y="${y - 6}" width="12" height="12" transform="rotate(45 ${x} ${y})"` +
           ` fill="${m.caught ? "var(--red)" : "var(--card)"}" stroke="var(--red)" stroke-width="2">` +
           `<title>${esc(TYPE_NAME[m.type])}: ${esc(m.step_name)} (${clock(m.t)}, ${m.caught ? "caught" : "missed"})</title></rect>`;
  }
  for (let t = 0; t <= d; t += 60) {
    out += `<text x="${X(t)}" y="${AXIS + 14}" text-anchor="middle">${clock(t)}</text>` +
           `<line x1="${X(t)}" x2="${X(t)}" y1="${AXIS - 4}" y2="${AXIS}" stroke="var(--faint)"/>`;
  }
  out += `<rect id="future" y="0" height="${AXIS - 6}" fill="var(--card)" opacity=".6"/>` +
         `<line id="playhead" y1="0" y2="${AXIS}" stroke="var(--ink)" stroke-width="2"/>`;
  svg.innerHTML = out;
  svg.onclick = ev => {
    const r = svg.getBoundingClientRect();
    seek((ev.clientX - r.left - LEFT) / (r.width - LEFT - 10) * d);
  };
  playhead(now());
}

function playhead(t) {
  const line = document.getElementById("playhead"), fut = document.getElementById("future");
  if (!line || !X) return;
  const x = X(Math.min(t, S.duration_s)), end = X(S.duration_s);
  line.setAttribute("x1", x);
  line.setAttribute("x2", x);
  fut.setAttribute("x", x);
  fut.setAttribute("width", Math.max(0, end - x));
}

// ------------------------------------------------------------------ playback
function seek(t) {
  if (!S.has_video) return;
  video.currentTime = Math.max(0, Math.min(t, S.duration_s));
  draw();
}

const playBtn = document.getElementById("play"), speedBtn = document.getElementById("speed");
const soundBtn = document.getElementById("sound");
playBtn.onclick = () => video.paused ? video.play() : video.pause();
video.onclick = () => playBtn.click();
video.onplay = () => { playBtn.textContent = "Pause"; requestAnimationFrame(loop); };
video.onpause = () => { playBtn.textContent = "Play"; sound.quiet(); draw(); };
video.onseeking = () => sound.quiet();
video.onseeked = draw;
video.onended = () => { playBtn.textContent = "Play"; };
soundBtn.onclick = () => {
  const on = sound.toggle();
  soundBtn.textContent = on ? "Sound on" : "Sound off";
  soundBtn.setAttribute("aria-pressed", on);
};
const voiceSelect = document.getElementById("voice");
function fillVoices() {                  // browsers list their voices late, and may change them
  if (!window.speechSynthesis) { voiceSelect.parentElement.hidden = true; return; }
  const voices = englishVoices(speechSynthesis.getVoices());
  const chosen = chooseVoice(voices, sound.voice());
  voiceSelect.innerHTML = (chosen ? "" : '<option value="">Browser default</option>') +
    voices.map(v => `<option value="${esc(v.name)}">${esc(v.name.replace(/ - English.*$/, ""))}</option>`).join("");
  voiceSelect.value = chosen ? chosen.name : "";
}
voiceSelect.onchange = () => sound.setVoice(voiceSelect.value);
if (window.speechSynthesis) speechSynthesis.addEventListener("voiceschanged", fillVoices);
fillVoices();
speedBtn.onclick = () => {
  video.playbackRate = video.playbackRate === 1 ? 2 : 1;
  speedBtn.textContent = `Speed ${video.playbackRate}x`;
};
document.addEventListener("keydown", e => {
  if (e.code === "Space" && e.target === document.body) { e.preventDefault(); playBtn.click(); }
  if (e.code === "ArrowRight" && e.target === document.body) seek(now() + 5);
  if (e.code === "ArrowLeft" && e.target === document.body) seek(now() - 5);
});
function loop() {
  draw();
  if (!video.paused) requestAnimationFrame(loop);
}
window.addEventListener("resize", () => S && timeline());

topBar("station", async () => { await load(); timeline(); nav(); })
  .then(load).then(() => { timeline(); nav(); });
