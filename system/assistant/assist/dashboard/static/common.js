// Shared by the three pages: the run switch (kept in ?run=), API calls, and text helpers.
const params = new URLSearchParams(location.search);
let RUN = params.get("run") || "pred_asdv14";   // v14-psr branch default (pred_vjepap_asd needs V-JEPA 2 outputs)

async function api(path) {
  let r;
  try {
    r = await fetch(path);
  } catch (e) {
    offline();
    throw e;
  }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function offline() {
  if (document.querySelector(".offline")) return;
  const div = document.createElement("div");
  div.className = "offline";
  div.innerHTML = "The dashboard server is not running. Start it with <code>python -m assist.dashboard</code> " +
                  "and reload this page.";
  document.body.prepend(div);
}

function withRun(href, run = RUN) {
  const u = new URL(href, location.href);
  u.searchParams.set("run", run);
  return u.pathname + u.search;
}

// Top bar: page links and the run switch. onRun(run) re-renders the page without a reload.
async function topBar(page, onRun) {
  const bar = document.querySelector(".bar");
  bar.innerHTML = `
    <a class="brand" href="${withRun("/")}">Assembly assistant replay</a>
    <nav>
      <a href="${withRun("/")}" ${page === "line" ? 'aria-current="page"' : ""}>Line</a>
      <a href="${withRun("/analytics.html")}" ${page === "analytics" ? 'aria-current="page"' : ""}>Results</a>
    </nav>
    <span class="spacer"></span>
    <span class="run-note"></span>
    <div class="switch" role="group" aria-label="Inputs"></div>`;
  const runs = await api("/api/runs");
  if (!onRun) {                          // the results page shows every run side by side
    bar.querySelector(".switch").remove();
    bar.querySelector(".run-note").remove();
    return;
  }
  const sw = bar.querySelector(".switch");
  const note = bar.querySelector(".run-note");
  const paint = () => {
    sw.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.run === RUN));
    note.textContent = runs.find(r => r.id === RUN)?.note || "";
    bar.querySelectorAll("a").forEach(a => a.href = withRun(a.getAttribute("href")));
  };
  for (const r of runs) {
    const b = document.createElement("button");
    b.textContent = r.label;
    b.dataset.run = r.id;
    if (r.unavailable) { b.disabled = true; b.title = r.unavailable; }
    b.onclick = () => {
      RUN = r.id;
      params.set("run", RUN);
      history.replaceState(null, "", location.pathname + "?" + params);
      paint();
      onRun && onRun(RUN);
    };
    sw.append(b);
  }
  paint();
}

const clock = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const pct = x => x == null ? "-" : `${Math.round(100 * x)}%`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const VERB_ING = { take: "Taking", fit: "Fitting", plug: "Plugging in", tighten: "Tightening", put: "Putting down",
                   pull: "Pulling out", loosen: "Loosening", check: "Checking", browse: "Browsing", align: "Aligning" };
const NOUNS = { pin_short: "short pin", pin_long: "long pin", pin_middle: "middle pin", partial_model: "model",
                objects: "parts", instruction: "instructions", small_screw_pin: "small screw pin" };
function describe(label) {
  const i = label.indexOf("_");
  const verb = i < 0 ? label : label.slice(0, i), noun = i < 0 ? "" : label.slice(i + 1);
  return `${VERB_ING[verb] || verb[0].toUpperCase() + verb.slice(1)} the ${NOUNS[noun] || noun.replaceAll("_", " ")}`;
}
function verbKind(label) {
  const v = label.split("_")[0];
  return v === "take" ? "take" : ["fit", "plug", "tighten"].includes(v) ? "install" : ["pull", "loosen"].includes(v) ? "remove" : "other";
}
const VERB_COLOR = { take: "var(--magenta)", install: "var(--ink)", remove: "var(--graphite)", other: "var(--faint)" };
const TYPE_NAME = { wrong_order: "Out of order", skipped_step: "Skipped step", incorrect_install: "Installed incorrectly",
                    possible_incorrect_install: "Possibly installed incorrectly", wrong_part: "Wrong part",
                    early_warning: "Early warning", cannot_verify: "Cannot verify",
                    order: "Out of order", skipped: "Skipped step", install: "Installed incorrectly" };
const SEVERITY_COLOR = { stop: "var(--red)", warn: "var(--amber)", info: "var(--graphite)" };
const OUTCOME_NAME = { caught: "real mistake", false_alarm: "false alarm", duplicate: "same mistake again" };
