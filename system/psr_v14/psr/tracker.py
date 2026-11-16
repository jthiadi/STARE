"""Causal PSR tracker: previous state + AR events + (optional) visual sub-part verification.

Per frame t it receives only: frame t (optional visual evidence), AR observation at t.
It keeps the assembly state (11 component bits) and emits step completions at the frame the
decision is made, so the measured delay includes any waiting for a better view.

v2 changes (motivated by validation error analysis):
  * soft legality: a group that the *believed* state says is impossible gets a penalty instead of
    being excluded, so one wrong commit no longer blocks the true step later;
  * objects handled since the last committed step are a second feature set;
  * evidence from abstained triggers carries over (decaying) to the next trigger.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .core import (PROC_INFO, INITIAL_STATE, STEP_DESC, apply_steps, bits, code, group_is_legal)
from .events import GENERIC, MOVE_ON, EventExtractor

LOOSEN = {"loosen_nut", "loosen_acorn_nut"}
PART_OUT = {"pull_pin_short", "pull_pin_middle", "pull_pin_long", "pull_screw_pin", "pull_small_screw_pin",
            "pull_objects", "pull_partial_model", "pull_wheel", "put_pin_short", "put_pin_middle", "put_pin_long",
            "put_screw_pin", "put_short_brace", "put_long_brace", "put_wheel", "put_partial_model"}

TABLES = Path(__file__).resolve().parent.parent / "tables.json"
DEMO_CONFIG = Path(__file__).resolve().parent.parent / "demo_config.json"


@dataclass
class TrackerConfig:
    w_prior: float = 1.0
    w_trigger: float = 1.0
    w_action: float = 1.0
    w_since: float = 0.0           # weight of "objects since last committed step" features
    w_ntrig: float = 0.0           # weight of "# fastening triggers since last committed step"
    w_vision: float = 0.0          # 0 disables visual verification
    null_bias: float = 0.0         # added to the "no step" hypothesis
    alpha: float = 3.0             # backoff strength for P(group | proc, state)
    illegal_pen: float = -1e9      # score added to groups impossible under the believed state
    w_carry: float = 0.0           # carry-over of posterior mass from abstained triggers
    carry_decay: float = 0.5
    wait_margin: float = 2.0       # defer decision when top-2 margin below this (vision only)
    max_wait: int = 15             # max frames to wait for a better view
    min_wait: int = 0              # always wait this many frames after a trigger
    min_prob: float = 0.0          # abstain (no step) unless the best group's posterior exceeds this
    hold: bool = False             # keep the decision open until the worker moves on (or max_hold)
    max_hold: int = 60             # frames
    wait_for_view: int = 0         # if no usable view yet when the hold ends, wait up to this many extra frames
    # ---- visual confirmation gate: a proposed step is only reported once the changed sub-part is seen ----
    confirm: bool = False
    confirm_thr: float = 0.5       # mean signed log-odds of the changed component(s) needed to confirm
    reject_thr: float = -1.0       # at or below this the step is rejected (part visibly not attached / not removed)
    confirm_min_views: int = 2     # usable views needed before confirming or rejecting
    max_confirm: int = 60          # frames to wait for a usable, decisive view
    confirm_recent: int = 0        # >0: judge on the mean of the last N usable views (else all views)
    vis_feats: bool = False        # give the step model per-component visual state before/after the trigger
    vis_before: int = 100
    vis_min_q: float = 0.0
    parallel_confirm: bool = False # keep several proposals open; apply each to the belief tentatively
    reselect: bool = False         # if the proposal is not confirmed, report another candidate the views confirm
    reselect_min_prob: float = 0.05
    unconfirmed: str = "emit"
    # ---- visual state re-sync: the verifier's view of each component overrides a wrong belief ----
    resync: bool = False
    resync_thr: float = 2.0        # |log-odds| a view must reach against the belief
    resync_n: int = 8              # consecutive usable views that must all disagree
    resync_expected_only: bool = True   # only steps that belong to this procedure
    confirm_comps: tuple = ()      # if set: only steps changing one of these components go through the gate
    confirm_mode: str = "parts"    # "parts": per-component classifiers; "contrast": crop vs train views of before/after state      # on timeout: "emit" (report, flagged unconfirmed) or "drop" (do not report)
    vision_rerank: bool = False    # vision only re-ranks which step; never changes step-vs-no-step
    vision_window: int = 10**9     # use visual evidence only within this many frames after the trigger
    post_window: int = 0           # >0: decide exactly this many frames after the trigger, using actions seen since
    removal_wait: bool = False     # after loosen_*, hold until a part is pulled/put away (or max_hold_removal)
    max_hold_removal: int = 150

    @classmethod
    def from_file(cls, path: Path = DEMO_CONFIG, **over):
        d = json.loads(Path(path).read_text()) if Path(path).exists() else {}
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d.update(over)
        return cls(**d)


def _bern(counts: dict, k: str, n_g: int, a: str, present: bool) -> float:
    p = (counts.get(f"{k}|{a}", 0) + 0.5) / (n_g + 1.0)
    return math.log(p if present else 1 - p)


class GroupModel:
    def __init__(self, tables: dict | Path = TABLES):
        t = tables if isinstance(tables, dict) else json.loads(Path(tables).read_text())
        self.t = t
        self.groups = [tuple(g) for g in t["groups"]]
        self.actions = t["actions"]
        self.n_trig_names = len(t["triggers"])

    @staticmethod
    def key(g):
        return ",".join(map(str, g))

    def score(self, proc, state, trigger, window, since, cfg: TrackerConfig, ntrig: int = 1, ctx=None):
        t = self.t
        s_code = code(state)
        n_ps = t["n_proc_state"].get(f"{proc}|{s_code}", 0)
        legal = {g: (g == () or group_is_legal(state, g)) for g in self.groups}
        tot_p = sum(t["n_group_proc"].get(f"{proc}|{self.key(g)}", 0) for g in self.groups if legal[g])
        n_legal = sum(legal.values())
        out = {}
        for g in self.groups:
            if not legal[g] and cfg.illegal_pen <= -1e8:
                continue
            k = self.key(g)
            n_g = t["n_group"].get(k, 0)
            p_proc = (t["n_group_proc"].get(f"{proc}|{k}", 0) + 0.5) / (tot_p + 0.5 * n_legal)
            p_state = (t["n_group_proc_state"].get(f"{proc}|{s_code}|{k}", 0) + cfg.alpha * p_proc) / (n_ps + cfg.alpha)
            lp = cfg.w_prior * math.log(p_state)
            lp += cfg.w_trigger * math.log((t["n_trig"].get(f"{k}|{trigger}", 0) + 0.5) / (n_g + 0.5 * self.n_trig_names))
            if cfg.w_action:
                lp += cfg.w_action * sum(_bern(t["n_act"], k, n_g, a, a in window) for a in self.actions)
            if cfg.w_since:
                lp += cfg.w_since * sum(_bern(t["n_since"], k, n_g, a, a in since) for a in self.actions)
            if cfg.w_ntrig and "n_ntrig" in t:
                lp += cfg.w_ntrig * math.log((t["n_ntrig"].get(f"{k}|{min(ntrig, 3)}", 0) + 0.5) / (n_g + 1.5))
            if g == ():
                lp += cfg.null_bias
            if not legal[g]:
                lp += cfg.illegal_pen
            out[g] = lp
        return out


@dataclass
class Decision:
    t0: int
    trigger: str
    window: frozenset
    since: frozenset
    base: dict
    ctx: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)
    ntrig: int = 1
    state: tuple = ()
    vis: dict = field(default_factory=dict)
    n_vis: int = 0


class PSRTracker:
    def __init__(self, proc: str, cfg: TrackerConfig | None = None, model: GroupModel | None = None,
                 vision=None):
        self.proc = proc
        self.cfg = cfg or TrackerConfig()
        self.model = model or GroupModel()
        self.vision = vision            # object with .group_evidence(feat, state, groups)
        self.state = bits(INITIAL_STATE[proc])
        self.events = EventExtractor()
        self.since: set = set()         # objects handled since the last committed step
        self.carry: dict = {}
        self.ntrig = 0                  # fastening triggers since the last committed step
        self.last_commit = 0            # frame of the last committed step
        self.pending: Decision | None = None
        self._prev_names: set = set()
        self.moved_on = False           # a "moving on" action started since the pending trigger
        self.part_out = False           # a part was pulled out / put away since the pending trigger
        self.y_hat: list[dict] = []
        self.log: list[dict] = []        # human-readable decision log for the UI
        self.vhist: list = []                  # (t, {k: logodds}) of usable views, for vis_feats
        self.cq: list = []                     # steps proposed, waiting for visual confirmation
        self.disagree = [0] * 11               # consecutive views contradicting the belief, per component
        self.observed = [None] * 11            # last smoothed visual log-odds per component (UI)

    # -------------------------------------------------------------------------------------
    def update(self, t: int, ar_names: list[str], frame_feat=None) -> list[dict]:
        emitted = []
        conf_feat = frame_feat
        if self.pending is None:          # visual evidence only counts for a decision that is already open
            frame_feat = None
        self.since |= {n for n in ar_names if n not in GENERIC}
        starts = set(ar_names) - self._prev_names
        ends = self._prev_names - set(ar_names)
        self._prev_names = set(ar_names)
        if starts & MOVE_ON:
            self.moved_on = True
        if ends & PART_OUT:
            self.part_out = True
        if self.cfg.vis_feats and conf_feat is not None and conf_feat.get("cls") is not None and self.vision is not None:
            if conf_feat.get("quality", conf_feat.get("q", -1)) >= self.cfg.vis_min_q:
                pv = getattr(self.vision, "parts", self.vision)
                p = pv.part_probs(conf_feat["cls"])
                self.vhist.append((t, {k: math.log(min(max(v, 1e-3), 1 - 1e-3) / (1 - min(max(v, 1e-3), 1 - 1e-3))) for k, v in p.items()}))
                if len(self.vhist) > 400:
                    self.vhist = self.vhist[-400:]
        for c in list(self.cq):
            emitted += self._confirm_step(t, conf_feat, c=c)
        if self.cfg.resync and conf_feat is not None and self.vision is not None:
            emitted += self._resync(t, conf_feat)
        for trig in self.events.update(t, ar_names):
            if self.cq and not self.cfg.parallel_confirm:   # a new fastening started: settle open confirmations now
                for c in list(self.cq):
                    emitted += self._confirm_step(t, None, timeout=True, c=c)
            self.ntrig += 1
            ctx = {"duration": trig.duration, "elapsed": t - self.last_commit, "tsince": trig.tsince}
            base = self.model.score(self.proc, self.state, trig.name, trig.window, frozenset(self.since), self.cfg,
                                    self.ntrig, ctx=ctx)
            if self.cfg.w_carry:
                for g in base:
                    base[g] += self.cfg.w_carry * self.carry.get(g, 0.0)
            if self.pending is not None:
                old = self._best(self.pending)
                new = max(base, key=base.get)
                if self.cfg.hold and (old == () or old == new):
                    self.log.append(self._superseded(self.pending, t))   # same step fastened again: keep the later one
                else:
                    emitted += self._commit(t, forced=True)
            self.pending = Decision(t, trig.name, trig.window, frozenset(self.since), base, ctx=ctx,
                                    ntrig=self.ntrig, state=self.state)
            self.moved_on = False
            self.part_out = False
        if self.pending is not None:
            d = self.pending
            for nm in ar_names:
                if t > d.t0:
                    d.post.setdefault(nm, t - d.t0)
            emitted += self._advance(t, frame_feat)
        return emitted

    def vis_window(self, t0, t1):
        """mean per-component log-odds in [t0 - vis_before, t0) and [t0, t1] (causal: t1 = now)."""
        b = [v for tt, v in self.vhist if t0 - self.cfg.vis_before <= tt < t0]
        a = [v for tt, v in self.vhist if t0 <= tt <= t1]
        out = {}
        for k in range(1, 11):
            mb = sum(x[k] for x in b) / len(b) if b else 0.0
            ma = sum(x[k] for x in a) / len(a) if a else mb
            out[k] = (mb, ma)
        return out

    @property
    def confirming(self):
        return self.cq[-1] if self.cq else None

    def finish(self, last_frame: int) -> list[dict]:
        """End of stream: settle the open decision and any step still waiting for confirmation."""
        out = []
        if self.pending is not None:
            out += self._commit(last_frame, forced=True)
        for c in list(self.cq):
            out += self._confirm_step(last_frame, None, timeout=True, c=c)
        return out

    def _totals(self, d: Decision):
        vis = {g: d.vis.get(g, 0.0) / max(d.n_vis, 1) for g in d.base}
        if self.cfg.vision_rerank and d.n_vis:
            steps = [g for g in d.base if g != ()]
            if steps:
                anchor = max(steps, key=lambda g: d.base[g])      # base model's best step keeps its margin to "no step"
                a = vis.get(anchor, 0.0)
                vis = {g: (0.0 if g == () else v - a) for g, v in vis.items()}
        return {g: lp + self.cfg.w_vision * vis[g] for g, lp in d.base.items()}

    def _advance(self, t, frame_feat):
        d = self.pending
        use_vis = self.cfg.w_vision > 0 and self.vision is not None
        if use_vis and frame_feat is not None and t - d.t0 < self.cfg.vision_window:
            vs = self.vision.group_evidence(frame_feat, self.state, list(d.base))
            if vs is not None:                       # None = view not good enough (occluded/blurred)
                for g, v in vs.items():
                    d.vis[g] = d.vis.get(g, 0.0) + v
                d.n_vis += 1
        if self.cfg.post_window:
            if t - d.t0 < self.cfg.post_window:
                return []
            ctx = dict(d.ctx, post=dict(d.post))
            d.base = self.model.score(self.proc, d.state, d.trigger, d.window, d.since, self.cfg, d.ntrig, ctx=ctx)
            return self._commit(t)
        if t - d.t0 < self.cfg.min_wait:
            return []
        if self.cfg.removal_wait and d.trigger in LOOSEN:
            if not self.part_out and t - d.t0 < self.cfg.max_hold_removal:
                return []
        elif self.cfg.hold and not self.moved_on and t - d.t0 < self.cfg.max_hold:
            return []
        if (use_vis and self.cfg.wait_for_view and d.n_vis == 0
                and t - d.t0 < min(self.cfg.max_hold, self.cfg.vision_window) + self.cfg.wait_for_view
                and t - d.t0 < self.cfg.vision_window):
            return []
        if self.cfg.vis_feats:
            d.base = self.model.score(self.proc, d.state, d.trigger, d.window, d.since, self.cfg, d.ntrig,
                                      ctx=dict(d.ctx, vis=self.vis_window(d.t0, t)))
        tot = self._totals(d)
        ranked = sorted(tot.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1e9
        if not use_vis or margin >= self.cfg.wait_margin or t - d.t0 >= self.cfg.max_wait:
            return self._commit(t)
        return []

    def _best(self, d: Decision):
        tot = self._totals(d)
        mx = max(tot.values())
        z = sum(math.exp(v - mx) for v in tot.values())
        best = max(tot, key=tot.get)
        return best if best == () or 1.0 / z >= self.cfg.min_prob else ()

    def _superseded(self, d: Decision, t):
        return {"frame": t, "trigger_frame": d.t0, "trigger": d.trigger, "window": sorted(d.window),
                "decision": [], "steps": [], "views": d.n_vis, "forced": False, "superseded": True,
                "prob": 0.0, "state_before": code(self.state), "state": code(self.state), "top": []}

    def _commit(self, t, forced=False):
        d = self.pending
        self.pending = None
        tot = self._totals(d)
        mx = max(tot.values())
        z = sum(math.exp(v - mx) for v in tot.values())
        post = {g: math.exp(v - mx) / z for g, v in tot.items()}
        best = max(tot, key=tot.get)
        prob = post[best]
        if best != () and prob < self.cfg.min_prob:
            best = ()
        ranked = sorted(tot.items(), key=lambda kv: -kv[1])[:4]
        after_ = apply_steps(self.state, best) if best != () else self.state
        changed_ = [k for k in range(len(after_)) if after_[k] != self.state[k]]
        gated = not self.cfg.confirm_comps or any(k in self.cfg.confirm_comps for k in changed_)
        if best != () and self.cfg.confirm and self.vision is not None and gated:
            # proposed, not reported: wait for the visual check of the changed sub-part
            after = apply_steps(self.state, best)
            c = {"group": best, "t_start": t, "d": d, "prob": prob, "forced": forced, "post": post,
                               "before": self.state, "tentative": self.cfg.parallel_confirm,
                               "ranked": ranked, "comps": [k for k in range(len(after)) if after[k] != self.state[k]],
                               "after": after, "lo_sum": 0.0, "n": 0, "seen": 0, "los": [],
                               "alts": {g: {"after": apply_steps(self.state, g), "los": []}
                                        for g, p in post.items() if g not in ((), best) and p >= self.cfg.reselect_min_prob}}
            self.cq.append(c)
            if c["tentative"]:              # belief moves on now; reverted if the step is never seen
                self.state = after
                self.carry = {}; self.since = set(); self.ntrig = 0; self.last_commit = d.t0
            return []
        return self._emit(t, d, best, prob, forced, post, ranked)

    # ---------------------------------------------------------------- visual confirmation gate
    def _view_logodds(self, feat, c):
        """Signed log-odds that the changed components are in their new state, for one view (None = unusable)."""
        if feat is None or feat.get("cls") is None:
            return None
        q = feat.get("quality", feat.get("q", -1))
        if self.cfg.confirm_mode == "cad":
            if q < getattr(self.vision, "min_q", 0.3) or feat.get("roi") is None:
                return None
            return self.vision.cad_score(feat, code(self.state), code(c["after"]))
        if self.cfg.confirm_mode == "contrast":
            if q < getattr(self.vision, "min_q", 0.25):
                return None
            return self.vision.contrast(feat["cls"], code(self.state), code(c["after"]))
        pv = getattr(self.vision, "parts", self.vision)
        if q < getattr(pv, "min_q", 0.25):
            return None
        p = pv.part_probs(feat["cls"])
        los = []
        for k in c["comps"]:
            if k not in p:
                continue
            pk = min(max(p[k], 1e-3), 1 - 1e-3)
            lo = math.log(pk / (1 - pk))
            los.append(lo if c["after"][k] == 1 else -lo)
        return sum(los) / len(los) if los else None

    def _confirm_step(self, t, feat, timeout=False, c=None):
        c = c if c is not None else self.confirming
        c["seen"] += 1
        lo = None if timeout else self._view_logodds(feat, c)
        if lo is not None:
            c["lo_sum"] += lo; c["n"] += 1; c["los"].append(lo)
            if self.cfg.reselect:
                for g, a in c["alts"].items():
                    l2 = self._view_logodds(feat, {"comps": [k for k in range(len(a["after"])) if a["after"][k] != c["before"][k]],
                                                   "after": a["after"]})
                    if l2 is not None:
                        a["los"].append(l2)
        R = self.cfg.confirm_recent
        recent = lambda xs: (sum(xs[-R:]) / len(xs[-R:])) if (R and xs) else (sum(xs) / len(xs) if xs else 0.0)
        mean = recent(c["los"])
        status = None
        if c["n"] >= self.cfg.confirm_min_views and mean >= self.cfg.confirm_thr:
            status = "confirmed"
        elif c["n"] >= self.cfg.confirm_min_views and mean <= self.cfg.reject_thr:
            status = "rejected"
        elif timeout or t - c["t_start"] >= self.cfg.max_confirm:
            status = "unconfirmed-emit" if self.cfg.unconfirmed == "emit" else "unconfirmed-drop"
        c["mean"] = mean
        if status is None:
            return []
        best = c["group"] if status in ("confirmed", "unconfirmed-emit") else ()
        if self.cfg.reselect and best == () and c["alts"]:
            ok = [(recent(a["los"]), g) for g, a in c["alts"].items()
                  if len(a["los"]) >= self.cfg.confirm_min_views and recent(a["los"]) >= self.cfg.confirm_thr]
            if ok:
                best = max(ok)[1]; status = "reselected"
        self.cq.remove(c)
        conf = {"status": status, "views": c["n"], "mean_logodds": round(mean, 2),
                "proposed": [STEP_DESC[s] for s in c["group"]], "proposed_at": c["t_start"]}
        if not c["tentative"]:
            return self._emit(t, c["d"], best, c["prob"], c["forced"], c["post"], c["ranked"], confirm=conf)
        # tentative: undo the proposal, apply what was actually seen
        state_before = self.state
        s = list(self.state)
        for k in c["comps"]:
            s[k] = c["before"][k]
        if best != ():
            a = apply_steps(c["before"], best)
            for k in range(len(a)):
                if a[k] != c["before"][k]:
                    s[k] = a[k]
        self.state = tuple(s)
        entries = [{"frame": t, "id": st, "description": STEP_DESC[st], "conf": 1} for st in best]
        self.y_hat += entries
        d = c["d"]
        self.log.append({"frame": t, "trigger_frame": d.t0, "trigger": d.trigger, "window": sorted(d.window),
                         "decision": list(best), "steps": [STEP_DESC[x] for x in best], "views": c["n"],
                         "forced": c["forced"], "prob": round(c["prob"], 3), "state_before": code(state_before),
                         "state": code(self.state), "top": [], "confirm": conf})
        return entries

    def _resync(self, t, feat):
        q = feat.get("quality", feat.get("q", -1))
        pv = getattr(self.vision, "parts", self.vision)
        if feat.get("cls") is None or q < getattr(pv, "min_q", 0.0):
            return []
        p = pv.part_probs(feat["cls"])
        busy = self.pending is not None or self.confirming is not None
        for k, pk in p.items():
            pk = min(max(pk, 1e-3), 1 - 1e-3); lo = math.log(pk / (1 - pk))
            self.observed[k] = lo if self.observed[k] is None else 0.7 * self.observed[k] + 0.3 * lo
            belief = 1 if self.state[k] == 1 else 0
            against = (lo >= self.cfg.resync_thr and belief == 0) or (lo <= -self.cfg.resync_thr and belief == 1)
            self.disagree[k] = self.disagree[k] + 1 if against else 0
        if busy:
            return []
        cands = [k for k in range(1, 11) if self.disagree[k] >= self.cfg.resync_n]
        if not cands:
            return []
        k = max(cands, key=lambda c: abs(self.observed[c] or 0))
        step = 3 * k if self.state[k] != 1 else 3 * k + 2
        info = next(p for p in PROC_INFO if p["id"] == step)
        if self.cfg.resync_expected_only and not info.get(f"expected_in_{self.proc}", False):
            self.disagree[k] = 0
            return []
        if not group_is_legal(self.state, (step,)):
            self.disagree[k] = 0
            return []
        state_before = self.state
        self.state = apply_steps(self.state, (step,))
        self.disagree[k] = 0
        self.since = set(); self.ntrig = 0; self.last_commit = t
        e = {"frame": t, "id": step, "description": STEP_DESC[step], "conf": 1}
        self.y_hat.append(e)
        self.log.append({"frame": t, "trigger_frame": t, "trigger": "visual re-sync", "window": [], "decision": [step],
                         "steps": [STEP_DESC[step]], "views": self.cfg.resync_n, "forced": False, "prob": 1.0,
                         "state_before": code(state_before), "state": code(self.state), "top": [],
                         "confirm": {"status": "visual re-sync", "views": self.cfg.resync_n,
                                     "mean_logodds": round(self.observed[k], 2), "proposed": [STEP_DESC[step]],
                                     "proposed_at": t}})
        return [e]

    def _emit(self, t, d, best, prob, forced, post, ranked, confirm=None):
        tot = None
        # carry-over bookkeeping
        if best == ():
            self.carry = {g: self.cfg.carry_decay * self.carry.get(g, 0.0) + (p if g != () else 0.0)
                          for g, p in post.items()}
        else:
            self.carry = {}
            self.since = set()
            self.ntrig = 0
            self.last_commit = d.t0
        state_before = self.state
        entries = [{"frame": t, "id": st, "description": STEP_DESC[st], "conf": 1} for st in best]
        self.state = apply_steps(self.state, best)
        self.y_hat += entries
        self.log.append({
            "frame": t, "trigger_frame": d.t0, "trigger": d.trigger, "window": sorted(d.window),
            "decision": list(best), "steps": [STEP_DESC[s] for s in best], "views": d.n_vis,
            "forced": forced, "prob": round(prob, 3), "state_before": code(state_before), "state": code(self.state),
            "top": [{"group": [STEP_DESC[s] for s in g] or ["(no step)"], "score": round(v, 2),
                     "prob": round(post[g], 3)} for g, v in ranked],
            "confirm": confirm,
        })
        return entries
