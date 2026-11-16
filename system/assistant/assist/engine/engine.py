"""Procedure engine: tracks every step's status from action events (and optionally assembly states) and fires
the mistake rules. Everything is causal: each call only uses what has been seen up to its time ``t``.

Step status: pending -> started (its part was taken) -> placed (its install / removal action began)
-> done (fastened, or its own action ended, or the assembly state shows it).

Rules
  early_warning       take / start of a step whose prerequisites have not even been started
  wrong_order         a step finishes while one of its prerequisites is not finished
  skipped_step        a step of a later subgoal finishes (or the task ends) while this step was never done
  incorrect_install   the state detector reports an error state (or -1 for the component) after an install;
                      or, when ``part_check`` is set, the per-part model repeatedly reads the part as mounted wrong
                      right after its install (this adds the alert only, a step's status never changes)
  possible_incorrect_install  the expected state is not seen within T seconds of the install
  wrong_part          the state shows a component this task does not use
  final_label         (off by default) the action-end label adds install checks and skip checks; alerts only
  cannot_verify       the evidence is weak (low action confidence, hands over the assembly, nothing detected)

An ambiguous action (``plug_short_pin`` can be any of three pins) goes to the step that makes it legal when
one exists, so only actions with no legal reading are flagged.
"""
from __future__ import annotations

from assist.engine.explain import explain
from assist.engine.graph import Task

PENDING, STARTED, PLACED, DONE = 0, 1, 2, 3
STATUS_NAMES = ["pending", "started", "placed", "done"]
MISTAKES = {"wrong_order", "skipped_step", "incorrect_install", "possible_incorrect_install", "wrong_part"}


def start_conf(ev, n: int) -> float:
    """Confidence an online system has when the event is confirmed: its first ``n`` frames only."""
    return round(sum(ev.confs[:n]) / len(ev.confs[:n]), 2) if ev.confs else round(ev.conf, 2)


class Engine:
    def __init__(self, task: Task, components: list[str], cfg: dict, use_state: bool):
        self.task, self.components, self.cfg, self.use_state = task, components, cfg, use_state
        self.status = {s: PENDING for s in task.steps}
        self.alerts: list[dict] = []
        self.log: list[dict] = []          # step status changes, for the dashboard timeline
        self.state = list(task.initial_state)
        self._event_step: dict[int, str] = {}   # id(event) -> step its "do" action was assigned to
        self._event_conf: dict[str, float] = {}  # step -> confidence of the action that placed it
        self._checks: list[dict] = []       # installs waiting for the state detector to confirm them
        self._obs: list[tuple[float, bool]] = []  # (time, usable) of every state observation
        self._occluded: list[tuple[float, bool]] = []
        self._last_state, self._repeat = None, 0
        self._fired: dict[tuple, float] = {}
        self._touched: dict[str, float] = {}     # step -> last time its status changed
        self._last_completion = -1.0              # time the last step was finished
        # steps on components the task changes twice (removed, then put back): one of the two changes is
        # often never seen by the state detector
        comps = [st.component for st in task.steps.values()]
        self._hidden = {s for s, st in task.steps.items() if comps.count(st.component) > 1}
        self._order_checks: list[dict] = []      # order / skip decisions waiting out the grace period
        self._recent: list[dict] = []       # last few action events, as evidence
        self._part_watch: dict[str, dict] = {}   # step just installed -> {until, high readings in a row}

    # ---------------------------------------------------------------- inputs
    def on_action_start(self, t: float, ev) -> None:
        self.tick(t)
        conf = start_conf(ev, self.cfg.get("min_frames", 5))
        self._recent = (self._recent + [{"action": ev.label, "start": round(ev.start, 1), "conf": conf}])[-4:]
        if ev.label in self.task.fasten_actions:
            return
        do = self._resolve(self.task.candidates(ev.label, "do"))
        if do:
            self._event_step[id(ev)] = do
            self._warn_if_early(t, do, conf)
            if self.status[do] < PLACED:
                self._set(t, do, PLACED, "actions")
                self._event_conf[do] = conf
            return
        take = self._resolve(self.task.candidates(ev.label, "take"))
        if take:
            self._warn_if_early(t, take, conf)
            if self.status[take] < STARTED:
                self._set(t, take, STARTED, "actions")

    def on_action_end(self, t: float, ev) -> None:
        self.tick(t)
        if ev.label in self.task.fasten_actions:
            placed = [s for s in self.task.order if self.status[s] == PLACED and self.task.steps[s].done_on == "fasten"]
            for s in placed:
                if self._touched[s] < self._last_completion:   # placed before another step finished: stale
                    self._set(t, s, STARTED, "actions")
                else:
                    self._actions_finished(t, s, self._event_conf.get(s, ev.conf))
            return
        sid = self._event_step.pop(id(ev), None)
        if sid and self.task.steps[sid].done_on == "do" and self.status[sid] < DONE:
            self._actions_finished(t, sid, ev.conf)

    def _actions_finished(self, t: float, sid: str, conf: float) -> None:
        """The actions say a step was just finished. Without a state source that completes it; with one, the
        state decides, and the install is only checked if the state says it was allowed (all prerequisites
        done) -- otherwise the action was most likely misread and it is ignored."""
        self._watch_part(t, sid)
        if not self.use_state:
            # its started prerequisites, and removals it physically needs, must have happened too (their own
            # action was missed: removals have no part to "take", and install actions are often not seen)
            for p in self.task.ancestors(sid):
                if self.status[p] in (STARTED, PLACED) or (self.status[p] < DONE and self.task.steps[p].value == 0):
                    self._complete(t, p, "inferred", conf)
            self._complete(t, sid, "actions", conf)
            return
        if all(self.status[p] == DONE for p in self.task.ancestors(sid)) and \
                not any(c["step"] == sid and not c["resolved"] for c in self._checks):
            self._checks.append({"step": sid, "t": t, "deadline": t + self.cfg["verify_window_s"],
                                 "conf": conf, "resolved": None})

    def on_final_label(self, t: float, label: str, conf: float) -> None:
        """The action-end label of an action that just ended (``assist/extract/action_end.py``). Off unless
        ``cfg["final_label"]`` is set; it only adds alerts and never moves a step:
          install       an install this label names, with its prerequisites done, is checked against the state like
                        one the live labels announced: no expected state within ``verify_window_s`` -> possible
                        incorrect install
          skip_grace_s  a step of a later subgoal was just performed while an earlier-subgoal step is not done: skipped,
                        unless it is done within this many seconds (None: off)
        Labels below ``min_conf`` and fastening actions are ignored."""
        self.tick(t)
        rule = self.cfg.get("final_label")
        if not rule or conf < rule["min_conf"] or label in self.task.fasten_actions:
            return
        sid = self._resolve(self.task.candidates(label, "do"))
        if sid is None or self.status[sid] == DONE:
            return
        if rule["install"] and self.use_state and all(self.status[p] == DONE for p in self.task.ancestors(sid)) \
                and not any(c["step"] == sid and not c["resolved"] for c in self._checks):
            self._checks.append({"step": sid, "t": t, "deadline": t + self.cfg["verify_window_s"], "conf": conf,
                                 "resolved": None, "source": "final_label"})
        if rule["skip_grace_s"] is not None:
            sg = self.task.steps[sid].subgoal
            earlier = [e for e in self.task.order if self.task.steps[e].subgoal < sg and self.status[e] < DONE]
            if earlier:
                self._order_checks.append({"step": sid, "deadline": t + rule["skip_grace_s"], "missing": [],
                                           "earlier": earlier, "conf": conf, "source": "final_label"})

    def on_state(self, t: float, state, conf: float = 1.0) -> None:
        """One state observation: an 11-character string, a list of ints, or ``"error_state"``."""
        self.tick(t)
        if not self.use_state:
            return
        usable = conf >= self.cfg["state_min_conf"]
        self._obs.append((t, usable))
        if not usable:
            return
        key = state if isinstance(state, str) else tuple(state)
        self._repeat = self._repeat + 1 if key == self._last_state else 1
        self._last_state = key
        if self._repeat != self.cfg["state_stable"]:
            return
        if state == "error_state":
            sid = next((c["step"] for c in reversed(self._checks) if not c["resolved"]), None) or self._last_done()
            if sid:
                self._resolve_check(sid)
                self._mistake(t, "incorrect_install", sid, conf=conf, source="state")
            return
        values = [int(c) for c in state] if isinstance(state, str) else list(state)
        for check in self._checks:
            if not check["resolved"]:
                step = self.task.steps[check["step"]]
                v = values[self.components.index(step.component)]
                if v == step.value:
                    check["resolved"] = "ok"
                    self.log.append({"t": round(t, 2), "step": check["step"], "status": "verified", "source": "state"})
        changed = [i for i, (a, b) in enumerate(zip(self.state, values)) if a != b]
        self.state = values
        changed.sort(key=lambda i: min((self.task.order.index(s) for s in
                                        self.task.steps_for(self.components[i], values[i])), default=99))
        for i in changed:
            comp, v = self.components[i], values[i]
            if v == -1:
                sid = (self.task.steps_for(comp, 1) or [None])[0]
                if sid:
                    self._resolve_check(sid)
                    if self.status[sid] < DONE:
                        self._set(t, sid, DONE, "state")
                    self._mistake(t, "incorrect_install", sid, conf=conf, source="state")
                continue
            steps = self.task.steps_for(comp, v)
            open_ = [s for s in steps if self.status[s] < DONE]
            if open_:
                self._infer_hidden(t, open_[0], conf)
                self._complete(t, open_[0], "state", conf)
            elif steps:
                pass                                # already done (e.g. by the actions)
            elif v == 1 and self.task.initial_state[i] == 0:
                self._mistake(t, "wrong_part", None, component=comp, conf=conf, source="state")
            else:                                   # a finished step was undone (e.g. taken off to redo it)
                for s in self.task.steps_for(comp, 1 - v):
                    if self.status[s] == DONE:
                        self._set(t, s, PENDING, "state")
        # A step that puts a component back to a value it already had (re-inserting the pins in maintenance)
        # never shows up as a change: it is done once the state shows the value and its prerequisites are done.
        for s in self.task.order:
            st = self.task.steps[s]
            if s in self._hidden and st.value == 1 and self.status[s] < DONE \
                    and values[self.components.index(st.component)] == 1 \
                    and all(self.status[p] == DONE for p in self.task.ancestors(s)):
                self._complete(t, s, "state", conf)

    def _infer_hidden(self, t: float, sid: str, conf: float) -> None:
        """The state proves ``sid`` was done. Prerequisites the camera cannot see (a pin taken out and put back
        before the next look at the assembly) must physically have happened: mark them done, not missing."""
        for p in self.task.ancestors(sid):
            st = self.task.steps[p]
            now = self.state[self.components.index(st.component)]
            # consistent with the current state: a re-insert must be visible now; a removal is consistent with
            # the part being out now, or back in (it was put back later)
            if p in self._hidden and self.status[p] < DONE and (now == st.value or st.value == 0) and \
                    all(self.status[q] == DONE for q in self.task.ancestors(p)):
                self._set(t, p, DONE, "inferred")

    def on_part_wrong(self, t: float, wrong: list[float], conf: float = 1.0) -> None:
        """One per-part look at the assembly: P(mounted wrong) per component. A part is only judged just after its
        own install, on looks with a good detection and clear hands, after ``looks`` high readings in a row."""
        check = self.cfg.get("part_check")
        self._judge(t, conf, "part", lambda sid: wrong[self.components.index(self.task.steps[sid].component)]
                    > check["thr"] if check else None)

    def on_state_match(self, t: float, similarity: dict[str, float], conf: float = 1.0) -> None:
        """One look: how much the view resembles each defined state (``scripts/mistakes/state_match.py``). Same rules as
        ``on_part_wrong``, but it asks whether the view looks like the state this install should have produced."""
        check = self.cfg.get("part_check")

        def unlike(sid: str) -> bool | None:
            step = self.task.steps[sid]
            values = list(self.state)
            values[self.components.index(step.component)] = step.value
            code = "".join(map(str, values))
            if code not in check["thr"] or code not in similarity:
                return None                         # no reference for this state: this look says nothing
            return similarity[code] < check["thr"][code]

        self._judge(t, conf, "match", unlike if check else None)

    def _judge(self, t: float, conf: float, source: str, reads_wrong) -> None:
        """Count, for each part just installed, looks in a row that read as wrong; alert only, status never changes."""
        self.tick(t)
        check = self.cfg.get("part_check")
        if not check or conf < self.cfg["state_min_conf"] or (self._occluded and self._occluded[-1][1]):
            return
        for sid, watch in list(self._part_watch.items()):
            if t > watch["until"]:
                del self._part_watch[sid]
                continue
            wrong = reads_wrong(sid)
            if wrong is None:
                continue
            watch["high"] = watch["high"] + 1 if wrong else 0
            if watch["high"] >= check["looks"]:
                del self._part_watch[sid]
                self._resolve_check(sid)
                self._mistake(t, "incorrect_install", sid, conf=conf, source=source)

    def _watch_part(self, t: float, sid: str) -> None:
        check = self.cfg.get("part_check")
        if check and self.task.steps[sid].value == 1:
            high = self._part_watch.get(sid, {}).get("high", 0)
            self._part_watch[sid] = {"until": t + check["window_s"], "high": high}

    def on_hands(self, t: float, occluded: bool) -> None:
        self._occluded.append((t, occluded))

    def tick(self, t: float) -> None:
        """Resolve order checks whose grace period and installs whose verification window has passed."""
        due = [c for c in self._order_checks if c["deadline"] <= t]
        self._order_checks = [c for c in self._order_checks if c["deadline"] > t]
        for c in due:
            missing = [p for p in c["missing"] if self.status[p] < DONE]
            if missing:
                self._mistake(c["deadline"], "wrong_order", c["step"], missing=missing, conf=c["conf"],
                              source=c["source"])
            for e in c["earlier"]:
                if self.status[e] < DONE:
                    self._mistake(c["deadline"], "skipped_step", e, trigger=c["step"], conf=c["conf"],
                                  source=c["source"])
        for check in self._checks:
            if check["resolved"] or t < check["deadline"]:
                continue
            check["resolved"] = "timeout"
            t0, t1 = check["t"], check["deadline"]
            seen = sum(1 for (ts, ok) in self._obs if t0 <= ts <= t1 and ok)
            occ = [o for (ts, o) in self._occluded if t0 <= ts <= t1]
            covered = sum(occ) / len(occ) if occ else 0.0
            if seen >= self.cfg["state_stable"] and covered < 0.5:
                self._mistake(t1, "possible_incorrect_install", check["step"], conf=check["conf"],
                              window_s=self.cfg["verify_window_s"],
                              **({"source": check["source"]} if "source" in check else {}))
            else:
                reason = (f"Hands covered the assembly {covered:.0%} of the time after the "
                          f"{self.task.steps[check['step']].name} was finished."
                          if covered >= 0.5 else
                          "The assembly was not detected after this step, so its state could not be checked.")
                self._alert(t1, "cannot_verify", check["step"], reason=reason)

    def finish(self, t: float) -> None:
        self.tick(t)
        for c in self._order_checks:          # the task ended inside their grace period: decide now
            c["deadline"] = t
        self.tick(t)
        for c in self._checks:
            c["deadline"] = min(c["deadline"], t)
        self.tick(t)
        last_seen = max((ts for ts, ok in self._obs if ok), default=None)
        blind = self.use_state and (last_seen is None or t - last_seen > self.cfg["stale_state_s"])
        for s in self.task.order:
            if self.status[s] < DONE:
                if blind:          # the state detector has not seen the assembly lately: cannot claim a skip
                    self._alert(t, "cannot_verify", s, reason="The assembly was not visible at the end of the "
                                f"task, so it is unknown whether the {self.task.steps[s].name} was done.")
                else:
                    self._mistake(t, "skipped_step", s, conf=1.0)

    # ---------------------------------------------------------------- state changes
    def _set(self, t: float, sid: str, status: int, source: str) -> None:
        self.status[sid] = status
        self._touched[sid] = t
        self.log.append({"t": round(t, 2), "step": sid, "status": STATUS_NAMES[status], "source": source})

    def _complete(self, t: float, sid: str, source: str, conf: float) -> None:
        self._set(t, sid, DONE, source)
        self._watch_part(t, sid)
        self._last_completion = t
        sg = self.task.steps[sid].subgoal
        # unfinished prerequisite in the same subgoal: wrong order; anything of an earlier subgoal: skipped.
        # Decided after a grace period: a prerequisite finished within it counts as done together.
        missing = [p for p in self.task.ancestors(sid) if self.status[p] < DONE and self.task.steps[p].subgoal == sg]
        earlier = [e for e in self.task.order if self.task.steps[e].subgoal < sg and self.status[e] < DONE]
        if missing or earlier:
            self._order_checks.append({"step": sid, "deadline": t + self.cfg["order_grace_s"], "missing": missing,
                                       "earlier": earlier, "conf": conf, "source": source})
        for c in self._checks:                       # the state confirmed an install the actions announced
            if c["step"] == sid and not c["resolved"]:
                c["resolved"] = "ok"
                self.log.append({"t": round(t, 2), "step": sid, "status": "verified", "source": source})

    def _resolve(self, cands: list[str]) -> str | None:
        """Which candidate step an action belongs to: one already in progress, else a legal one (preferring
        the one whose prerequisites the worker touched most recently), else the first."""
        open_ = [s for s in cands if self.status[s] < DONE]
        for s in open_:
            if self.status[s] in (STARTED, PLACED):
                return s
        legal = [s for s in open_ if not self._unstarted_prereqs(s)]
        if legal:
            return max(legal, key=lambda s: max((self._touched.get(p, -1.0) for p in self.task.ancestors(s)),
                                                default=-1.0))      # max keeps the first on ties
        return open_[0] if open_ else None

    def _unstarted_prereqs(self, sid: str) -> list[str]:
        sg = self.task.steps[sid].subgoal
        earlier = [e for e in self.task.order if self.task.steps[e].subgoal < sg]
        need = [p for p in self.task.order if p in set(self.task.ancestors(sid)) | set(earlier)]
        return [p for p in need if self.status[p] < STARTED]

    def _warn_if_early(self, t: float, sid: str, conf: float) -> None:
        missing = self._unstarted_prereqs(sid)
        if missing and conf >= self.cfg["low_conf"]:
            self._alert(t, "early_warning", sid, missing=missing, conf=conf)

    def _resolve_check(self, sid: str) -> None:
        for c in self._checks:
            if c["step"] == sid and not c["resolved"]:
                c["resolved"] = "bad"

    def _last_done(self) -> str | None:
        done = [e for e in self.log if e["status"] == "done"]
        return done[-1]["step"] if done else None

    # ---------------------------------------------------------------- alerts
    def _mistake(self, t: float, kind: str, sid: str | None, conf: float = 1.0, **extra) -> None:
        if conf < self.cfg["low_conf"]:
            reason = (f"A possible {kind.replace('_', ' ')} on this step, but the action was recognised with "
                      f"only {conf:.0%} confidence.")
            self._alert(t, "cannot_verify", sid, reason=reason, would_be=kind, conf=conf)
            return
        if (kind, sid, extra.get("component")) in self._fired:      # each mistake is reported once
            return
        self._fired[(kind, sid, extra.get("component"))] = t
        self._alert(t, kind, sid, conf=conf, **extra)

    def _alert(self, t: float, kind: str, sid: str | None, **extra) -> None:
        key = ("cooldown", kind, sid)
        if kind not in MISTAKES:
            if t - self._fired.get(key, -1e9) < self.cfg["alert_cooldown_s"]:
                return
            self._fired[key] = t
        step = self.task.steps.get(sid) if sid else None
        severity = "info" if kind == "cannot_verify" else "warn" if kind in ("early_warning",
                                                                            "possible_incorrect_install") \
            else (step.severity if step else "stop")
        alert = {"t": round(t, 2), "type": kind, "kind": "mistake" if kind in MISTAKES else
                 "warning" if kind == "early_warning" else "status",
                 "severity": severity, "step": sid, "component": step.component if step else extra.get("component"),
                 **{k: v for k, v in extra.items() if k != "component"},
                 "evidence": {"recent_actions": list(self._recent), "state": "".join(map(str, self.state)).replace("-1", "X")}}
        if "conf" in alert:
            alert["evidence"]["confidence"] = round(alert.pop("conf"), 2)
        alert.update(explain(alert, self.task, self.status))
        self.alerts.append(alert)

    # ---------------------------------------------------------------- views
    def allowed_next(self) -> list[str]:
        return [s for s in self.task.order if self.status[s] < DONE and not
                [p for p in self.task.ancestors(s) if self.status[p] < DONE]]

    def progress(self) -> tuple[int, int]:
        return sum(1 for s in self.status.values() if s == DONE), len(self.status)
