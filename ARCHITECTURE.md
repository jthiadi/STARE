# STARE: architecture of the final system (STARE-TCN, "v14 + TCN")

STARE recognises assembly procedure steps (PSR) and raises mistake alerts in egocentric IndustReal video, **online
and causally**: every output at frame $t$ uses only frames, labels and readings up to $t$. This document describes
the final system behind the reported results, with the formulas as implemented and the frozen settings. All
settings are in `system/psr_v14/model/config.json`; code paths are given per section.

| Result (32 IndustReal test recordings, official evaluator) | All: POS / F1 / delay | With errors: POS / F1 / delay |
|---|---|---|
| **STARE, end to end (labels delayed by measured compute; paper Table 4)** | **0.819 / 0.907 / 6.5 s** | **0.747 / 0.839 / 7.5 s** |
| STARE, compute excluded (as B3 and STORM-PSR report) | 0.812 / 0.904 / 6.3 s | 0.747 / 0.839 / 7.3 s |
| STORM-PSR (CVIU 2025) | 0.812 / 0.901 / 15.5 s | – |
| B3, IndustReal paper | 0.797 / 0.883 / 22.4 s | 0.731 / 0.816 / 20.4 s |

Mistakes, end to end: 9 of 32 found with the right type, 3.6 false alarms per hour (paper Table 6).

---

## 1. Overview

```
video, 10 fps ──┬─► (A) Action recognition: DINOv2 + VideoMAE features → causal verb/noun TCN → action a_t
                │
                ├─► (B) Visual state verification
                │      even t: localizer (DINOv2-S) → 336 px crop → 2 × DINOv2-B part verifiers → p_t(k), q_t
                │      odd  t: IndustReal ASD (YOLOv8m) → whole-assembly state c_t
                │
                └─► (C) State tracker (causal filter over assembly states)
                       Step transitions (on action triggers)  ┐
                       Evidence update (verifiers, ASD)       ├─► belief b_t(s) ─► step confirmation ─► completed steps
                                                              ┘
                     (D) Mistake detection
                       during the task: procedure engine on each reported step → wrong order / wrong part / skipped
                       at the end:      end-of-task check on unreported steps → skipped / possible incorrect install
```

**Assembly state.** IndustReal describes the assembly as $K = 11$ components. A state is a vector
$s \in \{0, 1, -1\}^{11}$ (absent, installed, installed incorrectly). Component 0 is the base, so the filter
tracks components $k = 1 \dots 10$. Initial state: `10000000000` (assembly) or `11110111111` (maintenance).
A *step* $j = 3k + m$ changes component $k$: $m = 0$ install ($s_k \to 1$), $m = 1$ incorrect install
($s_k \to -1$), $m = 2$ removal ($s_k \to 0$), the official IndustReal step encoding. A *step group* $g$ is a set of
steps completed together; it is legal from $s$ if its steps can be applied in order.

---

## 2. (A) Action recognition: causal verb × noun TCN

Code: `system/assistant/assist/extract/causal_pipeline.py`, `run_ar.py`,
`scripts/segmentation/train_ar_object_tcn.py`, `scripts/segmentation/combine_verb_noun_actions.py` (teammate's model,
used unchanged).

**Per-frame features** (all causal: frame $t$ uses frames $\le t$):

| Feature | Model | Input |
|---|---|---|
| scene | DINOv2-small, frozen | 3 views of frame $t$ (full, centre, hands) |
| hands, high resolution | DINOv2-small, frozen | hands region on a patch grid |
| hand crops | 2 fine-tuned DINOv2-small backbones (cross-fitted) | fixed hands region of frame $t$ |
| motion | VideoMAE-base (SSv2) | 16-frame clip **ending** at the latest anchor $\le t$ (anchors every 4 frames) |
| motion, IndustReal | 2 fine-tuned VideoMAEs (cross-fitted) | same causal clips |

Concatenated: $x_t \in \mathbb{R}^{8727}$.

**Causal multi-stage TCN** (one for verbs, one for nouns). 4 stages, each with 10 dilated residual layers, hidden 192.
Layer $i$ of a stage:

$$h \leftarrow h + \mathrm{Drop}\big(W^{(i)}_{1\times1}\,\mathrm{ReLU}(\mathrm{Conv}_{3,\;d=2^i}(\mathrm{pad}_{\text{left}}(h, 2\cdot 2^i)))\big)$$

Padding on the left only, so the output at $t$ sees inputs at $t' \le t$ (receptive field about 2,000 frames per stage).
Stage 1 reads $x$; each later stage reads the softmax of the previous stage. The output is the last stage's softmax.
Each target is an ensemble of 18 networks × 2 cross-fitted feature variants, probabilities averaged:
$P_t(\text{verb})$ over 10 verbs + background, $P_t(\text{noun})$ over 18 nouns + background.

**Action probability** over the 74 IndustReal actions $a = (\text{verb}_a, \text{noun}_a)$ plus background:

$$S_t(a) = \log P_t(\text{verb}_a) + \log P_t(\text{noun}_a), \qquad P_t(a) = \frac{e^{S_t(a)}}{\sum_{a'} e^{S_t(a')}}$$

No smoothing, no Viterbi. STARE uses the top action $a_t = \arg\max_a P_t(a)$ (none if background).

Training: the final TCN on train + validation; v14 was selected on **out-of-fold** validation labels, i.e. each
validation video labelled by folds that never trained on it.

---

### Final action label (Ensemble, shown in the interface)

When the TCN's event builder closes an action (a non-background label held 5 frames with probability >= 0.30 opens it;
another label held 5 frames closes it), the action is labelled again from its own frames only
(`system/assistant/assist/extract/action_end.py`, `run_action_end.py`):

- four clip models, 16 frames spread evenly over the action, 3 temporal samplings each, probabilities averaged:
  three V-JEPA 2 ViT-L models with the last 12 transformer layers fine-tuned and an attentive probe (hand crop seed 0,
  hand crop seed 1, work-area crop) and one fine-tuned VideoMAE; their mean is $p^{clip}_i(a)$;
- the TCN's mean probability over the action's frames, $p^{TCN}_i(a)$;
- $s_i(a) = (1-w)\log p^{TCN}_i(a) + w \log p^{clip}_i(a)$ with $w = 0.95$;
- an action-order prior (start and transition counts from the training recordings, additive smoothing 0.01), applied
  as a forward filter over the actions so far with strength $\lambda = 0.2$; the label is the argmax.

Settings chosen on validation (73.81 %), test scored once (75.07 % with true boundaries, 55.52 % on the actions the
system detects itself). It takes 1.23 s per action, so the state tracker keeps the TCN labels (paper Table 7).

## 3. (B) Visual state verification

Code: `system/psr_v14/psr/vision.py`, `train_verifier.py`, `predict_verifier.py`, `tools/run_asd.py`, `psr/v14.py`.
The two visual checks alternate to fit the compute budget: part verifiers on **even** frames, ASD on **odd** frames.

### 3.1 Assembly localizer (even frames)

DINOv2-small patch tokens $\{z_u\}$ of the frame are compared with foreground prototypes $F$ (patches inside the
assembly box of training frames, 4,000) and background prototypes $B$ (patches outside it, 8,000):

$$\sigma(u) = \overline{\mathrm{top5}}_{f \in F}\langle z_u, f\rangle \;-\; \overline{\mathrm{top5}}_{b \in B}\langle z_u, b\rangle$$

Patches with $\sigma(u) > 0$ form connected blobs. The blob with the largest $\sum \sigma$, weighted by
$e^{-d/640}$ (distance $d$ in pixels to the previous box, a temporal prior), is the assembly. It is cropped with 10 %
padding and resized to 336 px. Boxes smaller than $60 \times 60$ px are rejected (no reading).

**View quality** of the crop (blur and hand coverage):

$$q = \mathrm{clip}\!\left(\tfrac{\mathrm{Var}(\nabla^2 I)}{150}, 0, 1\right)\cdot \mathrm{clip}(1 - 2\,\rho_{\text{skin}}, 0, 1)$$

where $\rho_{\text{skin}}$ is the share of skin-coloured pixels (HSV rule).

### 3.2 Part verifiers (even frames)

Two DINOv2-base models on the 336 px crop; head on $[\text{CLS};\ \text{mean patch}]$
(Linear 1536→256, GELU, Dropout 0.2, Linear 256→10), sigmoid per component:
$p^{(v)}_t(k) = P(\text{component } k \text{ installed})$, $k = 1..10$.

| Verifier | Unfrozen blocks | Epochs |
|---|---|---|
| `verif_base_r336` | last 4 | 4 |
| `verif_base_r336_e6u6` | last 6 | 6 |

Trained on train + validation crops, labels from `PSR_labels.csv`, frames within ±20 of a state change skipped.
(Selection used 4-fold out-of-fold verifiers.) The two are averaged into a 3-class reading per component:

$$\big(\ell_t(k,\text{absent}),\ \ell_t(k,\text{installed}),\ \ell_t(k,\text{incorrect})\big)
= \log \mathrm{clip}\big(1 - \bar p_t(k),\ \bar p_t(k),\ 10^{-3}\big),\qquad \bar p_t(k) = \tfrac12\textstyle\sum_v p^{(v)}_t(k)$$

(each value clipped to $[10^{-4}, 1]$). The verifiers have no "incorrect" class, hence the constant $10^{-3}$.

### 3.3 Official IndustReal ASD (odd frames)

YOLOv8m `asd_best_IndustRealandSynthetic.pt` (IndustReal authors, 640 px) on the whole frame. Its most confident
box names one of 22 assembly states $c_t$ (11-bit installed/absent code), kept if confidence $\ge 0.3$ and not
"background" / "error_state".

---

## 4. (C) State tracker: causal filter over assembly states

Code: `system/psr_v14/psr/filter_tracker.py` (filter), `v11.py` (short-trigger weighting), `v14.py` (ASD evidence),
`tracker.py` + `lr_model.py` + `blend.py` (step model), `events.py` (triggers).

The tracker keeps a log-belief $\log b_t(s)$ over states, pruned to the 40 most likely (and $\log b > -25$),
normalised after every update. Per frame $t$, in this order:

### 4.1 ASD evidence (odd frames, v14)

If the ASD gives a state $c_t$: for every state $s$ whose installed/absent code differs from $c_t$,

$$\log b(s) \mathrel{+}= w \log \varepsilon, \qquad w = 1,\ \varepsilon = 0.02$$

(a factor of 0.02), then normalise.

### 4.2 Step transitions (on action triggers)

**Trigger.** A trigger fires at the first frame after a *fastening-type* action ends:
`tighten_nut`, `tighten_acorn_nut`, `loosen_nut`, `loosen_acorn_nut`, `fit_wheel`, `plug_wheel`, `pull_wheel`,
`fit_pulley`. It carries: the action, its duration $d$ (frames), the *window* $W$ (non-generic actions seen since the
previous trigger), the *since* set $\Sigma$ (actions since the last reported step), and the number of triggers since
then.

**Step model** $P(g \mid s, \text{trigger})$ over step groups $g$ (including "no step" $g = \varnothing$), a blend of
a count model (naive Bayes, `model/tables.json`) and a logistic regression (`model/lr.pkl`), both trained on
train-split triggers:

$$\log P_{\text{NB}}(g) = \log P(g \mid \text{proc}, s) + \log P(\text{trigger} \mid g)
+ 0.6 \sum_{a} \log P(\mathbb{1}[a \in W] \mid g) + 0.3 \sum_{a} \log P(\mathbb{1}[a \in \Sigma] \mid g)$$

with smoothed counts, $P(g \mid \text{proc}, s)$ backed off to $P(g \mid \text{proc})$ (strength 3), illegal groups
excluded. The logistic regression uses one-hot features of procedure, trigger, state bits, window, since set,
number of triggers, duration and elapsed time; its scores are renormalised over legal groups. Blend:

$$\mathrm{score}(g) = \log P_{\text{NB}}(g) + 4.0\ \log P_{\text{LR}}(g)$$

**Transition.** Every state $s$ moves to $s \oplus g$ (steps of $g$ applied) with

$$\log b'(s \oplus g) \mathrel{\oplus}= \log b(s) + 0.6\,\mathrm{score}(g) - \mathrm{LSE}_{g'}\!\big(0.6\,\mathrm{score}(g')\big)$$

($\oplus$ = log-sum-exp accumulation; 0.6 = `w_model`, a sharpness).

**Short-trigger weighting (v11).** Live action labels flicker, so a trigger from a short action is trusted less.
With $p = \min(1, d / d_0)$, $d_0 = 12$ frames:

$$b_{\text{new}}(s) = p\, b'(s) + (1 - p)\, b(s)$$

(the belief moves with probability $p$, otherwise stays).

### 4.3 Evidence update (even frames, part verifiers)

Each reading reweights every state by how well its components match:

$$\log b(s) \mathrel{+}= \tau\, w_q \sum_{k=1}^{10} \ell_t\big(k, s_k\big), \qquad \tau = 0.05,\quad w_q = \mathrm{clip}(q_t / 0.1,\ 0.1,\ 1)$$

$\tau$ is small because successive readings are strongly correlated; $w_q$ down-weights blurred or hand-covered views.

### 4.4 Step confirmation and reporting

After each reading, let $\hat s = \arg\max_s b(s)$ and $s^{\text{rep}}$ the last reported state. Steps are reported
when **all** hold:
1. $\hat s \ne s^{\text{rep}}$;
2. $b(\hat s) \ge \theta = 0.6$;
3. $\hat s$ has been the most likely state for $\ge 3$ consecutive readings;
4. **visual confirmation** of the changed components $C = \{k : \hat s_k \ne s^{\text{rep}}_k\}$ over the last
   $N = 6$ readings (at least 3):

$$m = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{|C|}\sum_{k \in C} \log\frac{\pi_i(k)}{1 - \pi_i(k)} \;\ge\; 5.0,
\qquad \pi_i(k) = \mathrm{clip}\big(e^{\ell_i(k, \hat s_k)}, 10^{-3}, 1 - 10^{-3}\big)$$

Then the steps between $s^{\text{rep}}$ and $\hat s$ (official state-to-step conversion) are reported **at frame
$t$**, and $s^{\text{rep}} \leftarrow \hat s$. The delay therefore includes any waiting for a confirming view.
(Unused options in the frozen config: no per-frame drift prior, $\varepsilon_{\text{drift}} = 0$; no visual recovery.)

---

## 5. (D) Mistake detection

Code: `system/psr_v14/psr/v11.py` (`mistake_alerts`), teammate's procedure engine
`system/assistant/assist/evaluation/eval.py` (`gt_mistakes`) and task graph
`system/assistant/assist/configs/industreal_graph.yaml`. Uses no labels: the engine is given the **predicted**
state sequence.

**Task graph.** Per task (assembly, maintenance): steps (one component and target value each), **subgoals** in
order (assembly: *chassis* → *front and wheels*), and **requires** (steps that must come first).

**During the task** (each reported step, replayed into a state sequence):

| Alert | Rule |
|---|---|
| Wrong order | a step completes while a required step of the same subgoal is not done (checked after a 10 s grace) |
| Skipped step | a step of a later subgoal completes while an earlier-subgoal step was never done (10 s grace) |
| Wrong part | a component this task never uses becomes installed |

**At the end of the recording** (once), for every required step $j$ (component $k$, target value $v$) never reported,
the engine proposes "skipped step". STARE checks it against its own belief:

$$\mathrm{done} = \sum_{s:\ s_k = v} b_T(s), \qquad \mathrm{seen} = \bar o_k \gtrless 0,\quad
\bar o_k = \tfrac{1}{20}\sum_{\text{last 20}} \big(\ell(k,\text{inst}) - \ell(k,\text{absent})\big)$$

| Condition (in this order) | Alert |
|---|---|
| done $< 0.5$ and not seen | **skipped step** (kept) |
| an install step ($v = 1$) with done $\ge 0.9$: believed done, yet never reported, i.e. never visually confirmed | **possible incorrect install** |
| otherwise (done $\ge 0.5$ or seen) | none (probably done, just not reported) |

The part verifiers only say installed / not installed, so incorrect installs are inferred this way.

**Scoring** (teammate's evaluator): alerts matched to the 32 annotated test mistakes by type and step within
±30 s, one-to-one. Result: 9 / 32 (wrong order 4/4, skipped 2/10, incorrect install 3/18), 3.2 false alarms / h.

---

## 6. Frozen settings

| Group | Setting | Value |
|---|---|---|
| Filter | emission temperature $\tau$ | 0.05 |
| | report threshold $\theta$ | 0.6 |
| | stable readings | 3 |
| | confirmation: readings $N$ / min / threshold | 6 / 3 / 5.0 |
| | view-quality weighting $q_{\text{ref}}$ / $q_{\min}$ | 0.1 / 0.1 |
| | step-model sharpness `w_model` | 0.6 |
| | beam | 40 states |
| Step model | LR weight / action weight / since weight | 4.0 / 0.6 / 0.3 |
| Short triggers | $d_0$ | 12 frames |
| ASD | weight $w$ / min confidence / $\varepsilon$ / frames | 1.0 / 0.3 / 0.02 / odd only |
| Mistakes | order grace / end check "done" / incorrect-install belief / recent readings | 10 s / 0.5 / 0.9 / 20 |

## 7. How it was chosen (no test data)

- Filter, step-model and verifier settings were developed on train + validation only (v9 to v12, with out-of-fold
  verifiers and leave-one-subject-out step models). v13's $\tau$, $\theta$ and $d_0$ came from a pre-registered
  train + validation selection with simulated live labels (`system/psr_v14/selection/v13/`).
- v14 ASD settings: pre-registered validation grid, ASD weight {0.3, 1, 3} × confidence {0.3, 0.5, 0.7} ×
  $\varepsilon$ {0.02, 0.05, 0.1} × {every, odd} frame, plus the v13 control; criterion = POS + F1 on live and
  oracle validation labels minus a delay term; eligible only if validation mistakes are not worse than v13.
  Latency rule: every-frame ASD exceeded 100 ms p95, so odd frames.
- Frozen (`model/SHA256SUMS`), then test scored once. The end-to-end re-tune (labels delayed by measured compute,
  `metrics/e2e/`) chose the same setting.
- A causality check cuts validation recordings at 96 points: no output before a cut changes (0 mismatches).

## 8. Compute (RTX 2080 Ti, `metrics/e2e/results/latency.json`)

| Part | Time per frame |
|---|---|
| Action recognition path (all feature models + TCN, 4-frame chunks) | 87.1 ms (mean) |
| v14 (localizer + verifiers on even frames, ASD on odd, filter) | 31.8 ms mean, 67.8 ms p95 |
| Label delay used end to end: 0.3 s chunk wait + AR + v14 p95 | 0.72 s (8 frames) |

Together 119 ms per frame on one GPU: above the 100 ms budget at 10 fps, so not real time on a single RTX 2080 Ti.
Each part alone is under 100 ms (two GPUs), not measured running together.

## 9. Where to find it

| Part | Path |
|---|---|
| Frozen PSR package (config, step models, hashes) | `system/psr_v14/model/` |
| PSR + mistakes code | `system/psr_v14/psr/` |
| Evaluation on a split | `system/psr_v14/eval_v14.py` |
| Action recognition pipeline | `system/assistant/assist/extract/causal_pipeline.py` |
| Full live pipeline per video (GPU) | `system/assistant/live/run_video.py` |
| Reproduce every reported number (CPU) | `metrics/reproduce_all.py` |
| End-to-end delay study | `metrics/e2e/` |
