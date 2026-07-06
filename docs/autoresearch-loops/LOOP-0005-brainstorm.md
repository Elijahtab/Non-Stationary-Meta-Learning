# LOOP-0005 — brainstorm (OPEN, created 2026-07-05)

**Goal:** Generate the next hypothesis family for the neuromodulation research program. The
feature-mask mechanism family is CLOSED (LOOP-0004, research-log 0005); the queue is empty;
compute is not the constraint — idea quality is.
**Verdict:** _(open — brainstorm output delivered 2026-07-05, see below; awaiting human queue
curation before LOOP-0006 can launch)_

## What a brainstorm session should do

Produce **3–7 queue-ready hypotheses** (one line each, mechanism + why it could beat frozen,
per `config/hypothesis_queue.md` format), plus a short research-note draft arguing the chosen
direction. No code. Output = queue entries (human commits them) + a `docs/research-notes/`
entry registering assumptions before any trial runs.

## Evidence base to reason FROM (read these, in order)

1. [AUTORESEARCH.md](../../AUTORESEARCH.md) — decisions register (what the scorer/gates
   actually measure NOW).
2. [LOOP-0004](./LOOP-0004-confirmation-sweep.md) + research-log
   [0005](../research-log/0005-2026-07-05-confirmation-sweep-prediction.md) — what closed & why.
3. Research-notes [0001](../research-notes/0001-trainable-vs-frozen-decoder.md)/
   [0002](../research-notes/0002-stabilizing-trainable-decoder.md) — the trainable-decoder
   null + the R1 co-adaptation instability findings.

## Constraints & hard-won priors (do not re-derive)

- **Closed (don't requeue):** suppress-mask variants — actor-only scope, two-sided/affine gain,
  FiLM per-channel, split actor/critic masks, context-dim expansion, channel gates, trainable
  decoder (± every LR-family stabilizer), longer horizons. 20 variants, zero wins vs frozen.
- **What consistently DOES carry signal** (research-log 0001, notes 0001): the Brain's
  **scalar-HP control** drives adaptation; and the *diagnostic* channels separate conditions
  even when scores don't — `mean_post_switch_neuromod_activity` (oracle 0.43 < neuromod 0.60 <
  random 0.77) and `post_switch_policy_kl` (oracle 0.0034 < 0.0072 < 0.011). The code
  CONTAINS regime information; multiplying features by its decode just isn't a lever that
  helps. Directions that USE the information differently are open: e.g. code → per-module
  learning-rate/plasticity gating; code → replay/memory selection or anchoring weight; code →
  world-model/dreaming modulation; code as an auxiliary prediction target (regime inference);
  code → optimizer state (momentum/trust-region) modulation. Treat these as seeds, not a queue.
- **Every hypothesis must name its predicted effect on** composite (= post-switch window
  success, nothing else) and, optionally, hit_rate_80 as a registered secondary — and should
  survive the question "why would this beat frozen at n=8, not just beat a noisy n=1 anchor?"
- Trial budget shape: ~85 min + ~$3.60 per n=1 trial; editable surface is
  `neuromod.py`/`network.py` (+ optional `signals.py`/`meta_agent.py`) — hypotheses requiring
  edits beyond that surface need a human decision first.

## Brainstorm output (2026-07-05) — candidate queue entries, ranked

Organizing insight: the closed family was 20 variants of "code transforms the **forward
pass**"; what provably carries causal signal is the **scalar-HP / learning-dynamics channel**
(research-log 0001), and the code provably contains regime information (diagnostics ordering).
So every candidate below routes the code's information *somewhere other than forward-path
feature modulation* — into gradients, value conditioning, objectives, or optimizer state.
A key input discovered while drafting: the Brain's 7 scalar levers already cover lr,
ent_coef, intrinsic_coef, imagined_horizon, replay_ratio, replay_prioritization, and
anchoring_weight (`meta_env._apply_action`) — so dream-/replay-gating seeds were **screened
out as duplicative** (see "Not queued" below).

Proposed one-line queue entries (human copies chosen ones into `config/hypothesis_queue.md`):

1. **Plasticity-gated features (gradient-side mask)** — decode the code into a per-feature gate on the *backward* pass of the shared features (forward untouched; code=0 ⇒ exact identity; heads keep full plasticity), so the Brain steers WHERE the encoder learns instead of what it computes — re-aiming the code's proven regime info at the proven-causal LR channel. Full argument + registered predictions: [research note 0003](../research-notes/0003-code-directed-plasticity-gating.md). Surface: `neuromod.py`+`network.py` ✓.
2. **Critic code-conditioning** — feed the 8-D code to the critic head as an extra *input* (concatenation; policy path untouched), so value re-fits per regime through a small fast-adapting pathway; targets the measured post-switch pathology (critic whiplash |ΔV|≈0.917 vs policy KL≈0.0014 → biased GAE exactly in scored windows); conditioning-as-input ≠ feature masking, so outside the closed family. Surface: `network.py` (+`neuromod.py`) ✓.
3. **Regime-inference auxiliary loss** — small head predicts the Brain's context code from inner features (code as *teaching signal*, not modulator), making the representation regime-separable so post-switch readout re-mapping is faster; doubles as the direct test of note-0001's still-open H2 probe. Surface: `network.py` + loss hookup in `ppo.py`/`train.py` — **needs human surface sign-off**.
4. **Code-norm-triggered optimizer-state flush** — decay/reset Adam second moments when ‖code‖ spikes; stale curvature estimates mis-scale post-switch updates and *no existing scalar lever touches optimizer state*. Surface: `train.py` — **needs human surface sign-off**.
5. **Code-weighted feature anchoring** — per-feature anchor penalty weighted by the decoded code ("protect these features, release those") instead of the uniform scalar `anchoring_weight`; targeted consolidation the existing global lever cannot express (that lever is the explicit control to beat). Surface: `ppo.py`/`train.py` — **needs human surface sign-off**.

**Predicted effects (registered per the loop's requirement — composite = post-switch window
success; hit_rate_80 optional secondary):**

1. Composite ↑ with **Δ ≥ +0.005** vs the 3-seed anchor to count as real (n=8 control CI
   half-width ≈ 0.0055); hit80 ↑/flat. Beats-frozen-at-n=8 case: acts deterministically on the
   learning dynamics of every post-switch window in every seed, via the channel with a proven
   causal path — not a baseline-draw or reliability-tail artifact.
2. Composite ↑ via less-biased GAE in post-switch windows; hit80 ↑. n=8 case: corrects a
   *measured systematic* bias present at every switch, not an n=1 selection effect.
3. Composite ↑ modestly; registered mechanism check: regime-decoding-probe accuracy ↑ (if the
   probe doesn't move, the hypothesis fails cleanly regardless of score). n=8 case:
   representation shaping is persistent and seed-independent.
4. Composite ↑ (faster post-switch step-size recalibration); hit80 registered as a *guarded*
   secondary — transient destabilization right after flushes could push it ↓. n=8 case: fires
   deterministically at every switch.
5. Composite ↑ only if targeted protection beats uniform anchoring (the scalar lever the Brain
   already owns); a null reads "consolidation targeting adds nothing over global" — still
   paper-useful. n=8 case: acts on every anchored update.

**Not queued (screened out, with reasons):** dream-trust / replay-selection gating — the Brain
already holds scalar levers for imagined_horizon, replay_ratio, and replay_prioritization, so
only per-sample selection would be new, and it is both outside the editable surface and low
prior; additive/bias forward modulation — inside the closed family (FiLM per-channel
scale+shift already failed); any suppress-mask/gain/trainable-decoder revival — closed by the
n=8 falsifier, guard stands.

## Sibling loops
None yet — the next screening campaign (LOOP-0006) starts from this loop's output.

## Pickup state

**Brainstorm delivered (2026-07-05); the loop now blocks on human curation.** Next actions:
1. Human reviews the 5 candidates above + [research note 0003](../research-notes/0003-code-directed-plasticity-gating.md)
   and commits the chosen entries into `config/hypothesis_queue.md` (updating its STATUS
   block and the AUTORESEARCH.md queue-register row **in the same commit**, per the register
   rule). Entries 1–2 are runnable on the current manifest surface; entries 3–5 first need a
   human decision to extend the editable surface (`config/research_manifest.toml`).
2. Then finalize this note (verdict) and launch LOOP-0006 screening under the 3-seed
   `baseline_primary` anchor — the first run pays its one-time ~3.8 h anchor, then amortizes
   via the fingerprint cache.
3. For hypothesis 1, the trial must add gradient-side diagnostics and must not misread the
   zeroed forward diagnostics as "modulation off" (note 0003, H3/A4).

## Links
`config/hypothesis_queue.md` (closed-family guard + format); `config/program_neuromod.md`
(trial brief the eventual campaign will use); spec
[05-neuromodulation.md](../spec/05-neuromodulation.md) (current mechanism, ground truth).
