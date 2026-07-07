# 0004 — Code-free plasticity & stability maintenance (LOOP-0007 family)

**Hypothesis (one sentence):** Post-switch performance is limited by two *measured* inner-agent
pathologies — critic value-whiplash and progressive plasticity loss — and attacking them
**directly** with code-free mechanisms (the Brain's proven scalar-lever channel, plasticity-
preserving architecture, and generic surprise detection) beats the frozen control where
routing the regime **code** into the inner agent (LOOP-0005/0006) did not.

## Why pivot here — the evidence that forces it

Two mechanism families are now exhausted:

- **Forward-modulation family (LOOP-0002/0004, CLOSED):** 20 variants of "code transforms the
  forward pass" (suppress masks, FiLM, gains, trainable decoder, …) — zero wins vs frozen at
  n=8.
- **Code-directed learning-dynamics family (LOOP-0005/0006, NULL):** routed the regime code
  into gradients (gradgate), critic input (critic_code), an aux loss (auxcode), and optimizer
  state (adamflush), plus scaled/combined variants. **Definitive n=8 verdict (RUN-20260706):
  NULL.** The two apparent reliability leads (critic_code, auxcode_hi) were n=3 sampling
  artifacts — hit_80 0.86→0.835 and 0.85→0.819 from n=3 to n=8, back inside the control band.
  gradgate_gain's composite lead failed holdout.

The through-line across BOTH closed families: **injecting the regime `code` into the inner
agent — anywhere, any pathway — does not help.** The code demonstrably *contains* regime
information (diagnostic ordering, research-log 0001), but using it as a modulation source is a
dead end. Meanwhile the one channel that consistently carries causal signal is the Brain's
**scalar-HP control** (research-log 0001). **So LOOP-0007 stops injecting the code and instead
targets the measured failure modes directly.**

### The two measured pathologies (our targets)
1. **Critic whiplash.** In the scored post-switch windows, value updates dwarf policy updates:
   |ΔV| ≈ 0.917 vs policy-KL ≈ 0.0014 (LOOP-0005 diagnostics). The critic over-corrects at the
   regime boundary → biased GAE exactly where composite is measured.
2. **Plasticity loss (assumed, flagged risk).** Continual-RL agents progressively lose
   plasticity across regimes (dormant units, feature rigidity; Sokar 2023 ReDo, Lyle 2023).
   We have NOT yet measured a plasticity statistic here — **A1 (risk): if plasticity loss is
   absent on this 2-regime task, candidates 2–3 predict null.** Registered probe below.

## Assumptions (explicit, with risk)
- **A1** plasticity loss is present and material over the 2 regimes (probe: dormant-unit
  fraction / effective rank of encoder features rising across the switch). If flat → cands 2,3
  are expected null.
- **A2** critic whiplash is *causal* for the composite, not merely correlated — a more stable
  critic yields less-biased GAE and faster true-return recovery. If the critic re-fit is
  actually *adaptive*, damping it will HURT (guard: hit_80).
- **A3** these mechanisms are implementable in the current surface
  (`neuromod.py`/`network.py` + the opened `ppo.py`/`train.py`) as Brain scalar levers or
  static arch/opt changes — **no code is fed to the inner agent** (that is the whole point).

## Candidate hypotheses (screen n≥8 from the start — the n=3 lesson)

Ranked by prior. Each is **code-free** (nothing routes the regime code into the inner agent).

1. **Decoupled critic learning rate (asymmetric-LR scalar lever).** Give the Brain a scalar
   `critic_lr_scale` lever *separate* from the actor/encoder LR, so it can DAMP the measured
   critic whiplash post-switch. Uses the proven scalar-HP channel; targets pathology (1)
   directly. NOTE this is the code-FREE analogue of LOOP-0006's `critic_code` (which fed the
   code to the critic and went null) — here the critic is not conditioned on anything, its
   *step size* is decoupled. **Predict:** composite ↑ via less-biased GAE in post-switch
   windows; hit_80 ↑. Surface: `network.py`/`ppo.py` + one lever in the Brain action space.
2. **Dormant-neuron reset (ReDo-style plasticity injection).** Periodically reset units whose
   post-switch activation is ~dormant, restoring capacity to adapt to the new regime. Trigger
   = a generic activation statistic, NOT the code. Literature-grounded (Sokar 2023).
   **Predict:** composite ↑ (restored plasticity → faster re-adaptation); registered mechanism
   probe = dormant-fraction ↓ after reset (if the probe doesn't move, fails cleanly). Surface:
   `train.py`/`network.py`. **Needs human surface sign-off (touches optimizer/param reset).**
3. **Plasticity-preserving normalization (static arch).** Add LayerNorm / feature
   normalization shown to mitigate plasticity loss (Lyle 2023) to the shared encoder — a
   static change, no lever, no code. **Predict:** composite ↑ / hit_80 ↑ via sustained
   adaptability across regimes; also a clean test of A1 (if it does nothing, plasticity loss
   probably isn't the bottleneck). Surface: `network.py`.
4. **Surprise-triggered exploration spike (generic change-point).** A detector on the inner
   agent's own TD-error surprise (NOT the Brain code) transiently spikes ent_coef/intrinsic at
   *detected* switches — faster than the Brain's `decision_interval` cadence. **Predict:**
   faster post-switch recovery (composite ↑); guard: over-exploration could depress within-
   regime hit_80. Surface: `ppo.py`/`signals.py`.
5. **Two-timescale encoder-vs-heads (uniform, code-free).** A scalar lever making the shared
   encoder learn slower than the heads, so features stay stable across regimes while readouts
   re-map fast. This is the code-FREE, uniform version of LOOP-0006's `gradgate` (which gated
   the backward pass by the code and went null) — isolating whether the *two-timescale idea
   itself* carries value once the (dead) code-gating is removed. **Predict:** composite ↑;
   null would cleanly retire two-timescale as well. Surface: `network.py`/`ppo.py`.

**Beats-frozen-at-n=8 argument (required):** each targets a *systematic, every-switch,
seed-independent* effect (critic step-size, dormant fraction, normalization, surprise timing,
feature timescale) — deterministic mechanisms, not reliability-tail or baseline-draw artifacts
of the kind that fooled us at n=3 in LOOP-0006.

## What is explicitly NOT in this family (guards)
- Anything that feeds the regime `code` to the inner agent — CLOSED across two families.
- Forward-feature masks/gains/FiLM/trainable-decoder — CLOSED (LOOP-0004 n=8 falsifier).
- Duplicating existing scalar levers (lr, ent_coef, intrinsic_coef, imagined_horizon,
  replay_ratio, replay_prioritization, anchoring_weight) — a NEW lever must be one the Brain
  cannot already express (critic_lr_scale and encoder/head timescale qualify; a global lr does
  not).

## Method discipline carried from LOOP-0006 (do not repeat the mistake)
- **Screen at n≥8, not n=3.** The reliability fork correctly flags *candidates*, but n=3
  over-selects lucky seed triples; a signature only counts if it survives n=8. (Fork itself
  stands — composite-non-degrade AND hit_80 > control CI-upper — just applied at n≥8.)
- Register each prediction before results; a probe-based candidate (2,3) must move its probe or
  it fails regardless of score.

## Pickup state
Brainstorm drafted 2026-07-07 (LOOP-0007), user-directed pivot after LOOP-0006 closed null.
User curated **all 5 candidates** and signed off the surface for cands 2 & 4 (train loop).

**Implemented (2026-07-07 ~05:58Z), on `autoresearch-run-20260706`:**
- **Cand 1 — decoupled critic LR** (`brain_neuromod_critic_lr_lo`, scale 0.5, commit 7d4c651).
  Resolved the open (a)-vs-(b) design as **(b)**: critic group LR *tracks* the Brain-lever/anneal
  main LR × scale (single write path `apply_inner_lr`, mirrored at all 3 LR sites). Default
  1.0 = off, baseline-identical.
- **Cand 5 — two-timescale encoder LR** (`brain_neuromod_encoder_lr_lo`, scale 0.5, commit
  0477a63). Same param-group mechanics; encoder LR = main × scale.
- **Cand 3 — plasticity LayerNorm** (`brain_neuromod_plasticity_norm`, commit e35b5a3). Static
  `nn.LayerNorm(flat_size)` on the shared encoder representation, before mask + heads; also the
  A1 test. Off = forward byte-identical to baseline.
- **Cand 4 — surprise-triggered spike** (`brain_neuromod_surprise_spike`, threshold 0.5, commit
  f758a45). Change-point detector on the agent's own per-update TD-error surprise (`value_loss`);
  on a detected jump, transiently ×2 ent_coef + intrinsic_coef for 3 updates. Off = no-op.
- **Cand 2 — ReDo dormant reset** (`brain_neuromod_redo`, interval 50, commit fa864a1). Every 50
  updates, reset dormant actor/critic head units (Sokar 2023); logs
  `redo_dormant_fraction_{actor,critic}` — the registered probe, which also serves as cand 3's
  A1 diagnostic. Trigger = generic activation stat (forward hooks), not the code.
- Tests: `test_decoupled_critic_lr.py` (11) + `test_plasticity_norm.py` (6) +
  `test_surprise_spike.py` (6) + `test_redo_reset.py` (5). Full regression incl. `test_meta_env`
  — **55 pass**. All 5 code-free, flag-guarded default-off; 31 conditions total.

**Box screening:** wave 1 launched 06:11Z — cands 1 & 5 seeds 1,2 (early cross-read), extending
to n=8 over follow-on waves; cands 3/4/2 queued (available on box after next pull). Gate each
with the reliability fork at **n≥8 ONLY**. For cands 2 & 3, the dormant-fraction probe must move
or they fail regardless of score.

**Next:** screen all 5 to n=8 on the box; then LOOP-0006 close-out (register/log/note/merge).
See [LOOP-0007 loop note](../autoresearch-loops/LOOP-0007-brainstorm.md).
