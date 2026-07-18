# 0015 — New-instrument port: dynamics-differing regime pairs (candidate generation)

**Status:** ❌ IP-1 RESOLVED DEAD 2026-07-17 same-day (LOOP-0018 calibration, $0): **C-IP-a
FAIL + C-IP-b FAIL** — see §Calibration outcome below. The `regime_effect="action_flip"`
mode stays in the tree (built, tested, default-off) but earns no ladder. IP-2 remains the
recorded fallback, requiring a fresh user call. Approval trail: user approved all three
decision points (env edit, IP-1, calibration) 2026-07-17; pre-reg
[log 0016](../research-log/0016-2026-07-17-action-flip-calibration-preregistration.md);
runs [LOOP-0018](../autoresearch-loops/LOOP-0018-action-flip-calibration.md).
Originally proposed as candidate generation only:
**Hypothesis (one sentence):** on an instrument whose regimes differ in *transition dynamics*
rather than reward alone, the regime signal outlives the learner's within-lag adaptation, so
the learned WHICH (content-addressed slot selection) — adjudicated dead on the current
instrument — becomes live.
**Owner:** Elijah · **Relates to:** [note 0013](./0013-learned-trigger-selection-gap.md)
(the three selection deaths this routes around) · [2026-07-16 hand-off]
(../hand-offs/2026-07-16-memory-campaign-complete.md) (strategy-fork option 2) ·
[action tree](../plans/2026-07-09-research-action-tree.md) §C (the scaling-axis caveats that
transfer here).

## Why this port (the recorded escape hatch)

LOOP-0013/0014/0015 killed learned slot selection three ways: trunk drift, false-fire churn,
and live-side adaptation. The final mechanism is instrument-structural: regimes differ **only
in the terminal reward override** ([regime_wrapper.py:72-97]
(../../src/lifelong_learning/envs/regime_wrapper.py#L72-L97) — transitions and observations
are identical across regimes), so the only regime-bearing surface in the world model is the
257-param reward head ([world_model.py:59](../../src/lifelong_learning/agents/ppo/world_model.py#L59)),
which re-fits to a new regime within the trigger's one-update lag — the "no switch" hypothesis
wins even at true switches (note 0013 final addendum). Note 0013 records two escape hatches:
oracle identity, or **regimes that differ in dynamics**. This note generates the candidates
for the second.

The structural bet: the WM's **next-state head** predicts 21×H×W logits — a far larger,
slower-adapting surface than the scalar reward head, and one whose error signal is dense
(every step) rather than sparse (terminations only). A dynamics difference should therefore
survive the trigger lag that killed reward fingerprints. **This is an assumption, and the
calibration rung below is designed to falsify it cheaply before any ladder is built.**

## What any port must preserve

- **The ladder logic transfers, numbers don't:** control → learned method → O2 ceiling re-runs
  per instrument; composites are instrument-local (the note 0012 cross-scale caveat). Scorer /
  composite definition untouched (frozen surface).
- **Master-level edits:** a new wrapper + env registration touches `src/lifelong_learning/envs`
  (immutable per manifest) — requires a register row and user approval; presets stay
  master-additive.
- **Brain comparability:** the 19-dim Brain obs is scale-blind, so trained Brains *run* on a
  new instrument, but their normalizer was fit on the reward-flip instrument — descriptive
  only until retrained (action-tree §C caveat).

## Candidates

**IP-1 — Action-semantics flip (recommended).** Regime B swaps the meaning of left/right (or
rotates the action map). Same states, same observations, same reward function — the
*transition function given the action* changes. This is the exact inverse of the current
instrument (dynamics differ, reward identical), giving the cleanest two-instrument contrast
for the paper. Regime signal: next-state prediction error, dense at every step. Build: ~10
lines, a permutation table in/next to `ActionReduceWrapper`, regime-indexed. Risks: the
policy may relearn mirrored controls quickly, shrinking the forgetting cost and hence the
ceiling (calibration measures exactly this); an agent mid-corridor experiences the flip as
instant, so detection may become *too easy* (fine for the WHICH question, weaker for the
WHEN story).

**IP-2 — Slippery-floor regimes (difficulty-tunable backup).** Regime B makes `forward` slip
with probability p (action repeats or veers). Stochastic dynamics change; signal dense; p
tunes how far apart the regimes sit — a knob the current instrument lacks. Risks: composite
noise rises with episode-length variance; extreme p floors success and saturates the scorer.

**IP-3 — Layout swap.** Regimes toggle between two fixed wall layouts (same goal). Strong,
instant regime signal in both the encoder and next-state error. Risks: under FullyObs the
layout is *visible*, so WHICH becomes nearly oracle-easy — closer to task-ID conditioning
than to inferred identity; goal-discovery cost may dominate the composite.

**IP-4 — Regime-correlated observation channel (rejected in-draft).** Cosmetic signal, no
dynamics change; trivially spoofs the WHICH question without testing memory. Recorded only
so nobody re-proposes it.

## The $0 calibration rung (gates before any ladder)

Run before committing to the port (n=8 home evals per arm, one afternoon, $0 — after
approval of the env edit):

- **C-IP-a (headroom exists):** from-scratch O2-analog ceiling on the new instrument
  ≥ +0.10 composite over its own control. Below that, the port buys no room for memory
  methods — stop.
- **C-IP-b (the premise):** the banked-vs-live next-state-error contrast at true switches
  survives ≥ 4 consecutive updates (vs <1 for the reward head on the current instrument,
  note 0013). If next-state error also adapts within the lag, the live-side-adaptation
  mechanism generalizes and the learned-WHICH is dead here too — stop before building
  anything further.

Both gates get a proper pre-registration (research-log entry) if and when the port is
approved; the numbers above are design targets, not registered thresholds yet.

## Decision points for the user (none pre-approved)

1. Approve the master-level env edit (new regime wrapper + registration + register row)?
2. Which candidate first — recommendation: **IP-1**, with IP-2 as the tunable backup?
3. Green-light the calibration rung (~1 home-eval afternoon, $0) once (1) lands?

## Calibration outcome (2026-07-17, LOOP-0018 — both gates FAIL, IP-1 dead)

- **C-IP-a FAIL — no headroom.** Flip-control composite **0.9532** (hit80 1.000): the
  instrument is nearly saturated. The registered risk ("the flip may be too easy") was the
  outcome, and for a structural reason the draft under-weighted: swapping left↔right maps
  every policy to its mirror, and the mirror is *equally competent* — PPO re-reaches it in
  a handful of episodes, so there is nothing to forget. Worse, the O2-analog restore
  **hurts** (−0.0373, p=0.034): restoring a stale snapshot when adaptation is nearly free
  is counterproductive — a clean new instance of the O1 "well-timed interventions can
  hurt" family.
- **C-IP-b FAIL — no persistent signal.** Median WM next-state-error persistence at true
  switches: **0 updates** (56 switches). Turn transitions move one direction feature in a
  21×H×W prediction, so the flip's error contribution dilutes into the mean; the dense
  "next-state head carries the WHICH" bet fails at least in mean-error form here.
  **Scope note (honest):** this reading is confounded by C-IP-a — with adaptation this
  fast, live-side adaptation erases the signal in policy space too. The claim that
  transfers: on every instrument *tested so far*, the regime signal dies inside the
  learner's adaptation lag. "Instrument-general" remains unproven, not proven.
- **What the $0 bought:** the port died at the calibration rung, before any ladder spend —
  the gates did their job. The mode stays in the codebase (default-off, regression-tested)
  as infrastructure for any future variant.
- **Open decision (user's, not pre-approved):** IP-2 (slippery-floor, difficulty tunable
  via slip probability p — a knob IP-1 lacked) is the recorded fallback. Its case improved
  in one way (stochastic dynamics can't be mirrored away — asymmetric competence between
  regimes) and worsened in another (C-IP-b suggests mean WM error is a weak readout; an
  IP-2 registration should gate on a *turn-transition-conditioned* or per-feature error
  instead). Recommendation: only worth it if the learned-WHICH matters to the paper's
  story; the method result (+0.1188 blind flip) stands without it.

## Links

Selection-family anatomy: [note 0013](./0013-learned-trigger-selection-gap.md) · fork:
[2026-07-16 hand-off](../hand-offs/2026-07-16-memory-campaign-complete.md) §Next steps ·
manifest (immutable surfaces): [config/research_manifest.toml](../../config/research_manifest.toml)
· calibration numbers: `evals/wave1_scores.json` (`ipcal`) · loop:
[LOOP-0018](../autoresearch-loops/LOOP-0018-action-flip-calibration.md)
