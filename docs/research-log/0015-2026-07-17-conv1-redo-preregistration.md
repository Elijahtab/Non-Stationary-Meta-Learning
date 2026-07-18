# 0015 — 2026-07-17 — conv1-targeted ReDo (branch F rung): pre-registration (LOOP-0017)

**Registered BEFORE any results exist.** User green-light 2026-07-17 ("1. and 2."): run the
branch-F intervention rung, $0/home. Context: LOOP-0016 met branch F's registered activation
condition for the first time — **first-conv-layer dormancy accumulates 0.125→0.479** (note
0014) while conv2/3 and heads fall. This rung asks the follow-up the paper needs either way:
does eliminating that accumulation move the composite?

## Honest expectations (registered up front)

The heads-ReDo history is a warning (LOOP-0007's small-n mirage); O2 says the bottleneck is
knowledge restoration, not learning dynamics; and the oracle decomposition bounds the entire
*encoder* restoration slice at ~7pp of the ceiling (heads+enc 96.3% − heads 89.6%, note
0012), i.e. ~+0.016 composite — likely **below** the +0.05 screening MDE (note 0011 power
analysis). A null here is therefore informative and expected: it bounds first-layer
plasticity loss as *behaviorally cheap* at this horizon, which is exactly the sentence the
C4 scope correction needs. The gate is written so a PASS must clear the MDE honestly.

## Mechanism (committed pre-launch, `redo_conv1_interval`)

`redo_reset_conv1` — `redo_reset_heads` mechanics at conv1 channel granularity: every N
updates, channels with normalized mean-abs post-ReLU activation ≤ τ=0.025 (the probe's
primary threshold) are reset — incoming filter re-initialised (orthogonal), bias zeroed,
conv2's incoming weights for the channel zeroed (output-neutral reset), Adam moments
cleared. Flag-guarded, default-off. **Single disclosed configuration, no tuning on scored
seeds: interval = 10 updates (~20k steps), τ = 0.025.** The dormancy probe stays on
(interval 1) for mechanism verification.

## Arm (8 evals, home 5070, ~1.1 h, $0)

`redoc1_e1..8`: standard eval protocol, LOOP-0009 seed-1 ep130 Brain, no other mechanism
flags. References archived: `loop9_s1_model_e*` (n=16 composite control) and `dorm_e*`
(the no-intervention dormancy trajectories). Scored by `scripts/score_wave1.py redoc1`.

## Pre-registered gates

- **P-F1a (mechanism holds — read FIRST):** conv1 accumulation eliminated: median-over-runs
  final-window mean f8 ≤ 0.15 (vs 0.479 untreated) AND median rise (f8 − early trough)
  ≤ +0.05. If the reset cannot hold conv1 dormancy down, P-F1b is not read.
- **P-F1b (behavioral payoff):** composite gain vs archived control ≥ **+0.05** with Welch
  p < 0.05 — the honest MDE bar. **Registered null reading:** P-F1a pass + P-F1b fail ⇒
  "first-layer plasticity loss is behaviorally cheap (< +0.05 composite) at this horizon" —
  a bound, not an absence claim; goes into the paper's plasticity-probe paragraph and closes
  branch F again (no further reset rungs without a new instrument or horizon).
- **Harm flag (descriptive):** gain ≤ −0.02 ⇒ the O1 precedent (well-timed interventions can
  hurt) — report prominently.
- **Extension rule:** deciding quantity within ±0.02 of its bar → extend to n=16.

## Registered risks

Resetting input filters mid-training perturbs every downstream feature even with zeroed
outgoing weights (the re-grown filters change future gradients) — the harm flag exists for
this. The 10-update cadence is a judgment call matched to the accumulation timescale
(~390-update climb), not tuned; if the mechanism gate fails on cadence grounds the honest
move is one re-registration with a disclosed iteration count, not silent retries. Dormant-at-
init channels get reset early and harmlessly (they carry no function yet).
