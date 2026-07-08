# 0005 — The untested controller: adaptivity probe, bottom-rung control, and the oracle rung

**Hypothesis (one sentence):** The three-family null (LOOP-0002/04, 0005/06, 0007) is not a
statement about mechanisms but about the controller and the instrument — the scout-instrument
Brain is an effectively **untrained noise source around its action-space midpoints**, so the
program has never measured either (a) what the Brain adds over tuned static hyperparameters,
or (b) what a *timed* intervention could achieve — and the three controls pre-registered here
(D0 / O1 / O2) measure exactly those two things.

## Finding 1 (measured, 2026-07-07): the scout Brain is ~untrained; its control signal is noise

Offline analysis of the pushed LOOP-0004 frozen-control artifacts (`origin/results`
`confirm_g0..g3`, condition `brain_neuromod`, seeds 1–8) — no new runs:

**(a) Episode-mean lever values sit exactly on the bound midpoints.**

| lever | settled value (mean of last-episode means, n=8) | midpoint of Brain bounds |
| --- | --- | --- |
| lr | 0.00158 | 0.00155 ([1e-4, 3e-3]) |
| ent_coef | 0.0501 | 0.0505 ([1e-3, 0.1]) |
| intrinsic_coef | 0.2529 | 0.2505 ([1e-3, 0.5]) |
| imagined_horizon | 15.61 | 15.5 ([1, 30]) |
| replay_ratio | 0.2444 | 0.25 ([0, 0.5]) |

Raw episode-mean actions confirm it: all five logged action dims average **0.00 ± 0.03** in
[-1, 1]. A zero action maps linearly to the exact midpoint of every lever bound
(`MetaEnv._apply_action`), and a tanh-Gaussian policy emits mean-zero actions precisely when
it is at (or near) its init (`actor_mean` orthogonal-init gain 0.01).

**(b) The learned obs-response is 4–40× smaller than the policy's own sampling noise.**
Loading pushed `brain_model.pt` checkpoints (control seeds 1 & 5, redo seed 1) and sweeping
the 19-dim observation over generous post-switch ranges (±3σ; also a stylized post-switch
trajectory): the deterministic policy-mean moves at most **0.07–0.23** tanh-units per lever
dim across the whole sweep (0.01–0.14 along the post-switch trajectory), against a fixed
sampling std of **0.607** (`actor_log_std` init −0.5, unchanged by training). The response
directions are **seed-inconsistent** (seed 1 moves intrinsic/replay most; seed 5 ent/horizon;
redo-s1 lr/ent) — drift, not a learned switch-response. Weight norms grow ~3× from init
(0.039 → 0.09–0.11): *some* gradient flow happened, but nowhere near a policy.

**(c) Why:** the scout benchmark trains the Brain from scratch inside the scored run with
`brain_episodes=4`, `pretrain_episodes=0`, `brain_lr=1e-4` — roughly 160 Brain gradient steps.
The paper's §4.3 Table-2 Brain (avg success 0.7307 vs static 0.4638) was trained 40–50
episodes and then evaluated **frozen** — a different operating point the scout instrument
structurally cannot express.

**Consequences.** (i) The frozen control 0.5534 is *the inner learner under mid-bound HPs
jittered widely every 10 updates* — not a competent meta-controller. (ii) The three-family
null re-reads as "nothing helps when the controller is untrained noise" — every intervention
handed a better channel to a controller with no capacity to use any channel. (iii) The
paper-era Brain-vs-static comparison has never been checked on this instrument, at this
scorer, at n≥8 — and its static baseline's tuning status is unknown.

## Assumptions (explicit, with risk)

- **B1** The A1 probe's obs sweep covers the observation region the Brain actually visits.
  Risk: real visited obs may be more extreme; mitigated by the stylized post-switch
  trajectory probe and by D0 measuring behavior, not weights.
- **B2** Episode-mean ≈ policy behavior. Risk: within-episode swings could cancel in the
  mean — but (b) bounds the deterministic response directly from the checkpoint, closing
  this loophole.
- **B3** The 5 unlogged action dims (replay_prioritization, anchoring_weight, 8-d code)
  behave like the logged ones (probe (b) says yes: code-dim response ≤ 0.23, same scale).

## Pre-registered experiments (screen n≥8; one variant each, no multiplicity)

All three implemented 2026-07-07 on `autoresearch-run-20260706`, flag-guarded default-off,
committed BEFORE any screening results exist. Control comparison: LOOP-0004 frozen control
n=8 — composite **0.5534 [0.5478, 0.5588]**, hit80 **0.829 [0.815, 0.846]**.

### D0 — `brain_constant_action` (bottom rung: does the Brain add ANYTHING here?)
Severs the Brain: zero action every decision (= exact mid-bound HPs, zero code — the levels
the control empirically averages, minus the noise); Brain PPO updates skipped.
- **P-D0a (primary, predicted):** composite within the control CI at n=8 — the Brain's
  jitter neither helps nor hurts; the Brain adds nothing over tuned static HPs on this
  instrument. → The paper's Table-2 hierarchy does not replicate here; the program's
  null is over-determined; write-up + instrument discussion.
- **P-D0b:** composite > control CI-upper — the Brain's noise actively *hurts*; ditto but
  stronger (the "controller" is a handicap).
- **P-D0c:** composite < control CI-lo — the stochastic HP schedule is load-bearing
  (dithering-as-exploration): a real, surprising finding about WHY the Brain helps;
  characterize before any pivot.

### O1 — `brain_critic_lr_oracle` (oracle rung: is a *timed* critic damp worth anything?)
Ground-truth-timed critic-LR damp: on each detected regime switch the critic optimizer
group runs at main_lr × 0.5 for the detection update + 15 following, else tracks main
exactly. Upper-bounds ANY Brain-learned critic damp (the adaptive A2 test that cand 1's
static null could not answer).
- **P-O1 (predicted):** REJECT at n=8 (composite ≤ control mean, no hit80 win) — per
  reading ② (the post-switch critic re-fit is functional, not pathological) and the
  static cand-1 neutrality. A rejection kills A2 outright: no learned lever can beat a
  perfectly-timed one.
- If instead composite/hit80 clears the fork at n=8: first positive in three families →
  promoting `critic_lr_scale` to a Brain action dim acquires a guaranteed-achievable
  target — but ONLY worth doing on an instrument where the Brain can actually train
  (see Finding 1c).

### O2 — `brain_oracle_policy_swap` (ceiling: how much headroom does the instrument have?)
Diagnostic only, never a method: bank the full learner (model, world model, both Adam
states) per regime at switch-away; restore on revisit (6–7 of the run's 7 switches are
revisits). Measures the zero-forgetting ceiling of the composite.
- **P-O2 (predicted):** composite substantially above control (≥ +0.05) — the post-switch
  window mostly measures relearning time, which a zero-forgetting agent skips. The GAP
  (topline − control) is the instrument's true headroom (reading ①).
- If topline − control < ~0.02: the instrument has essentially no headroom above the
  frozen control — the three-family null was structurally inevitable, and the write-up
  can state it as a property of the instrument, not of the 25+ mechanisms.
- n=4 acceptable for a first ceiling estimate (it is not a candidate; no fork gate).

**Decision matrix (joint):** D0≈control + O1 null + large O2 gap → the instrument has
headroom but neither the untrained Brain nor timed critic damping reaches it → the binding
constraint is the controller's training budget (instrument pivot: trained-Brain eval
protocol, more brain_episodes, or pretraining) — NOT more inner mechanisms. D0≈control +
small O2 gap → no headroom; write up the null with confidence. D0c or O1 win → follow that
thread first.

## Surface note

D0/O2 touch `scripts/train_brain.py` + `agents/brain/meta_env.py` (+ `agents/ppo/train.py`
for O1/O2) — **control/diagnostic conditions, not mechanisms**, flag-guarded default-off,
user-approved 2026-07-07 ("implement the whole plan", session plan
`docs/plans/2026-07-07-session-plan-redo-endgame-oracle-rung.md`). Default-off paths are
byte-identical to baseline (tests: `tests/test_oracle_controls.py`, 13 tests; full suite
148 pass). Same commit batch fixes the results-push race (`scripts/cloud/push_results.sh`
retry-on-fresh-tip; the race silently dropped ~20 per-seed dirs across LOOP-0006/0007) and
makes `brain_neuromod/*` probes durable (`run_seed_sweep.py` → `probes/<run>_probes.json`
in the pushed sweep dir; the LOOP-0007 redo gate was probe-blind without this).

## Links

- Session plan: `docs/plans/2026-07-07-session-plan-redo-endgame-oracle-rung.md` (main tree)
- Interpretation layer: `docs/hand-offs/2026-07-07-three-family-null-strategic-inflection.md`
- Predecessor family + predictions: [note 0004](./0004-code-free-plasticity-stability.md)
- Paper §4.3 Table 2 (the founding Brain-vs-static claim this re-tests):
  `docs/references/Combating Catastrophic Forgetting.pdf`
- Instrument definition: `fast_switch_scout_v2` in
  `src/lifelong_learning/research/benchmarking.py` (`brain_episodes=4` — Finding 1c)

## Addendum (2026-07-07 ~19:05Z) — the recovered probe adjudicates A1: FALSE

The redo dormant-fraction probe (note 0004's pre-registered A1 adjudicator, recovered from
the box inner logs via the new `probes/` pipeline; all 10 seeds, 240 checks each):

- Dormant fraction is **highest at the first check (~0.92 actor / ~0.77 critic)** — a fresh
  random net measured on a narrow obs batch is mostly-inactive ReLUs — and **falls
  monotonically to ~0.35** by run end (Q1 0.88 → Q4 0.35 actor; 0.75 → 0.36 critic;
  seed-1 shown, all seeds match the max/mean signature).
- It never rises across the run or across regime switches. The accumulating-dormancy
  signature from many-regime, long-horizon continual RL (Sokar 2023, Lyle 2023) is
  **absent on this 2-regime fast-switch task**.
- Consequences: **A1 FALSE** — cands 2 (ReDo) and 3 (plasticity_norm) lose their premise;
  redo's marginal n=8 Path-B blip (which already failed at n=10) is expected to be noise;
  the pre-registered expectation for the W10 holdout is REJECT. Note also that early-run
  ReDo was re-initializing ~90% of head units every 50 updates with barely any composite
  effect — further evidence of how insensitive this instrument is to inner-head surgery.
- Caveat: the trace comes from redo-ON runs (the probe only logs when the mechanism is
  on), so the trajectory includes reset effects; but resets can only *lower* subsequent
  dormancy readings, so the absence of any rising trend is conservative.

## Outcome addendum (2026-07-08) — all pre-registered predictions resolved

D0: between P-D0a/P-D0b (composite 0.5608 ≥ control — Brain adds ≤0; jitter bought ~4 pts
hit80). O1: P-O1 confirmed (0.5451/0.8058, REJECT — A2 dead). O2: P-O2 confirmed at scale
(0.8022 n=8 — +0.249 headroom; "no headroom" branch dead). The decision-matrix outcome and
the full axis, plus the T1-series matched trained-Brain contrast (+0.0292, p<0.01, n=32/arm),
are consolidated in [note 0006](./0006-controls-axis-thesis-relocated.md).
