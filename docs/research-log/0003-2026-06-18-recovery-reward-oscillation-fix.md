# 0003 — `recovery` reward farms oscillation; add `recovery_v2`

- **Date:** 2026-06-18
- **Status:** Accepted — implemented; validation pending the `calib8x8` sweep ([0002](./0002-2026-06-18-calib8x8-cloud-sweep-plan.md))
- **Touches:** `src/lifelong_learning/agents/brain/meta_env.py` (new `reward_mode`),
  `scripts/train_brain.py` (CLI choices), `scripts/run_seed_sweep.py` (`calib8x8` now uses it),
  `tests/test_meta_env.py`. **Not** the autoresearch editable surface; benchmark/scorer untouched.

## Context

While reviewing whether the `recovery` meta-reward is well-designed, we found a reward-hacking
pathology and confirmed it bites in real runs (including the headline paper run). The reward
([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py), evaluated each Brain step):

```
recovery        =  max(0, s − s_prev) · 5.0     # dominant coefficient; ASYMMETRIC
maintenance     =  s · 0.3
urgency_penalty = −max(0, 0.8 − s) · 3.0
failure_penalty = −failure_rate · 0.5
```

**The bug:** `recovery` rewards upswings at `5·Δs` but clamps downswings to 0 (free). So:

1. **It rewards noise.** `success_rate` is noisy; `E[max(0, noise)] > 0` and grows with variance.
   At equal mean success, a *jerkier* trajectory earns more `recovery` than a smooth one — the
   reward prefers instability, the opposite of forgetting-resistance.
2. **It prefers sawtooth to stability.** Oscillating `0.5↔0.9` out-earns holding `0.9` (~3×, even
   after the urgency penalty); deeper dips earn more.

### Evidence (diagnostic on real runs, subsampled to reward cadence)

`brain_2_regimes_8x8_neuromod_20260315-180839` (the 0.7498 paper run, `decision_interval=1`,
390 Brain steps, only **7** regime switches):

| metric | value | reading |
| --- | --- | --- |
| net success progress | +0.89 | the genuine learning over the run |
| gross upswing (rewarded 5×) | 10.55 | ⇒ ~92% of `recovery` reward is **not** net progress |
| oscillation ratio (gross_up / \|net\|) | 11.9× | |
| within-regime downswing | 7.9 of 9.66 | drops are mostly instability, **not** the 7 switches |
| step-to-step reversals | 40% | constant direction-flipping (monotonic recovery ⇒ ~0%) |

`calib5x5_brain_neuromod_seed0` shows the same shape (osc ratio 3.9×, 45% reversals). So the Brain
trains on a signal that is **~75–92% noise**, not the post-switch recovery the term is named for.
This plausibly contributes to the [0001](./0001-2026-06-18-calibration-scoring.md) null result: a
controller fed a mostly-noise reward won't produce a clean neuromodulation signal to detect.

## Decision

Add **`reward_mode="recovery_v2"`** (additive — `recovery` is preserved for A/B and paper
comparability) and point `calib8x8` at it:

```
delta_sr        =  s − s_prev
in_post_switch  =  steps_since_switch ≤ post_switch_window_steps      # ~0.5·regime, in Brain steps
progress_weight =  5.0 if in_post_switch else 1.5
progress        =  delta_sr · progress_weight     # SYMMETRIC (drops cost what gains pay)
maintenance     =  s · 0.5
failure_penalty = −failure_rate · 0.5
reward          =  progress + maintenance + failure_penalty
```

## Rationale

- **Symmetric progress = potential-based shaping.** A symmetric `Δs` telescopes to net change
  `s_end − s_start` over any path, so up/down noise nets to zero and oscillation can't be farmed.
  It also *penalizes* within-regime forgetting instead of leaving it free.
- **Switch-gated weighting, not an asymmetric bonus.** Fast recovery is incentivized by
  up-weighting the *same symmetric* progress (5.0 vs 1.5) inside a post-switch window — so the
  window rewards climbing back **and** punishes re-forgetting, without reopening the farm. The
  window mirrors the scorer's `0.5·steps_per_regime`, converted to Brain steps via
  `inner_num_envs · inner_num_steps · decision_interval`.
- **Maintenance anchors the optimum.** Telescoping progress alone is indifferent to absolute
  level; the `s·0.5` term makes "reach high *and hold it*" optimal (rewards area under success),
  which also breaks path-indifference in favor of *fast* recovery.
- **Dropped the 0.8 urgency kink** — arbitrary threshold; the maintenance slope provides a smooth
  low-success opportunity cost instead.

### Tests

`tests/test_meta_env.py`:
- `test_recovery_v2_progress_is_symmetric` — an up-then-down oscillation leaves only maintenance;
  progress contributions cancel (the core anti-farming property).
- `test_recovery_v2_switch_gates_progress_weight` — weight is 1.5 outside the window, 5.0 the step
  a switch is detected.

Full suite green (meta_env + benchmarking + train_brain_cli: 34 passed).

## Outcome

_(To be filled after `calib8x8`.)_ Expect: lower success-rate variance / reversal rate vs the
`recovery` runs, and — the real test — a cleaner neuromodulation signal so conditions separate
(`oracle ≥ neuromod > random > no_neuromod`). Re-run the oscillation diagnostic on a `recovery_v2`
run to confirm the osc ratio collapses toward ~1.

## Follow-ups

- [x] Implement `recovery_v2` + CLI choice + tests; point `calib8x8` at it.
- [ ] Relaunch the local pilot on `recovery_v2` (the in-flight pilot used the old `recovery`).
- [ ] After the sweep: re-run the oscillation diagnostic on a `recovery_v2` run; record osc ratio.
- [x] Add the reward A/B as a sweep condition: `brain_neuromod_recovery` (neuromod on old
      `recovery`) vs `brain_neuromod` (on `recovery_v2`) — matched control, same env/di. The pilot's
      oscillation comparison was confounded, so this sweep cell is the clean reward test.
- [ ] If `recovery_v2` clearly wins, consider making it the default reward mode.
