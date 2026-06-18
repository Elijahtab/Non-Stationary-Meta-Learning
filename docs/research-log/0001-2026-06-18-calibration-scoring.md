# 0001 — `calib_signal` null result + composite-score redesign

- **Date:** 2026-06-18
- **Status:** Proposed (awaiting confirmation before implementing)
- **Touches:** `src/lifelong_learning/research/benchmarking.py` (**immutable_surface** — human
  infra change, invalidates all cached baselines/fingerprints); a new harder calibration preset
  (8×8). No autoresearch-agent-editable files.

## Context

The `calib_signal` sweep ([`sweeps/calib_signal/`](../../sweeps/calib_signal/)) finished
2026-06-18 — all **12/12 cells** scored (4 conditions × 3 seeds), exit 0:
`brain_no_neuromod`, `brain_neuromod`, `brain_random_code`, `brain_oracle_code`.
Preset `calib5x5`: `MiniGrid-MultiGoal-5x5-v0`, 2 regimes, 400k inner timesteps,
`steps_per_regime=100k`, 30 brain episodes, `reward_mode=recovery`.

Purpose of the sweep: a **signal-detection calibration** — *can the frozen benchmark tell the
neuromodulation conditions apart at all?* before we point the autoresearch loop at it.

**Result — the conditions are statistically indistinguishable on composite score:**

| Condition | Composite (mean) | 95% CI | post-switch success | neuromod activity |
|---|---|---|---|---|
| `brain_random_code` | 0.7565 | [0.739, 0.767] | 0.638 | 0.77 |
| `brain_neuromod` | 0.7556 | [0.753, 0.758] | 0.640 | 0.60 |
| `brain_oracle_code` | 0.7534 | [0.745, 0.759] | 0.635 | 0.43 |
| `brain_no_neuromod` | 0.7397 | [0.728, 0.756] | 0.625 | — |

Two diagnoses, both grounded in the scorer
([`_compute_composite_score`](../../src/lifelong_learning/research/benchmarking.py#L643),
weights `0.5·post_switch + 0.25·hit_rate_80 + 0.25·norm_steps_to_80`):

1. **Ceiling / no headroom at 5×5.** `hit_rate_80 ≈ 0.99` for every condition (25% of the score
   is a dead constant); `steps_to_80 ≈ 25k–29k` against a 100k regime → `norm_steps_to_80 ≈ 0.75`
   for everyone. The only term carrying signal is the 50% post-switch term, spanning just
   0.625→0.640 — smaller than seed noise.
2. **The saturated terms actively distort the ranking.** `brain_neuromod` has the *best*
   post-switch success (0.640) yet `brain_random_code` *wins* the composite (0.7565), purely on
   a slightly better (noisy) `steps_to_80`. The thing we optimize for lost to random because of
   near-constant terms.

The mechanism is *not* dead: `neuromod_activity` (oracle 0.43 < neuromod 0.60 < random 0.77) and
`post_switch_policy_kl` (oracle 0.0034 < neuromod 0.0072 < random 0.011) separate cleanly. The
mask responds to the context code as designed — there's just no task pressure for it to matter at
this difficulty.

## Decision

Two complementary changes, made in **one pass**, then **re-baseline once**:

1. **Difficulty bump — new `calib8x8` preset.** Move to `MiniGrid-MultiGoal-8x8-v0` (matches the
   paper's final-run scale) and/or shorten `steps_per_regime` / add regimes, to de-saturate the
   recovery terms and open headroom in the post-switch transient.
2. **Composite-score redesign — pure post-switch for calibration.** Set
   `composite_score = mean_post_switch_window_success_rate` (drop `hit_rate_80` and
   `steps_to_80`) for the calibration phase.

   Reintroduce a **95-threshold reliability/speed facet** (`hit_rate_95` + `steps_to_95`) only for
   the **final scored benchmark**, not calibration.

## Rationale

- **The window average already encodes speed.**
  [`summarize_post_switch_success`](../../src/lifelong_learning/research/benchmarking.py#L270)
  averages success over `[switch+500, switch+500+0.5·steps_per_regime]` — the entire first half
  of the post-switch regime. A fast recoverer accumulates high-success points early → high mean;
  a slow one is dragged down. So `steps_to_X` is **largely redundant** with the window term, and
  `hit_rate_X` is a binary reliability floor. Dropping the threshold terms loses no speed
  dimension.
- **Calibration's job is detection, not robustness.** Maximum power, minimum noise → a single
  on-target metric is the cleanest instrument. The threshold terms are saturated *and* already
  flipped the ranking (see Context). 80→95 *now* would keep partly-redundant terms and extra
  variance in exactly the phase that needs neither — hence "right idea, one phase too early."
- **`hit_rate_95`/`steps_to_95` belong in the final benchmark**, where cross-switch reliability
  ("did it sustain ≥95% on *each* switch") matters more than detection power. Note these already
  *spread* in the 5×5 data (`hit_rate_95`: 0.83→0.89), confirming they carry signal once promoted.
- **Constraint:** `benchmarking.py` is in `research_manifest.toml` `[immutable_surface]`. Changing
  the scorer is a legitimate *human* infra decision, but it invalidates every cached baseline and
  fingerprint — all post-change composite numbers are incomparable to prior ones. Re-baseline
  before any autoresearch session.

### Open question being decided in this entry

80→95 **vs** drop the two threshold terms entirely for the calibration run.
**Recommendation: drop them for now** (pure post-switch), per the rationale above.

## Outcome

_(To be filled after the re-baselined `calib8x8` sweep.)_ Success criterion: the ordering
`oracle ≥ neuromod > random > no_neuromod` emerges with non-overlapping CIs — i.e. the benchmark
now has discriminative power and is safe to hand to the autoresearch loop.

## Follow-ups

- [x] Confirm scoring choice — **pure post-switch** chosen.
- [x] Implement scorer change (composite = `mean_post_switch_window_success_rate`, removed
      `_normalize_recovery_steps`) + `calib8x8` preset; updated `tests/test_benchmarking.py`
      (composite now 0.8875). Full manifest suite: 58 passed. Spec doc 07 updated to match.
- [~] **Pilot in flight:** `sweeps/calib8x8_pilot` — single cell `calib8x8_brain_neuromod_seed0`
      (run dir `runs/calib8x8_brain_neuromod_seed0_20260618-104758`). Validates that 8×8
      de-saturates the scorer *and* measures per-cell wall-clock + core usage to size the cloud box.
- [ ] If pilot looks good → run full 12-cell matrix **on cloud** (CPU-bound; many cores is the
      lever). NOTE: `calib8x8` uses `inner_num_envs=16`, so budget ~16 cores/cell
      (`CORES_PER_CELL=16` → `max_parallel ≈ nproc/16`), **not** the doc's default 8 (which was for
      the 5×5 `brain_num_envs=1` presets). GPU is not the constraint (~0.5 GB/stream observed).
- [ ] Re-baseline; invalidate stale fingerprints. Record sweep outcome in the Outcome section above.
- [ ] If signal confirmed (`oracle ≥ neuromod > random > no_neuromod`, non-overlapping CIs): define
      the final-benchmark composite (post-switch + 0.95 reliability/speed facet) as a separate record.
