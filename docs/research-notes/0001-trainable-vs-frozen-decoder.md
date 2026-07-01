# 0001 — Trainable vs. Frozen Neuromodulation Decoder

**Status:** experiment ready (code landed 2026-06-30) · awaiting sweep results
**Owner:** Elijah · **Relates to:** [plan §3a](../plans/workshop-task-free-neuromodulation.md),
[spec 05](../spec/05-neuromodulation.md), [box diagram](../spec/neuromodulation-box-diagram.md)

## Hypothesis (one sentence)

> Letting the neuromodulation **decoder train** — so the inner agent *learns to interpret* the
> Brain's 8-D context code instead of reading it through a fixed random projection — will
> improve post-switch recovery over the frozen-decoder configuration used in the paper.

## Background: the finding this rests on

The paper's neuromodulation is sold as *learned, context-dependent routing*. But we found (and
verified by gradient check, 2026-06-30) that **only half of it is learned**:

- The Brain **does** learn the 8-D context code (its action).
- The **decoder** that expands those 8 numbers into the ~4,096-value feature mask is **frozen at
  random initialization** — `set_context_code` decodes under `no_grad` and the mask enters the
  forward pass as a detached buffer, so no gradient ever reaches the decoder's ~1M weights.
  Confirmed: `decoder.weight.grad is None` after an inner backward (frozen), non-`None` and
  drifting (trainable). See [spec 05](../spec/05-neuromodulation.md).

So in the paper regime the Brain must steer through a **fixed random cipher** it cannot reshape;
it can only pick codes that *happen* to expand into useful masks. This is reservoir-like:
functional, but capped by whatever random expansion it was born with.

**Assumption (A1):** the frozen-decoder behavior held in the paper's neuromod runs, not just
current code. The forward path (`set_context_code` under `no_grad` → detached buffer → `forward`)
is the core design, unchanged, so this is very likely — but worth a one-line confirm against the
paper-era commit before making the claim in print.

## The change (what "trainable" does)

A flag-guarded path (`trainable_neuromod`, default **off** = frozen = paper-faithful):

- **Frozen (default):** unchanged. Mask is the detached snapshot buffer; decoder never trains.
- **Trainable:** the mask is **re-decoded in-graph** each forward (`active_mask()`), so the inner
  PPO gradient reaches the decoder and the inner Adam optimizer updates it alongside the
  encoder/heads. The **code stays detached**, so *only the decoder learns to interpret the code* —
  the Brain is unaffected and still trained solely by its meta-reward.

Mechanically this is two learners co-adapting toward a **communication protocol**: the Brain
learns *what to say*, the inner agent learns *what to do when it hears it*. See
[plan §3a](../plans/workshop-task-free-neuromodulation.md).

Implementation detail (disclose in paper): with `trainable`, the distillation **anchor model**
runs unmodulated (its non-persistent `current_code` is never set), whereas frozen mode anchors on
the snapshot mask. This affects only the anchoring penalty when `anchoring_weight > 0` (a minor
regularizer); we consider it negligible but note it for honesty.

## What we hope will occur (predictions)

- **H1 (primary):** trainable ≥ frozen on `composite_score` (post-switch window success),
  ideally by a margin that survives multiple seeds. Rationale: removing the random-cipher
  bottleneck lets the mask become genuinely useful rather than accidentally useful.
- **H2 (mechanism):** the **regime-decoding probe** (context code → hidden regime, see
  `scripts/analyze_regime_decoding.py`) reads *higher* for trainable, and/or the decoded **mask**
  becomes more regime-separable — because a trained decoder can organize the code→mask map so
  regimes map to distinct, reusable masks.
- **H3 (dynamics):** `brain_neuromod/decoder_weight_norm` drifts under trainable (confirmed in a
  pilot: +0.08 in one short episode) and is flat under frozen. This is the "did the manipulation
  take" check, not a result.
- **H4 (closing the oracle gap):** trainable should move toward the `brain_oracle_code` upper
  bound — if a *learned* code + *learned* decoder approaches oracle performance, that's strong
  evidence the mechanism can carry real regime information when both halves adapt.

## How we'll measure it

- **Conditions (head-to-head):** `brain_neuromod` (frozen) vs `brain_neuromod_trainable`, plus the
  existing `brain_no_neuromod`, `brain_random_code`, `brain_oracle_code` for context.
- **Runner:** `scripts/run_seed_sweep.py --preset <calib8x8|scout5x5> --conditions brain_neuromod
  brain_neuromod_trainable --seeds 0 1 2 3 4` → per-condition mean ± bootstrap 95% CI.
- **Primary metric:** `composite_score`. **Secondary:** BWT/FWT, the decoding probe (H2),
  decoder-norm drift (H3), and distance to oracle (H4).
- **Scale note:** the 5×5 calibration saturated the scorer (everyone recovers; see
  [research-log 0001](../research-log/0001-2026-06-18-calibration-scoring.md)); prefer `calib8x8`
  so conditions can actually separate.

## Possible outcomes and how we'd read each (all publishable)

1. **Trainable clearly wins.** Cleanest story: "neuromodulation was throttled by a frozen decoder;
   unfreezing it — letting the inner agent learn to interpret the controller's signal — recovers
   the gains." Frozen becomes the baseline the paper improves on.
2. **No difference.** Then the mask family itself (suppress-only, single bottleneck) is the limit,
   not trainability → motivates the affine/gain mask and multi-site variants; and it *sharpens*
   the negative result ("even fully learned, this gating contributes little here").
3. **Trainable is worse / unstable.** Interesting in its own right: two co-adapting learners can
   chase a moving target. Report the instability, and test stabilizers (lower/separate decoder LR,
   slower Brain, warm-up) — a real finding about controller↔learner co-training.

## Risks / watch-items

- **Two moving parts (R1):** decoder trains at the inner LR, which the Brain also controls via
  lever 0 — coupling that could destabilize. If unstable, give the decoder its own (smaller) LR.
- **Confound (R2):** trainable adds ~1M trainable params to the inner net. A fair "does
  *interpretation* help" claim may want a param-matched control (e.g. a trainable decoder fed a
  *fixed* code) — consider adding if H1 is positive.
- **Seed noise (R3):** the frozen neuromod-vs-no-neuromod gap was single-seed and small; hold
  trainable to the same ≥5-seed + CI bar before claiming anything.
- **A1 above:** confirm the paper-era runs were frozen before writing "the paper's decoder was
  frozen."

## Connection to the paper

This turns the paper-era result into a **named baseline** ("frozen decoder") that the workshop
paper *explains* (the gradient wall) and *improves on* (learned interpretation), and reframes the
contribution as **emergent controller↔learner communication for task-free continual RL** — a
sharper, more novel hook than "we added neuromodulation." Whatever H1's sign, the decoding probe
(H2) + oracle gap (H4) give a mechanism story independent of the headline number.

---
_Update this note with results once the sweep lands (fill in H1–H4 outcomes and pick the outcome
branch above)._
