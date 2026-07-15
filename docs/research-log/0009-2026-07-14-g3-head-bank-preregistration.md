# 0009 — 2026-07-14 — G3 head-bank de-oracling screen: pre-registration (LOOP-0012)

**Registered BEFORE any results exist.** Follows the Wave-1 routing verdict
([note 0012](../research-notes/0012-ceiling-decomposition.md): heads carry ~90% of the
ceiling → W2A). Campaign note: [LOOP-0012](../autoresearch-loops/LOOP-0012-g3-head-bank.md).

## Method under test

The **head bank**: K persistent weight slots for the inner learner's actor/critic heads
(`train.py`, flag-guarded default-off; weights-only, no optimizer state — the ladder measured
optimizer share ~4%). The de-oracling ladder replaces the O2 oracle's two free ingredients one
at a time:

| arm | trigger (WHEN) | selection (WHICH) | what it isolates |
| --- | --- | --- | --- |
| `g3_oracle` | ground-truth switch | ground-truth regime id | implementation equivalence vs the ladder's heads swap |
| `g3_ar1` | **learned** (value-loss change-point, threshold 1.0) | K=2 flip ("other") | A-R1: the trigger alone |
| `g3_ar1ve` | learned | **learned** (banked-critic value error on the fresh rollout) | the fully learned method |

**Trigger calibration (desk, disjoint data):** threshold 1.0 chosen on the archived control
arm's value-loss traces — precision 0.84, recall 0.85, mean lag 0.8 updates, cooldown 8
(`scripts/calibrate_surprise_trigger.py`). The exploration-spike machinery (cand-4) stays
decoupled — the bank has its own detector state.

## Design

n=8 eval seeds (1–8)/arm, LOOP-0009 seed-1 Brain, eval protocol, home 5070, $0
(24 evals; driver `scripts/run_wave1_evals.py g3`). References are archived, same
protocol/Brain/seeds: control = `loop9_s1_model_e1..8` (no bank), ceiling slice =
`wave1_decomp_heads_e1..8` (oracle heads swap, gain +0.2180). Scored + adjudicated by
`scripts/score_wave1.py g3`; in-run trigger precision/recall computed from the logged
`head_bank_trigger_fired` steps vs the true switch schedule (window 4 updates).

## Pre-registered gates

- **P-G3a (equivalence):** `g3_oracle` is statistically indistinguishable from the heads-swap
  arm (Welch p>0.05) AND captures ≥80% of the heads gain vs control. Failure reading: the
  bank implementation differs from the validated swap mechanism — fix before interpreting
  the learned arms.
- **P-G3b (A-R1, the tree's registered kill):** `g3_ar1` gain vs control ≥ **+0.05** AND
  in-run trigger precision ≥ **0.60**. If either fails: the inner surprise signal cannot
  recover the trigger even with free selection — no Brain gate at ~10× coarser cadence can,
  and the mem-Brain trigger premise is dead (LOOP-0010's restore-gate lever dies with it).
- **P-G3c (selection):** `g3_ar1ve` ≥ `g3_ar1` − 0.02 composite (content-addressable
  selection at least matches the K=2 flip; false-positive fires should self-correct).
- **Prediction (stated):** `g3_ar1` captures ≥ half of the heads gain (≥ +0.10); `g3_ar1ve`
  matches or beats it. A false K=2 flip loads the wrong head mid-regime but is repaired at
  the next true switch; with precision 0.84 we expect ~1 wrong-head episode per run.
- **Extension rule:** any gate whose deciding quantity lands within ±0.02 composite (or
  ±5pp precision) of its threshold → extend that arm to n=16 (seeds 9–16) before
  adjudicating; convergence-vs-decay applies.

## What this feeds

P-G3b+c pass ⇒ the **learned-O2 method result** exists (headline: a self-contained
mechanism, no Brain retraining, recovers most of the forgetting ceiling) and LOOP-0010's
mem-Brain levers get their premise. Box need appears only after this: Brain fine-tunes on
the new interface (~4×BR50 ≈ 1.5 box-nights) — user ping at that point.
