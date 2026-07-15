# 0013 — Learned WHEN works, learned WHICH doesn't (yet): the G3 de-oracling result

**Status:** ✅ RESOLVED 2026-07-14 (LOOP-0012 screen; gates pre-registered in
[research-log 0009](../research-log/0009-2026-07-14-g3-head-bank-preregistration.md) before results).
**Owner:** Elijah · **Relates to:** [note 0012](./0012-ceiling-decomposition.md) (the heads
verdict this builds on) · [LOOP-0012](../autoresearch-loops/LOOP-0012-g3-head-bank.md) (ops) ·
[note 0009](./0009-memory-levers-preregistration.md) (the mem-Brain draft this informs).

## Question (as registered)

Can the two oracle ingredients of the heads-swap ceiling slice — the trigger (WHEN to swap)
and the selection (WHICH slot) — be replaced by learned mechanisms?

## Results (n=8/arm, eval protocol, references archived; 24/24 evals clean)

| arm | trigger | selection | composite | gain vs control (0.6202) | % of heads slice (+0.2180) |
| --- | --- | --- | --- | --- | --- |
| g3_oracle | oracle | oracle | 0.8409 | +0.2206 (p=1.3e-7) | **101%** |
| g3_ar1 | **learned** | K=2 flip | 0.7383 | **+0.1180 (p=1.6e-4)** | **54%** |
| g3_ar1ve | learned | **learned (value-error)** | 0.6218 | +0.0016 (p=0.95) | 0.7% |

Trigger stats in-run (logged fires vs true schedule): ar1 precision 0.70 / recall 0.75;
ar1ve precision 0.84 / recall 0.88. ar1's hit80 = 1.000 (every switch recovered to 80%).

- **P-G3a PASS:** the head bank is implementation-equivalent to the validated heads swap
  (vs heads arm: Δ+0.0026, p=0.84). All learned-arm deltas are attributable to
  trigger/selection, not mechanism.
- **P-G3b PASS (A-R1 premise ALIVE):** a value-loss change-point detector with no oracle
  information recovers **54% of the ceiling slice**. The trigger cost (~0.10 composite vs
  oracle timing) traces to in-run precision dropping to 0.70 — the bank's own flips perturb
  the value-loss stream, and each false fire at K=2 loads the *wrong* head mid-regime.
- **P-G3c FAIL (registered kill for this selector):** content-addressable selection by
  **critic value fit** carries ~no regime information here. Interpretation: the regimes are
  visually identical (no cue in obs) and differ only in *which goal pays*; immediately
  post-switch the rollout is failure-heavy under either head's value model, so value-error
  is uninformative-to-misleading. This echoes the program's older finding that the Brain's
  obs stream contains switch *timing* but not regime *identity*.

## Where the identity signal actually is (assumption for the next rung, stated)

The regimes differ **only in the reward function** — so the discriminative statistic is
reward-prediction error, not value fit: bank the world model's tiny reward head
(Linear(256,1), 257 params) per slot as a **regime classifier** and select the slot whose
banked reward head best predicts the fresh (s,a)→r transitions. This is A-R2 retargeted by
the note-0012 decomposition (the WM is useless as *memory* but its reward head may be
exactly right as an *address*). Risks, stated: post-switch reward sparsity may starve the
classifier the same way it starved value fit; and the WM latent the reward head reads is
itself continually trained (drift between banking and scoring).

## Consequences

- **The learned-O2 method exists** at K=2 with degenerate selection: +0.118, half the
  oracle slice, zero oracle bits, zero Brain retraining. Paper method-section candidate as
  is; the remaining ~0.10 is the trigger-precision and selection frontier.
- **Three competing next rungs (strategy fork, not pre-decided by the tree):**
  (a) trigger hardening — hysteresis / flip-back-on-no-improvement to push precision 0.70 →
  0.85+ at K=2; (b) **reward-head-classifier selection** (above) — unlocks K>2 and
  content-addressing; (c) Brain-in-the-loop selection/trigger (the LOOP-0010 mem-Brain
  levers, needs Brain fine-tunes = box). Recommendation: (b) first ($0, small build), it
  subsumes (a)'s failure mode analysis and gates (c) honestly.
- LOOP-0010's restore-gate lever premise survives P-G3b; its *selection* lever design must
  not assume value-error addressing (this note's negative).

## Links

Raw: `evals/g3_*`, `evals/wave1_scores.json` (local) · adjudicator `scripts/score_wave1.py g3` ·
pre-reg [log 0009](../research-log/0009-2026-07-14-g3-head-bank-preregistration.md) ·
ops [LOOP-0012](../autoresearch-loops/LOOP-0012-g3-head-bank.md).
