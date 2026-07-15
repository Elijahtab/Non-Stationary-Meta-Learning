# 0013 — Learned WHEN works, learned WHICH doesn't (yet): the G3 de-oracling result

**Status:** ✅ RESOLVED 2026-07-14 — **with a same-day correction: the P-G3c "selection null"
was an implementation artifact, not a science result.** The value-error arm ran without a
slot-allocation rule, so the bank never contained a second slot and the selector was never
exercised (its exactly-at-control composite, +0.0016, is the fingerprint of
bank-without-restore). The gate verdict FAIL stands as recorded, but §Interpretation's claim
that "value fit carries no regime identity" is **withdrawn as untested**. Both
content-addressable selectors re-run properly (spawn-until-full allocation) in
[LOOP-0013](../autoresearch-loops/LOOP-0013-selection-rung.md) /
[log 0010](../research-log/0010-2026-07-14-selection-rung-preregistration.md).
The A-R1 trigger result (P-G3b) is unaffected — its "other" selection needs no bank lookup.
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
- **P-G3c FAIL — ⚠ corrected same day: artifact, not science.** The arm's selector could
  only choose among *banked* slots, and without an allocation rule the bank never grew past
  the active slot — every fire re-selected slot 0 and the arm degenerated to
  bank-without-restore (hence composite == control to 3 decimals). The original
  interpretation ("value fit carries no regime identity") is withdrawn as untested; the
  visual-identity/reward-only reasoning survives as a *hypothesis* the LOOP-0013 re-run
  tests head-to-head against the reward-fingerprint selector.

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

## LOOP-0013 re-run results (2026-07-15, allocation fixed — the honest adjudication)

| arm | n | gain vs control | p | trigger prec/rec | restore rate |
| --- | --- | --- | --- | --- | --- |
| ar1 (K=2 flip) | **16** | **+0.1188** | 4.7e-7 | 0.69 / 0.73 | 1.00 (forced) |
| ar1re (reward fingerprint) | 8 | −0.0092 | 0.71 | 0.82 / 0.80 | **0.02** (1/55 fires) |
| ar1ve2 (value fit) | 8 | +0.0074 | 0.64 | 0.82 / 0.80 | 0.16 (9/55) |

- **P-G3e PASS:** the learned-trigger headline converges — n=8 +0.1180 → n=16 **+0.1188**
  (52% of the +0.228 slice at n=16 references). This is now the program's method number.
- **P-G3d and P-G3d-ve both FAIL — real nulls this time,** and the mechanism is identified:
  with working triggers (0.82 precision), both selectors restored almost never. Banked heads
  are scored through the **live, continually-trained trunk** (WM trunk for reward
  fingerprints, model encoder for value fit); by the first revisit the feature space has
  drifted for ~49 updates, so the stale banked head loses the error comparison to the
  always-fresh live head — selection collapses to "stay" and the arm behaves like control.
  The registered trunk-drift risk (log 0010) is the binding one; the reward-vs-value
  hypothesis remains untested *between* the selectors because drift dominates both.
- **Consequence — the selection frontier is now precisely "drift":** the candidate fixes are
  (i) drift-robust fingerprints: bank the *scoring path* per slot (the full ~1–2 MB WM, or
  frozen feature hooks) so fingerprints are evaluated in their own feature space; or
  (ii) hand selection to the mem-Brain lever (LOOP-0010, needs Brain fine-tunes = box).
  At K=2 the flip needs no selection, so the *method* stands at +0.119 regardless; selection
  matters for K>2 and for false-fire robustness.
- **Trigger decomposition:** oracle/oracle = 101% of slice vs learned trigger = 52% — the
  entire gap is trigger cost (false fires load the wrong head; misses delay restore).
  Trigger hardening at threshold 1.5 (desk: precision 0.95, recall 0.81) is registered as
  the next $0 arm ([log 0011](../research-log/0011-2026-07-15-trigger-hardening-preregistration.md)).

## Links

Raw: `evals/g3_*`, `evals/wave1_scores.json` (local) · adjudicator `scripts/score_wave1.py g3` ·
pre-reg [log 0009](../research-log/0009-2026-07-14-g3-head-bank-preregistration.md) ·
ops [LOOP-0012](../autoresearch-loops/LOOP-0012-g3-head-bank.md).
