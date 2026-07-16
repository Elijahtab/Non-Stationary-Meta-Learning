# 0013 — 2026-07-16 — Drift-robust fingerprint selection: pre-registration (LOOP-0015)

**Registered BEFORE any results exist.** User green-light 2026-07-16: run the drift-robust
fingerprint rung, $0/home only. Context: LOOP-0013 adjudicated content-addressed selection
as drift-bound (banked heads scored through the live trunk almost never restore — 1/55
fires), and LOOP-0014 identified false-fire churn as the cost that faster triggers cannot
beat. This rung attacks both with one mechanism.

## Mechanism (committed pre-launch, `head_bank_select="reward_fp"`)

Each slot banks its **entire world model** (~1.5M params, scoring-only — never loaded into
the live learner) as a regime fingerprint, so fingerprints are scored in their **own frozen
feature space** — removing exactly the drift bias LOOP-0013 identified. On a trigger fire,
candidate slots are scored by reward-prediction MSE on the freshest (s,a)→r transitions:

- **banked slots** through their own banked WMs (a scratch module; the live WM is untouched);
- **the active slot** through the LIVE WM — deliberate asymmetry: the live model *is* the
  "no switch happened" hypothesis, so at K=2 the selector is a **fire verifier**: flip on
  true switches, stay on false fires (the LOOP-0014 churn suppressor).

Trigger and everything else identical to the method arm (per-update value loss, threshold
1.0, cooldown set on every fire including suppressed ones): any delta vs `g3_ar1` is
attributable to selection behavior. Restores load **policy heads only** (the decomposition's
WM-restore null stands; fingerprints are classifiers, not memory).

## Arm (8 evals, home 5070, ~1.1 h, $0)

`g3_fp` × eval seeds 1–8. References archived: control (n=16), `g3_ar1` (n=16, +0.1188,
the blind-flip method), `g3_ar1re` (−0.0092, the drift-bound null being fixed),
`wave1_decomp_heads` (oracle slice). Scored by `scripts/score_wave1.py fp`.

## Pre-registered gates

- **P-FP1 (selection engages and verifies):** of fires inside the TP window (≤8,192 steps
  after a true switch), ≥ 75% flip; of fires outside it, ≥ 75% stay. (The drift-bound arm
  managed a 2% overall restore rate; the blind flip flips 100% indiscriminately.)
- **P-FP2 (non-inferiority):** gain ≥ gain(ar1, n=16) − 0.02, i.e. ≥ **+0.0988**. Passing
  means content addressing is unlocked (K>2 viable, correct-slot selection for free) at no
  composite cost.
- **Secondary, no gate — the suppression dividend:** gain ≥ ar1 + 0.02 (+0.1388) would
  demonstrate that verifying fires pays on top of unlocking selection (~31% of ar1's fires
  are false and each false flip costs wrong-head time the verifier should refuse).
- **Readings.** FP1+FP2 pass → the drift diagnosis was correct and complete; the learned-O2
  method gains honest WHICH; the K>2 door opens. FP1 pass / FP2 fail → the verifier
  classifies correctly but selection timing/overhead costs composite — report the anatomy.
  FP1 fail → even own-feature-space fingerprints cannot classify regimes from 2,048-step
  windows; content addressing is dead on this instrument at any drift treatment, and the
  method remains the blind flip.
- **Extension rule:** deciding quantities within ±0.02 composite (or ±5 pp of a behavior
  bar) → extend to n=16 before adjudicating.

## Registered risks

Reward sparsity within a single 2,048-step window may starve the MSE contrast at some
fires (the same risk v1 carried; now each candidate at least reads its own features);
the banked-WM fingerprint was trained under its regime's *competent* policy while fresh
post-switch data comes from the *wrong* policy's state distribution — reward prediction is
state-based so this should transfer, but it is an assumption, stated.
