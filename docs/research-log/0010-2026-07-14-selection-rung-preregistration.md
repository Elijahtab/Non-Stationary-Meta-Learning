# 0010 — 2026-07-14 — Selection rung re-run + A-R1 extension: pre-registration (LOOP-0013)

**Registered BEFORE any results exist.** Context: LOOP-0012's P-G3c arm is invalidated as an
implementation artifact (no slot allocation → selector never exercised; correction recorded
in [note 0013](../research-notes/0013-learned-trigger-selection-gap.md)). This rung re-runs
content-addressable selection properly and extends the A-R1 headline to n=16.

## Mechanism changes under test (committed pre-launch)

1. **Spawn-until-full allocation** (both content selectors): on a trigger fire, if any slot
   is unused, allocate the lowest unused slot; content addressing begins once all K slots
   hold fingerprints. Registered risk: a false fire during the fill phase burns a slot on a
   duplicate regime (bounded at K=2 by head similarity).
2. **Reward-fingerprint selector** (`head_bank_select="reward_error"`): each slot banks the
   WM reward head (257 params); selection = argmin banked-reward-head MSE on the freshest
   (s,a)→r transitions, scored through the live WM trunk. In this mode a restore also loads
   the slot's reward head (keeps fingerprints coherent; incidentally fixes the stale-dream
   channel for the returning regime). Registered risks: post-switch reward sparsity may
   starve the classifier; trunk drift between banking and scoring.

## Arms (16 + 8 evals, home 5070, ~3.3 h, $0)

- `g3_ar1re` × seeds 1–8: surprise trigger (thr 1.0) + reward_error selection, K=2.
- `g3_ar1ve2` × seeds 1–8: surprise trigger + value_error selection, K=2 — the honest re-run;
  head-to-head against ar1re adjudicates note 0013's "identity lives in reward, not value"
  hypothesis as science rather than artifact.
- `g3_ar1` × seeds 9–16: extension of the +0.118 headline to n=16 (precision, no new claim).

References archived (control `loop9_s1_model_e*` n=16; heads swap n=8). Scored by
`scripts/score_wave1.py g3re`.

## Pre-registered gates

- **P-G3d (reward_error):** gain(ar1re) ≥ gain(ar1) − 0.02. Prediction: ≥ ar1 (false fires
  self-correct under content addressing, unlike the forced K=2 flip).
- **P-G3d-ve (value_error re-run):** same bar. Prediction (weakly held, from note 0013's
  reasoning): FAILS it while ar1re passes — identity is in the reward function, not value
  fit. If BOTH pass, selection is easy and the hypothesis was wrong in the other direction;
  if BOTH fail, content addressing at this granularity is dead and selection defers to the
  mem-Brain lever (the box decision) or K=2-flip stands as the method.
- **P-G3e (ar1 n=16 convergence):** pooled n=16 gain within ±0.03 of the n=8 +0.1180 and
  p<0.01 (convergence-vs-decay on the headline).
- **Extension rule:** deciding quantities within ±0.02 composite of a bar → extend that arm
  to n=16 before adjudicating.
