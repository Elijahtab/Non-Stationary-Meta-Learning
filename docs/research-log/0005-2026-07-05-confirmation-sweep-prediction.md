# 0005 — Registered prediction: actor-only & gain-α0.5 confirmation sweep (n=8)

- **Date:** 2026-07-05 (registered BEFORE any sweep results existed; sweep launched the same
  day on a 4× RTX 3060 Vast.ai box, branch `confirmation-sweep-conditions` @ `9991565`)
- **Status:** Accepted (sweep in flight)
- **Touches:** nothing — this is a pre-registration record. Sweep uses the `scoutv2` preset
  (programmatic copy of `fast_switch_scout_v2`) and conditions `brain_neuromod` (control),
  `brain_neuromod_actor_only`, `brain_neuromod_gain05`, seeds 1–8, out dirs `confirm_g{0..3}`.

## Context

The v2 (8×8) scout campaign produced two n=1 signals with the same signature — composite flat,
`hit_rate_80` up: actor-only 0.875 vs baseline 0.786 (and a +1.2pt echo on the 3-seed holdout),
gain α=0.5 0.893 vs 0.786. Both were rejected by the loop's zero-tolerance composite gates
(−0.0002 and −0.0008 respectively). n=1-vs-n≤3 cannot adjudicate deltas this small; this sweep
can. Full trail: docs/research-log/0004, autoresearch/handoffs/2026-07-05-autoresearch-dual-box.md.

## Prediction (registered)

1. **Primary endpoint (composite):** both arms land within the frozen control's 95% CI at n=8
   (no real composite gain — consistent with everything since notes 0001/0002).
2. **Secondary endpoint (hit_rate_80, registered):** BOTH arms beat the frozen control's mean,
   and at least one shows CI separation at n=8. Directional confidence: gain-α0.5 > actor-only
   (larger n=1 jump).
3. **Falsifier:** if neither arm's hit_rate_80 CI separates from control, the n=1 signals were
   noise; both hypotheses close as rejected and the "context-code routing is inert" thesis
   (notes 0001/0002) stands at full strength.

## Decision rule (pre-committed)

- hit80 CI-separated AND composite not worse → promote: factorial arm (actor-only × gain) next,
  plus paper analysis of the reliability/latency trade-off.
- Otherwise → close both; no further tuning of this mechanism family without a new idea.

## Outcome

_(to be filled from sweeps/confirm_g{0..3} summaries)_
