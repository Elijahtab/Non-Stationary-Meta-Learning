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

## Outcome — FALSIFIER FIRED (2026-07-05, n=8 per arm, results branch confirm_g{0..3})

| condition | composite mean [95% CI] | hit_rate_80 mean [95% CI] | med steps→80 |
| --- | --- | --- | --- |
| frozen control | **0.5534** [0.5478, 0.5588] | 0.829 [0.815, 0.846] | 44,758 |
| actor-only | 0.5479 [0.5437, 0.5528] | 0.829 [0.810, 0.850] | 45,825 |
| gain α=0.5 | 0.5460 [0.5416, 0.5514] | 0.820 [0.795, 0.845] | 47,850 |

- **Prediction 2 (the registered secondary) failed flatly:** neither arm beats the control's
  hit_rate_80 mean by anything (actor-only Δ +0.000, gain Δ −0.009); no CI separation anywhere.
- **Prediction 1 half-held:** actor-only sits (barely) inside the control's composite CI;
  gain α=0.5 falls below it (−0.0074) — mildly *worse* than predicted.
- **The n=1 "hit80 signature" was selection on noise — specifically the baseline's.** The scout
  baseline's hit80 of 0.786 (seed 0) was a low draw of a distribution whose true mean is ~0.83;
  every variant compared against it looked reliability-improving. Regression to the mean, caught
  exactly as the pre-registration intended.
- **The holdout gate is vindicated:** its "hair-thin" zero-tolerance rejections of trials 1/3/4
  were all CORRECT verdicts — the mechanisms were never better.

**Per the pre-committed decision rule: both hypotheses CLOSE as rejected; no further tuning of
this mechanism family without a qualitatively new idea.** Combined record across notes
0001/0002 and this campaign: 7 trainable-decoder variants + 6 mechanism-family variants at v1 +
5 retests at v2 + 2 arms at n=8 — the frozen-decoder, shared-suppressive-mask baseline has
never been beaten. The "context-code routing is inert; the Brain's scalar-HP control drives
adaptation" thesis now rests on its strongest evidence, with pre-registered methodology.
