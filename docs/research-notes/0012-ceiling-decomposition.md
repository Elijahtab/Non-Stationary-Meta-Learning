# 0012 — The ceiling decomposed: ~90% of the zero-forgetting headroom lives in the policy heads

**Status:** ✅ Ladder half RESOLVED 2026-07-14 (batch 1 of LOOP-0011; gates pre-registered in
[research-log 0008](../research-log/0008-2026-07-14-wave1-oracle-rungs-preregistration.md) before results).
3-regime half (P-W1c) pending — addendum will follow.
**Owner:** Elijah · **Relates to:** [note 0006](./0006-controls-axis-thesis-relocated.md) (the O2
ceiling being decomposed) · [note 0011](./0011-wave0-desk-probes.md) (Wave-0 groundwork) ·
[LOOP-0011](../autoresearch-loops/LOOP-0011-wave1-oracle-rungs.md) (ops) ·
[action tree](../plans/2026-07-09-research-action-tree.md) (the ★ G-DECOMP edge).

## Hypothesis (as registered)

> The +0.25 zero-forgetting swap ceiling is an unattributed mixture of policy-weight,
> world-model, and optimizer-curvature memory; a scope-filtered swap ladder
> ({heads | heads+encoder | world_model | full}) attributes it and routes the memory phase.

## Design

Frozen LOOP-0009 seed-1 Brain, eval protocol (16 envs, `decision_interval=1`, 800k, 2×100k),
oracle swap at ground-truth switches with `swap_scope` filtering what is banked/restored
(partial scopes carry **no optimizer state**). n=8 eval seeds/arm; control = archived
`loop9_s1_model_e1..8` (same Brain/protocol/seeds, no swap). Frozen scorer via the validated
adapter. Driver `scripts/run_wave1_evals.py`; scorer/adjudicator `scripts/score_wave1.py`;
raw: `evals/wave1_decomp_*`, `evals/wave1_scores.json`.

## Results (2026-07-14, 32/32 evals clean)

| arm | n | composite | sd | hit80 | hit95 |
| --- | --- | --- | --- | --- | --- |
| control (no swap) | 8 | 0.6202 | 0.045 | 0.982 | 0.589 |
| heads | 8 | **0.8383** | 0.027 | 0.964 | **0.857** |
| heads+encoder | 8 | 0.8546 | 0.027 | 0.982 | 0.875 |
| world_model | 8 | 0.6003 | 0.052 | 0.964 | 0.571 |
| full (original O2) | 8 | 0.8636 | 0.027 | 0.946 | 0.857 |

- **P-W1a PASS:** H = full − control = **+0.2434** (Welch p=2.8e-8) ≥ +0.05. The ceiling
  re-anchors at the eval protocol at essentially the box-protocol magnitude (+0.249) — the
  headroom is protocol-robust, not an artifact of the box measurement.
- **P-W1b — routing (all margins decisive, no extension triggered):**
  - share(heads) = **+89.6%** of H (Δ+0.2180, p=9.3e-8) → ≥40% bar cleared by 50 pp:
    **W2A weight-space memory (G3 per-regime heads) opens.**
  - share(heads+encoder) = +96.3% (Δ+0.2343) → encoder adds ~7 pp on top of heads.
  - share(world_model) = **−8.2%** (Δ−0.0200, p=0.42, null) → <20% bar: **the MoWM subtree
    does NOT open.** Restoring the WM alone transports nothing (directionally *negative*).
  - full − heads+encoder = +0.009 (~4% of H) → optimizer-curvature memory is ~negligible.
- Reliability facet moves with the composite: hit95 0.589 → 0.857 under heads swap.

## Interpretation

Regime knowledge on this instrument is stored ~90% in the actor/critic heads, ~7% in the
encoder, ~4% in optimizer state, and ~0% in the world model. This kills the "mixture of world
models" *mechanism* as the vessel for the memory phase (prior-paper §4.2's premise — the WM
rung IS oracle-routed MoWM, and it is null), while confirming the memory *direction* at full
strength. The cheapest sufficient memory is per-regime policy/value heads — exactly branch
G3 (K head pairs + selection ladder: oracle one-hot → self-inferred → Brain code-argmax),
with A-R1's learned trigger composing into a Brain-free learned O2 (~+0.22 at eval protocol
if trigger+selection cost nothing — the realistic target the paper's method section needs).

**Registered caveats that still apply:** single Brain (seed-1) — scope shares are one
controller draw; partial scopes carried no optimizer state (share may shift a few pp with
matched Adam restore); n=8 (all deciding margins ≥50 pp from thresholds, so the ladder rule
did not require n=16).

## Consequences routed

- **W2A opens:** G3 (K per-regime head pairs, `nn.ModuleList` + `set_active_head`) is the
  next build; its selection ladder starts oracle-selected (zero Brain retrain).
- **W2B (MoWM/D) closed at the oracle rung** — D-R0/D-O/D-1 are struck from the queue
  (their premise — WM state transports regime knowledge — failed at its upper bound).
- **B1/LOOP-0010 unchanged** (interface work rides G3's Brains later).
- Paper: the decomposition table is a new headline result (C2 sharpens: "the bottleneck is
  memory, *specifically policy-head memory*"); prior-paper §4.2 audit gains the WM-null.

## Links

Pre-registration: [research-log 0008](../research-log/0008-2026-07-14-wave1-oracle-rungs-preregistration.md) ·
ops + pickup: [LOOP-0011](../autoresearch-loops/LOOP-0011-wave1-oracle-rungs.md) ·
raw scores: `evals/wave1_scores.json` (local) · K=3 difficulty screen: pending addendum.
