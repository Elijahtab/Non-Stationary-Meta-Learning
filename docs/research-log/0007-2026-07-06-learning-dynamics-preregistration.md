# 0007 — Registered predictions: LOOP-0006 learning-dynamics screening (box 1–4, home 5)

- **Date:** 2026-07-06 (registered BEFORE any screening results exist; box wave 1 not yet
  launched at registration time)
- **Status:** Accepted (campaign launching)
- **Touches:** master-implemented, flag-guarded conditions on branch
  `autoresearch-run-20260706`: `brain_neuromod_gradgate`, `brain_neuromod_critic_code`,
  `brain_neuromod_auxcode` (coef 0.05), `brain_neuromod_adamflush` (threshold 0.25) —
  preset `scoutv2`, out dirs `sweeps/confirm_ld_*` / `explore_ld_*`. Home: queue entry 5
  (code-weighted anchoring) via the audited trial loop under `baseline_primary`.

## Context

LOOP-0005 brainstorm output (loop note + research note 0003): route the context code into
learning dynamics instead of forward-path modulation (that family closed at n=8,
research-log 0005). Full per-hypothesis arguments and predictions: LOOP-0005 note §
"Brainstorm output". Registered coefficient choices (judgment calls, not tuned):
aux_code_coef = 0.05, adam_flush_threshold = 0.25.

## Comparison anchor (box)

LOOP-0004 frozen control, n=8, same `scoutv2` preset and GPU class (results
confirm_g0..g3): composite **0.5534 [0.5478, 0.5588]** (per-seed σ ≈ 0.008),
hit_rate_80 0.829 [0.815, 0.846]. Candidate cells use seeds from the same 1..8 range.

## Predictions (registered; composite = post-switch window success)

1. **gradgate** (top pick, note 0003 H1): composite ↑, needs Δ ≥ +0.005 vs control mean
   to count as real; hit80 ↑ or flat. Mechanism check: mask stats vary with switch phase.
2. **critic_code**: composite ↑ via less-biased post-switch GAE; hit80 ↑.
3. **auxcode**: composite ↑ modestly; mechanism check (registered): regime-decoding-probe
   accuracy ↑; a score-flat/probe-up outcome still informs note-0001 H2.
4. **adamflush**: composite ↑; hit80 is a *guarded* secondary (transient post-flush
   destabilization may push it ↓).
5. **code-weighted anchoring** (home): composite ↑ only if targeted protection beats the
   uniform scalar `anchoring_weight` lever; adjudicated by the frozen-benchmark gates
   (3-seed `baseline_primary`, then holdout).

**Family falsifier:** if no arm's multi-seed composite mean exceeds the control's n=8 CI
(≥ 0.5588 at n≥3), the learning-dynamics family joins the forward-modulation family and
the "scalar-HP control is sufficient" thesis stands on both the activity and plasticity
sides (note 0003, outcome 2).

## Decision rules (pre-committed, box screening)

- **Wave 1 (4 cells):** each candidate × seed 1, n=1. Kill only if composite <
  0.5374 (control mean − 2·per-seed σ) — i.e., clearly destabilized; otherwise continue.
- **Wave 2 (4 cells):** seeds 2–3 for the two best wave-1 candidates (n=3 each); survivors
  of the kill rule that didn't make top-2 wait in the Confirmation Queue.
- **Promotion:** candidate n=3 mean > control n=8 mean (0.5534) → holdout cells
  (seeds 11/23/37, zero regression vs the holdout baseline) per the user's directive;
  n=3 mean > control CI hi (0.5588) → additionally extend to n=8 for a CI-separated claim.
- **Rejection:** n=3 mean ≤ control mean → science verdict
  `primary_score_did_not_improve`, benchmark-scoped, close on this benchmark.
- **Idle capacity:** backfill with seeds 2–3 of remaining candidates, then n=8 extension
  cells, in queue-rank order.

## Addendum 2026-07-06 (registered before any scout results): idle-capacity explore variants

Four flag-combo conditions for idle box GPUs (n=1 screens, `explore_ld_*` out dirs, NOT part
of the confirmation decision rules above — a promising explore hit enters the Confirmation
Queue like any lead):

- `gradgate_gain` (gate ∈ [0.5, 1.5]): composite ↑ if amplify-where-relearning beats
  protect-only; watch for instability (amplified gradients).
- `critic_code_gradgate` (entries 1+2 stacked): ≥ the better of its parents if the
  mechanisms are complementary (plasticity steering + value conditioning); below either
  parent = interference.
- `auxcode_hi` (coef 0.15): dose-response check for entry 3.
- `adamflush_lo` (threshold 0.1): more frequent flushes; composite ↑ only if flushing is
  cheap — hit80 ↓ expected if not.

## Fork addendum (registered 2026-07-06 ~19:00Z, user-directed — reverses the register row)

A run is **accepted on EITHER path**: (A) primary composite improves (n-mean > control mean
0.5534), OR (B) **reliability signature** — composite does NOT degrade (n-mean ≥ control CI-lower
0.5478) AND hit_80 beats baseline (> control hit80 CI-upper 0.846). Applies retroactively to
LOOP-0006 screening and forward to all runs. **Critical correction (learned this campaign): the
fork is only trustworthy at n≥8.** At n=3 it flagged two "reliability leads" (critic_code
hit80 0.860, auxcode_hi 0.851) that were lucky-triple sampling artifacts — both regressed into
the control band at n=8 (0.835, 0.819). The fork correctly flags *candidates*; n≥8 is required to
*confirm* one. This is now the register decision "Gate refinement — ADOPTED (fork)".

## Outcome — **NULL** (LOOP-0006 closed 2026-07-07 04:43Z)

**No condition in the code-directed learning-dynamics family beats the frozen control at n=8 on
the scout primary.** Full verdict/incident trail: `autoresearch/live/RUN-20260706.md`.

- **All 4 registered candidates rejected.** auxcode KILLED wave 1 (0.5347 < kill line); gradgate
  (n=3 0.53658), critic_code (n=3 0.55195), adamflush (n=3 0.54517) all rejected
  `primary_score_did_not_improve`. The note-0003 H1 prediction (gradgate Δ ≥ +0.005) was
  **FALSIFIED** (gradgate n=3 sat 0.0168 *below* control).
- **Explore/scaled variants (addendum) also null.** gradgate_gain's composite lead (n=3 0.55578)
  **FAILED holdout** (hit80 collapse on seeds 11/23/37) → seed-specific. auxcode_hi and critic_code
  produced n=3 *reliability* signatures (Path B) that **both evaporated at n=8**: critic_code
  hit80 0.860→0.8348 (inside control CI), auxcode_hi 0.851→0.8192 (below control mean). The
  auxcode dose-response is fully closed (0.05 killed / 0.10 degraded n=5 / 0.15 washed n=8 /
  0.25 collapsed). No stacking (critic_code×auxhi) helped.
- **Family falsifier met on both paths.** The "scalar-HP control is sufficient" thesis (note 0003
  outcome 2) now stands on BOTH the forward-modulation (LOOP-0002/0004) and the learning-dynamics
  (LOOP-0005/0006) families: **injecting the regime code into the inner agent — any pathway — does
  not robustly improve fast-switch mean success OR reliability.**
- **Method lesson (carried to LOOP-0007):** screen at n≥8 from the start; n=3 over-selects. →
  Pivot to the code-free plasticity/stability family ([note 0004](../research-notes/0004-code-free-plasticity-stability.md)).
