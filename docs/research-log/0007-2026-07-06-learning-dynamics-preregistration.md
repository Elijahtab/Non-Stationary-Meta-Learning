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

## Outcome

_(open — filled at gates; living doc `autoresearch/live/RUN-20260706.md` has interim state)_
