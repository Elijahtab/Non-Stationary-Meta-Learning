# LOOP-0002 — v1-scout-campaign (2026-07-04)

**Goal:** First real campaign: 5 Fable trials against `fast_switch_scout_v1` (5×5, n=1).
**Verdict:** 6 science rejections in 1–3% band below baseline — later shown to be an
instrument artifact (v1 saturated); triggered the scout v2 switch (research-log 0004).

## Hardware
Home RTX 5070 (scout ~76 min/run at ~545 SPS×streams; holdout not reached).

## Code state
`trainable-vs-frozen-sweep` @ `ce4c819`; `config/research_manifest.toml` (primary = scout_v1);
launcher `scripts/run_local_loop.ps1` (created mid-loop after a 429 interruption).

## Runs
Ledger sessions `20260704-022146` (baseline 0.6622/hit80 0.991 + trial 1, then four
`research_command_failed` fast-fails) and `20260704-133654` (resume: trials 1–5). 6 verdicts
total across both sessions.

## Obstacles
- **Subscription 429** ("session limit, resets 4:40am PT") killed trials 2–5 of the first
  session in ~2.5 s each; the wrapper's `is_error→exit 1` path fast-failed them correctly
  (retryable infra failures, no benchmark waste). Resumed next window; baseline cache hit.

## Statistics & verdicts (all rejected, `primary_score_did_not_improve`)
actor-only 0.6541 · gain α=1.0 0.6552 · FiLM 0.6470 · split-masks 0.6411 ·
context-16 0.6622 (near-no-op) · channel gates (free-pick) 0.6507 — vs baseline 0.6622,
whose hit_rate_80 of **0.991** sat at ceiling (the tell that the instrument couldn't reward wins).

## Cost
~$22 agent fees; ~9 h home GPU.

## Sibling loops
None concurrent; verdicts carried into LOOP-0003 as benchmark-scoped prior evidence.

## Pickup state
(historical) Superseded by research-log 0004: primary → `fast_switch_scout_v2`.

## Links
`autoresearch/trial_results.jsonl` (sessions above); `docs/research-log/0004-*.md`;
queue verdict annotations in `config/hypothesis_queue.md`.
