# LOOP-0006 — learning-dynamics campaign (OPEN, started 2026-07-06)

**Goal:** Screen the 5 LOOP-0005 hypotheses (context code → learning dynamics instead of
forward-path modulation): box battle-tests entries 1–4 as master-implemented conditions,
home screens entry 5 via the audited trial loop. Non-rejected candidates go to holdout;
if everything closes, spin a new brainstorm loop.
**Verdict:** _(open)_

## Hardware
Home: RTX 5070 12 GB (local screening loop). Box: Vast.ai `76.67.137.57:26061`,
4× RTX 3060 12 GB, 80 vCPU → `BOX_CAPACITY = 4` parallel cells (1 cell/GPU at ~6.5 GB).

## Code state
Main tree: `trainable-vs-frozen-sweep` + queue/surface commits (5 live queue entries,
`ppo.py`/`train.py` opened as optional surface). Run worktree branch:
`autoresearch-run-20260706` — master-implemented flag-guarded conditions
`brain_neuromod_gradgate`, `brain_neuromod_critic_code`, `brain_neuromod_auxcode`,
`brain_neuromod_adamflush` (preset `scoutv2`). Box clones this branch.

## Runs
_(in flight — see living doc `autoresearch/live/RUN-20260706.md`)_

## Obstacles
_(open)_

## Statistics & verdicts
Box comparison anchor: LOOP-0004 frozen control at n=8, same preset/GPU class —
composite **0.5534 [0.5478, 0.5588]**, hit_rate_80 0.829 [0.815, 0.846]. Home anchor:
3-seed `baseline_primary` (first trial pays the one-time ~3.8 h anchor run).
Pre-registered predictions + decision rules: research-log 0007 (in the run worktree).

## Cost
_(open — rails: MAX_BOX_HOURS_PER_DAY 12, MAX_AGENT_SPEND_PER_GATE ≈ 4 trials)_

## Sibling loops
[LOOP-0005](./LOOP-0005-brainstorm.md) (source of the queue); home and box halves of this
campaign share this note.

## Pickup state
Read the living doc `autoresearch/live/RUN-20260706.md` (gitignored) — it holds current
allocation, gate logs, and master decisions. If the run is dead: check
`autoresearch/*.exitcode`, the manifest ledger, and the `results` branch for pushed
`confirm_*`/`explore_*` sweeps; verdicts route per research-log 0007 decision rules.

## Links
Queue: `config/hypothesis_queue.md` (LIVE section) · predictions: research-log 0007 ·
note [0003](../research-notes/0003-code-directed-plasticity-gating.md) · living doc
`autoresearch/live/RUN-20260706.md` · results branch dirs `confirm_ld_*`/`explore_ld_*`.
