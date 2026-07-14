# LOOP-0011 — Wave-1 oracle rungs: G-DECOMP ladder + 3-regime screen (2026-07-14–OPEN)

**Goal:** Decompose the zero-forgetting ceiling (which component of the O2 snapshot — policy
heads, encoder, world model, optimizer state — carries the headroom?) and screen whether the
memory premium grows with regime count. These two answers route which memory subtree
(weight-space / MoWM / mem-Brain) gets the first Wave-2 spend.
**Verdict:** OPEN.

## Hardware

Home RTX 5070 (12 GB, 31 GB system RAM) only — $0. **No box** (LOOP-0009's box destroyed;
user call 2026-07-14: home ladders approved, ping the user when a box is needed for Wave 2).
Sequential evals (~8 min each, T-series precedent).

## Code state

`Auto-Research` @ the pre-launch commit carrying: W0e swap-scope plumbing
(`_snapshot_learner`/`_restore_learner` scope filter + `eval_brain.py --policy_swap_topline
--swap_scope`, 151 tests green, smoke-verified), driver `scripts/run_wave1_evals.py`,
pre-registration [research-log 0008](../research-log/0008-2026-07-14-wave1-oracle-rungs-preregistration.md).
**No src/scripts edits while a batch is in flight** (evals import from this tree).

## Runs

- **Batch 1 `ladder`:** 4 swap scopes × 8 eval seeds = 32 evals (`evals/wave1_decomp_*`);
  control = archived `evals/loop9_s1_model_e1..8`. ~4.5 h.
- **Batch 2 `k3`:** 3 arms × 8 seeds = 24 evals (`evals/wave1_k3_*`) at `num_regimes=3`;
  launches only after Batch 1 scores. ~3.5 h.
- Logs: `evals/wave1_logs/`. Idempotent re-run fills gaps after any interruption.

## Statistics & verdicts

Pre-registered gates P-W1a/b/c in [research-log 0008](../research-log/0008-2026-07-14-wave1-oracle-rungs-preregistration.md)
— committed before results. Scoring: `scripts/score_eval_dir.py`. Ladder rule: n=8 → extend
to 16 if any deciding share lands within ±10 pp of its threshold; convergence-vs-decay.

## Cost

$0 (home evals only); ~8 h GPU wall-clock across both batches.

## Sibling loops

[LOOP-0009](./LOOP-0009-trained-brain-replication.md) (supplies the Brain + the control arm) ·
[LOOP-0010](./LOOP-0010-memory-levers.md) (consumes the routing verdict) ·
[note 0011](../research-notes/0011-wave0-desk-probes.md) (Wave-0 groundwork).

## Pickup state

Batch 1 launched 2026-07-14 (background, home 5070). On completion: score arms
(`score_eval_dir.py --arm "heads=evals/wave1_decomp_heads_e*" ...` vs
`control=evals/loop9_s1_model_e[1-8]*`), adjudicate P-W1a/P-W1b, update this note + note 0011
lineage, then launch Batch 2 (`k3`), adjudicate P-W1c, close the loop with the routing verdict
and the Wave-2 box request (user ping).
