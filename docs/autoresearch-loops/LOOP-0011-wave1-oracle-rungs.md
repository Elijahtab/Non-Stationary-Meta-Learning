# LOOP-0011 — Wave-1 oracle rungs: G-DECOMP ladder + 3-regime screen (2026-07-14, closed same-day)

**Goal:** Decompose the zero-forgetting ceiling (which component of the O2 snapshot — policy
heads, encoder, world model, optimizer state — carries the headroom?) and screen whether the
memory premium grows with regime count. These two answers route which memory subtree
(weight-space / MoWM / mem-Brain) gets the first Wave-2 spend.
**Verdict:** OPEN — **Batch 1 (ladder) RESOLVED 2026-07-14: P-W1a PASS (H=+0.2434 at eval
protocol, p=2.8e-8); P-W1b routes to W2A weight-space memory** — share(heads)=+89.6%,
share(heads+encoder)=+96.3%, share(world_model)=−8.2% (null) → **G3 opens, MoWM subtree
closed at its oracle rung**; no extension needed (all margins ≥50 pp). **Batch 2 (k3)
RESOLVED same day: P-W1c FAIL** — (o2−model)@K3 = +0.2351 ≤ H@K2 = +0.2434, the memory
premium is flat in regime count → the scaling axis dies its registered cheap death (16×16
loses its rationale). Trained-vs-init at K=3: +0.020 (p=0.13, descriptive). Full numbers:
[note 0012](../research-notes/0012-ceiling-decomposition.md). 56/56 evals clean, $0.

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

**CLOSED 2026-07-14 — both batches resolved same-day** (scores + adjudication:
`scripts/score_wave1.py`, raw `evals/wave1_scores.json` + `evals/wave1_decomp_*`/`wave1_k3_*`,
local). The routing verdict is in force:

1. **Next build: G3 weight-space memory** (K per-regime actor/critic head pairs,
   `nn.ModuleList` + `set_active_head` in `network.py`; inactive heads get no gradient so
   Adam keeps per-head moments free). Selection ladder per the action tree: oracle one-hot →
   self-inferred (value-error / SupSup entropy, no Brain retrain) → Brain code-argmax.
   Then **A-R1** (surprise trigger, `_update_surprise_spike` at K=2 "restore the other one")
   composes with G3 into the learned-O2 method rung. All eval-mode, home-runnable, $0.
2. **Closed by this loop:** MoWM/D subtree (WM rung null at its upper bound); scaling axis
   C (premium flat in K; 16×16 needs a new rationale). Struck from the queue: D-R0, D-O,
   D-1/2/3, C-3reg Brains (C2), C-16×16 (3b) absent new justification.
3. **Box:** NOT needed for G3's oracle/self-inferred rungs (home evals). The box ping goes
   to the user when learned-O2 passes and Brain fine-tunes (A-R3/B1, ~4×BR50 ≈ 1.5
   box-nights ≈ $20–35) are justified.
