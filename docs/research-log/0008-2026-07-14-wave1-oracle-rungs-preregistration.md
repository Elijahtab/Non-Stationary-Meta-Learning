# 0008 — 2026-07-14 — Wave-1 oracle rungs: pre-registration (G-DECOMP ladder + 3-regime screen)

**Registered BEFORE any results exist** (checklist rule; commit timestamp is the record).
User strategy call 2026-07-14: Wave-1 home ladders approved ("start a new autoresearch run");
box stays destroyed until Wave 2. Campaign note: [LOOP-0011](../autoresearch-loops/LOOP-0011-wave1-oracle-rungs.md).
Plan source: [2026-07-09 action tree](../plans/2026-07-09-research-action-tree.md) (★ G-DECOMP + C-3reg).

## Shared design

- **Instrument:** frozen-Brain evals at the LOOP-0009 eval protocol (16 inner envs,
  `decision_interval=1`, 800k steps, 100k/regime), driver `scripts/run_wave1_evals.py`,
  scored by the validated `scripts/score_eval_dir.py`. Home 5070, sequential, $0.
- **Brain:** LOOP-0009 seed-1 ep130 (`brain_model.pt`); its matched init where an init arm
  is called for. Rationale: largest replicated per-seed effect (+0.088) and an archived
  same-protocol n=16 no-swap arm (`evals/loop9_s1_model_e*`) that serves as the ladder's
  control without re-running it. Risk (stated): a single Brain — scope shares are measured
  for one controller draw; cross-Brain generalization is a later rung if needed.
- **n=8 eval seeds (1–8) per arm** first rung; extend to 16 (seeds 9–16) before adjudicating
  any gate whose deciding quantity lands within ±10 percentage points of its threshold.
  Convergence-vs-decay applies to every extension.
- **Comparisons are within-protocol only** (note 0006's cross-protocol warning): the
  box-protocol O2 number (+0.249) is NOT the reference; the ladder re-anchors headroom at
  the eval protocol itself.

## Batch 1 — G-DECOMP swap-scope ladder (32 evals)

Arms: `--policy_swap_topline --swap_scope {heads | heads+encoder | world_model | full}`.
Control **C** = archived `loop9_s1_model_e1..8` (same Brain/protocol/seeds, no swap).
Headroom **H** = mean(full) − mean(C). Share(s) = (mean(scope s) − mean(C)) / H.

- **P-W1a (re-anchor):** H > 0, Welch p<0.05, and H ≥ +0.05 composite. Failure reading
  (registered): if H < +0.05 the zero-forgetting ceiling does not survive the eval protocol
  at this magnitude — a protocol-interaction finding that pauses the memory phase for
  redesign (ping the user; do not proceed to Wave 2 on K=2 evidence).
- **P-W1b (routing, the tree's gate):**
  - share(heads) ≥ 40% of H → weight-space subtree (W2A / G3) opens first;
  - share(world_model) ≥ 20% of H → MoWM subtree (W2B / D) opens (co-opens on mixed verdict);
  - share(heads) < 20% AND share(heads+encoder) < 40% → **weight-space memory is killed**
    (tree's G kill) and the routing defaults to D (world-model) or C (scaling) by whichever
    share is larger.
- **Registered caveat:** partial scopes restore weights only (no optimizer state) — shares
  may undercount their branch; H − max(partial shares summed sensibly) upper-bounds the
  optimizer-curvature + interaction share. Report it; do not gate on it.
- **Prediction (weakly held, stated for honesty):** heads+encoder carries the majority —
  regimes differ only in reward, so the policy/value function is where regime knowledge
  concentrates; the WM share should be small (shared dynamics) *unless* dream traffic
  matters more than the 3%-of-gradients audit suggests.

## Batch 2 — C-3reg 3-regime difficulty screen (24 evals; launches after Batch 1 scores)

`num_regimes=3` (blocks A B C A B C A B at 100k/regime — 5 revisit switches), zero inner-code
change. Arms: init (matched init, no swap) / model (trained, no swap) / o2 (trained + full swap).

- **P-W1c (difficulty scaling):** (mean(o2) − mean(model)) at K=3 > H at K=2 → restoration
  headroom grows with regime count → the scaling axis (branch C / C2) stays live and becomes
  the paper's difficulty-vs-ceiling figure candidate. If ≤ H: "task too easy" is dead — regime
  count does not increase the memory premium; branch C loses its cheap rung (16×16 remains
  separately gated and expensive).
- **Secondary, descriptive (no gate):** trained-vs-init at K=3 (does the +0.05-class effect
  survive a third regime?); hit80/hit95 facets by arm.
- **Registered caveat:** K=3 composites are not comparable to K=2 composites (revisit/window
  mixture differs) — only within-K contrasts are read.

## What is deliberately NOT in Wave 1

A-R1 (surprise-triggered swap), D-R0 (stale-dream ablation), D-O (dream-boost oracle): each
needs new flag plumbing; they launch only after the ladder routes the subtree (and A-R1's
trigger design consumes the ladder's scope answer). No box work: Wave 1 is home-only per the
2026-07-14 user call; the box request goes to the user when a Wave-2 subtree opens.
