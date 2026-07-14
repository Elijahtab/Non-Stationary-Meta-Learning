# 0011 — Wave-0 desk probes: screening power, dead-dim noise, and the 0.95 facet

**Status:** ✅ RESOLVED 2026-07-14 — all three $0 probes from the
[2026-07-09 action tree](../plans/2026-07-09-research-action-tree.md) Wave 0 ran on archived
artifacts (no new compute). Headlines: **the registered H4 kill rule FIRES** (autoresearch-sized
screens cannot see +0.029-sized effects); **B-R0 confirms the dead-dim noise premise** (all five
trained Brains still emit init-scale noise on the inert code dims); **exploratory: the trained
Brain moves the hit-0.95 reliability facet** (+0.143 t1, +0.071 pooled LOOP-0009) where hit80
was saturated.
**Owner:** Elijah · **Relates to:** [action tree](../plans/2026-07-09-research-action-tree.md)
(branches H, B, and the paper's H1/H2 items) · [note 0006](./0006-controls-axis-thesis-relocated.md)
(t1 arms) · [note 0007](./0007-trained-brain-replication.md) (LOOP-0009 arms) ·
[note 0008](./0008-paper-skeleton.md) (paper claims this feeds).

## Method (one paragraph)

`scripts/wave0_retro_stats.py` (committed; the provenance lesson from note 0006 applied) scores
every archived eval through the **validated adapter staging + frozen scorer** — trust anchors
assert exact reproduction of note 0006 (t1 Δ **+0.0292**) and note 0007 (LOOP-0009 pooled Δ
**+0.0455**) before any probe runs; both **PASS**. Arms: `t1_trained` / `t1_init` (n=32 each,
March Brain) and `loop9_s{1..4}_{trained,init}` (n=16 each, fresh Brains). Raw probe output:
`evals/wave0_stats.json` (local, reproducible by re-running the script).

## Finding 1 — W0a screening power: the registered H4 kill rule FIRES

Bootstrap-subsampling (B=4000, without replacement) the t1 arms to autoresearch-sized cells,
Welch α=0.05:

| effect | n/arm | power (2-sided) | power (1-sided) | P(Δ>0) |
| --- | --- | --- | --- | --- |
| +0.0292 (t1) | 8 | **0.24** | 0.37 | 0.96 |
| +0.0292 (t1) | 16 | **0.49** | 0.71 | 1.00 |
| +0.0881 (loop9 s1) | 8 | 1.00 | 1.00 | 1.00 |
| +0.0450 (loop9 s2) | 8 | 0.61 | 0.81 | 1.00 |
| −0.0270 (loop9 s3) | 8 | 0.00 | 0.00 | 0.01 |
| +0.0758 (loop9 s4) | 8 | 1.00 | 1.00 | 1.00 |

**Registered rule (action tree, branch H):** power < 0.60 at n=8 AND < 0.80 at n=16 ⇒ H4 dies.
**0.24 and 0.49 → H4 dies as registered.** An 8-IR trained-controller screening cell cannot
discriminate effects at the scale of the one confirmed positive result.

**Calibration nuance (recorded, not a verdict change):** the same probe shows screens ARE
adequately powered for **≥ +0.05** effects (0.61–1.00 at n=8) — the scale the memory branches
target (ceiling slices of +0.249). If a future session wants trained-controller screening for
G/B variants, that is a **new registration with a minimum-detectable-effect ≥ +0.05 gate**, a
human call — the H4 entry as written stays dead. Risk flagged: this nuance is post-hoc; treat
the ≥+0.05 power numbers as design inputs, not as evidence H4 "really" survived.

## Finding 2 — W0b / B-R0: the dead-dim noise premise is REAL

`actor_log_std` (init −0.5 ⇒ σ≈0.607) read from all five trained Brains (March + 4 fresh):

| ckpt | HP dims 0–6 (mean) | code dims 7–14 (mean) |
| --- | --- | --- |
| march_s0 | −0.532 | −0.498 |
| loop9_s1 | −0.552 | −0.503 |
| loop9_s2 | −0.533 | −0.501 |
| loop9_s3 | −0.503 | −0.510 |
| loop9_s4 | −0.552 | −0.500 |

Trained σ: HP dims 0.587, **code dims 0.605 vs init 0.607** — 130 episodes of training moved
the dead dims' noise scale by ~0.3%. With `brain_ent_coef=0.0` and no reward gradient through
the inert pathway (note 0006, oracle_code NULL), nothing ever shrinks it: **every deployed
Brain emits full exploration-scale noise on 8 inert action dims, and their log-probs dilute
PPO's summed objective on the 7 live dims.** B1's premise (prune 15→7) is confirmed real;
whether the dilution *costs* anything measurable remains B1's experiment (4×BR50, per the tree).
Assumption to keep explicit: "no gradient pressure" is inferred from the pathway-inert result +
flat log_std, not from inspecting gradients directly.

## Finding 3 — W0d / H1+H2: IQM & CIs are clean; the 0.95 facet moves (exploratory)

- **Robustness:** IQM tracks the mean everywhere (pooled LOOP-0009 ΔIQM **+0.0527** vs Δmean
  +0.0455) — the effect is not outlier-driven; if anything it is stronger in the interquartile
  mass. Seed-stratified bootstrap 95% CI for the pooled Δ: **[+0.033, +0.058]**.
- **The pre-authorized 0.95 facet (exploratory — NOT pre-registered as confirmatory):** hit80
  was saturated/identical between arms (~0.96–0.98, notes 0006/0007), but at the 0.95 threshold
  the trained Brain moves reliability: t1 **0.714 vs 0.571 (+0.143)**; LOOP-0009 pooled
  **+0.071**, per-seed +0.018 / +0.214 / −0.098 / +0.152 (signs match the composite Δs, 3/4
  positive). Reading: trained HP meta-control doesn't just raise average post-switch recovery —
  it increases the fraction of switches that recover *nearly completely*. For the paper this
  belongs in C1's results as a clearly-labeled exploratory facet (the "composite is one facet"
  limitation in note 0008 is now partially addressed); it must NOT be promoted to a
  confirmatory claim without a fresh pre-registered replication arm.

## Implications routed into the tree

- **H4:** dead as registered. Autoresearch screening returns only if re-registered with an
  MDE ≥ +0.05 gate (human decision).
- **B1:** premise confirmed; stays live as the LOOP-0010 interface vehicle (unchanged priority:
  behind the Wave-1 gate).
- **Paper (note 0008):** add IQM + stratified CI to C1/C5 methods; add the hit95 exploratory
  facet to C1 results + limitations.
- **Wave 0 remaining:** W0c (encoder-dormancy probe build — C4's blind spot) and W0e
  (`eval_brain.py` mechanism-flag plumbing — unlocks Wave-1 free home ladders, G-DECOMP first).

## Links

Script: `scripts/wave0_retro_stats.py` · raw: `evals/wave0_stats.json` (local) · arms:
notes [0006](./0006-controls-axis-thesis-relocated.md) / [0007](./0007-trained-brain-replication.md) ·
plan: [2026-07-09 action tree](../plans/2026-07-09-research-action-tree.md) §3 H & B, §5 Wave 0.
