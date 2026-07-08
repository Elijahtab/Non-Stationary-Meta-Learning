# 0008 — Paper skeleton: "Trained hyperparameter meta-control helps a little; memory is the bottleneck"

**Status:** ✍️ DRAFT 2026-07-08 — the consolidated outline the notes 0001–0007 accrete into.
Not a hypothesis note; this is the write-up scaffold. Update as the replication (note 0007 /
LOOP-0009) and any memory phase (note 0009 / LOOP-0010) resolve.
**Owner:** Elijah · **Depends on:** LOOP-0009 replication for the *positive* claim's strength
(see §Venue & the one dependency).

## One-line thesis

On a fast-switching non-stationary benchmark, a *trained* outer controller that meta-modulates
an inner continual learner's hyperparameters **does** improve post-switch recovery — but the
gain is small (**+0.029 composite, ~5% relative, p<0.01, n=32/arm**) and is **~12% of the
zero-forgetting memory-restoration ceiling (+0.25)**, relocating the catastrophic-forgetting
bottleneck from *learning dynamics* to *policy/world-model knowledge restoration*.

## Working title candidates

- *"Meta-control helps, memory decides: locating the catastrophic-forgetting bottleneck in a
  controlled continual RL benchmark."*
- *"A ceiling, not a controller: what a zero-forgetting oracle reveals about neuromodulatory
  meta-learning."*

## Abstract (draft)

Neuromodulatory meta-control — an outer policy that sets an inner learner's plasticity/stability
hyperparameters online — is an appealing route to catastrophic-forgetting mitigation. We build a
controlled instrument for it (an outer PPO "Brain" over an inner Dyna-PPO continual learner on a
fast-switching MiniGrid regime schedule) and run the full control ladder that the field usually
skips: a no-controller static-HP bottom rung, oracle-timed and oracle-informed upper bounds on
the modulation pathway, and a zero-forgetting policy-swap ceiling. We find (1) a *trained* Brain
beats a matched random-init Brain by +0.029 composite (p<0.01, n=32/arm) in a single-variable
contrast — a real but small effect that **converges** with sample size; (2) the modulation
*pathway* is inert even given ground-truth regime information, and the post-switch critic
"whiplash" it was meant to damp is functional, not pathological; (3) a zero-forgetting oracle
leaves **+0.25** composite on the table — 8× the trained-control gain — locating the bottleneck
in knowledge restoration rather than learning dynamics. Along the way we document three
small-sample "mirages" that died under replication and the convergence-vs-decay criterion that
separated them from the one real effect. We argue the next lever is explicit memory, not better
meta-control.

## Contributions (claim → evidence map)

| # | Contribution | Evidence | Source |
| --- | --- | --- | --- |
| C1 | **A positive, controlled result:** trained HP meta-control improves post-switch recovery, single-variable matched contrast | trained ep130 0.5577 vs init 0.5285, Δ+0.0292, t≈2.97, p<0.01, n=32/arm; estimate *converges* (+0.046→+0.0275→+0.0292 at n=8/16/32) | note 0006; **replication:** note 0007 / LOOP-0009 |
| C2 | **The bottleneck is memory, not dynamics:** a zero-forgetting policy-swap ceiling leaves +0.25 (8× the C1 gain) | O2 composite 0.8022 (sd 0.009) vs control 0.5534 | note 0005 (pre-reg), 0006 |
| C3 | **The modulation pathway is inert:** perfect regime information through the code→mask channel does nothing; the critic "whiplash" is functional | oracle_code 0.5481 (NULL); O1 oracle-timed critic damp 0.5451 (REJECT) | note 0006 |
| C4 | **Plasticity loss is absent at this horizon** — dormant-unit fraction *falls* 0.9→0.35, never accumulates | recovered dormant-fraction probe | note 0005 addendum |
| C5 | **A methods contribution:** three reproduced small-n mirages, the convergence-vs-decay criterion, the eval/box protocol-confound trap, and the instrument audit (an untrained controller sat in the scored screening loop for three families) | LOOP-0002/0004/0006/0007 history + note 0005 adaptivity probe | notes 0004–0006 |
| C6 | **A tuned static baseline reaches 80%** on ~79% of switches — auditing/correcting the prior "never reaches 80%" (weak-baseline artifact) | D0 composite 0.5608, hit80 0.786 | note 0006 |

## Section plan

1. **Introduction.** Catastrophic forgetting in non-stationary RL; the neuromodulation/meta-control
   idea; the claim that the field reports meta-control gains without the controls that bound them;
   our thesis (helps a little, memory decides).
2. **Related work.** Meta-learned optimizers / hyperparameter control; plasticity-loss &
   dormant-neuron literature (we test and reject it here); memory / world-model replay lineage
   (the project's own MoWM §4.2 heritage — the ceiling points back to it); oracle/ceiling
   methodology in RL.
3. **The instrument.** Outer PPO Brain (MLP; 7 scalar HP levers + 8-d code, `neuromod.py`) over
   inner Dyna-PPO (world model + policy, 16 envs, 800k steps, 2 regimes × 100k, 7 switches);
   the composite (post-switch window success) + hit80 metrics; the frozen scorer
   (`score_brain_run`). **Instrument-audit subsection:** the scored screening cells trained the
   Brain only ~4 episodes → an effectively untrained controller; validate the controller before
   screening its enhancements (C5).
4. **The control ladder (Experiments).** control → D0 (no controller) → O1 (oracle-timed damp)
   → oracle_code (ground-truth code) → **O2 (zero-forgetting ceiling)**; the single-variable
   trained-vs-init contrast at the eval protocol; the cross-protocol warning (why box and eval
   numbers are non-comparable). **Table 1 = the ladder.**
5. **Results.** C1 (positive, converged), C2 (ceiling relocates the bottleneck), C3 (pathway
   inert / whiplash functional), C4 (no plasticity loss). **Fig 1 = ladder bars; Fig 2 =
   trained-vs-init per-Brain dots + convergence-with-n; Fig 3 = dormant-fraction trajectory.**
6. **Methods / reliability (the mirage section).** The three small-n mirages (critic_code,
   auxcode_hi, redo), the convergence-vs-decay criterion, the n≥8 discipline, the protocol
   confound. This is the paper's honesty spine and a reusable contribution.
7. **Discussion.** Bottleneck relocation → explicit memory is the next lever (segue to the MoWM ×
   Brain memory-levers programme, note 0009 / LOOP-0010); what the small-but-real C1 means for
   the neuromodulation literature.
8. **Limitations.** Single benchmark family (MiniGrid regimes); MLP Brain / fixed 15-lever
   interface; 4 training seeds (df=3 cross-seed) — mitigated by pooled-eval + per-Brain
   directionality (note 0007); composite is one facet (0.95 reliability facet deferred).

## Figures & tables (build list)

- **Table 1** — control ladder (control/D0/O1/oracle_code/O2) with composite + hit80 + verdict
  (ready from note 0006).
- **Table 2** — trained vs init per-Brain (after LOOP-0009: rows = seeds 1–4 + pooled; cols =
  init/selected-ckpt composite, Δ). *Currently one row (March seed 0); LOOP-0009 fills it.*
- **Fig 1** — ladder bar chart with the +0.25 ceiling gap annotated.
- **Fig 2** — trained-vs-init: per-Brain paired dots + the n=8/16/32 convergence panel.
- **Fig 3** — dormant-fraction 0.9→0.35 trajectory (C4).

## Venue & the one dependency

- **Workshop-ready today:** C2–C6 stand on committed data; C1 is real at n=32 eval seeds.
- **CoLLAs-tier needs the one mandatory fix:** C1 rests on **one** Brain *training* seed. LOOP-0009
  (note 0007) trains ≥4 fresh Brains and replicates the matched contrast across training seeds.
  - **If it replicates (P-R1a):** C1 upgrades to "across training seeds"; Table 2 fills; submit.
  - **If it fails (P-R1c):** C1 is withdrawn honestly; the paper becomes the C2–C6 methods +
    controls + ceiling contribution, and the memory phase (note 0009 / LOOP-0010) becomes the
    route to a positive result. **The paper survives either branch** — only C1's strength moves.

## Links

- Results this synthesizes: notes [0005](./0005-untested-controller-bottom-rung-oracle.md),
  [0006](./0006-controls-axis-thesis-relocated.md) (§Paper skeleton is the seed of this doc).
- The C1 replication that gates venue: note [0007](./0007-trained-brain-replication.md) /
  [LOOP-0009](../autoresearch-loops/LOOP-0009-trained-brain-replication.md).
- The next-lever programme the discussion opens: note
  [0009](./0009-memory-levers-preregistration.md) / LOOP-0010 (draft).
- Prior-art claims audited: `docs/references/Combating Catastrophic Forgetting.pdf` §4.2–4.3 +
  Table 2.
