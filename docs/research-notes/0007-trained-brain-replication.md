# 0007 — Multi-training-seed replication of the trained-Brain effect

**Status:** ✅ RESOLVED 2026-07-11 — **P-R1a PASSED, replicates.** 4 fresh Brains (seeds 1–4)
trained to ep130; pooled trained-vs-init **Δ +0.0455 composite, p=1.3e-6, n=16/arm**, 3/4 Brains
directionally positive, estimate converged (n=8 +0.054 → n=16 +0.046). See §Results.
Pre-registered 2026-07-08 (predictions below untouched).
**Owner:** Elijah · **Relates to:** [note 0006](./0006-controls-axis-thesis-relocated.md) (the
single-training-seed contrast being replicated), [note 0005](./0005-untested-controller-bottom-rung-oracle.md)
(the untrained-controller finding), [LOOP-0009](../autoresearch-loops/LOOP-0009-trained-brain-replication.md)
(campaign/ops note), hand-off `docs/hand-offs/2026-07-08-loop-0008-findings-strategic-fork.md`.

## Hypothesis (one sentence)

> The trained-Brain effect of note 0006 (**+0.0292 composite, t≈2.97, p<0.01, n=32/arm**) is a
> property of *Brains trained this way*, not of the single March training seed — fresh Brains
> trained from scratch at the same configuration will replicate a positive trained-vs-init
> contrast at the eval protocol.

## Background: why this is mandatory, not optional

- The program's one positive result rests on **one** outer-training seed (seed 0,
  `runs/brain_2_regimes_8x8_neuromod_20260315-180839`). The n=32 in note 0006 varied only
  *evaluation* seeds — it rules out eval noise, not "this Brain got lucky in training."
- Meta-RL outer-loop training is notoriously high-variance across training seeds; "how many
  meta-training seeds?" is the first reviewer question. With one, the honest claim is
  existential ("there exists a trained Brain that helps") — workshop-tier only.
- The program's own history is the strongest argument: three effects (critic_code, auxcode_hi,
  redo) looked real and died under replication along the axis they hadn't varied. Training-seed
  randomness is exactly the axis the T-series never varied.

## Design (pre-registered)

- **N=4 fresh Brain training runs, seeds 1–4**, configuration identical to the March run's
  `config.txt` except the seed: 130 episodes, `brain_num_envs=8` (async), `brain_lr=1e-4`,
  `reward_mode=recovery`, inner 16 envs / 800k total steps / 100k per regime,
  `decision_interval=1`, `pretrain_episodes=1` (recovery), per-episode checkpoints.
- **Checkpoint selection per Brain:** the same documented selector used for ep130 — best
  10-episode-average training reward — applied **before** any eval score exists (selection
  hygiene; note 0006 documented this selector against the losing ep91 single-episode-peak
  alternative).
- **Contrast per Brain:** selected checkpoint vs the **same seed's saved init checkpoint**
  (matched-per-seed init control, as the March run kept `brain_init_seed0.pt`), evaluated
  frozen at the **eval protocol** (16 inner envs, `decision_interval=1`, 800k/100k schedule).
  Within-protocol only — note 0006's cross-protocol warning is load-bearing.
- **Eval ladder per arm per Brain:** n=8 → n=16 → (n=32 if marginal), with the
  **convergence-vs-decay criterion** applied to the pooled estimate — the test that separated
  the real T-series effect from the program's three mirages.
- **Scoring:** frozen scorer only, via the eval-staging adapter recreated as
  `scripts/score_eval_dir.py` (note 0006 Provenance), **validated by re-scoring the archived
  `evals/t1_*` dirs — it must reproduce note 0006's numbers exactly** before any new eval is
  scored.

## Pre-registered predictions

- **P-R1a (primary):** pooled trained-vs-init Δcomposite > 0 with p<0.01 at the final ladder
  rung, **AND** ≥3 of 4 Brains individually directionally positive. This is the paper-grade
  replication bar.
- **P-R1b (secondary):** per-Brain training-dose trend holds (selected checkpoint > init;
  broadly monotone as in seed 0's init < ep91 < ep130).
- **P-R1c (registered failure reading):** pooled Δ ≈ 0, or the estimate decaying with n, or
  ≥2/4 Brains negative → the March Brain was a favorable training draw; the positive claim is
  **withdrawn**; the paper pivots to the methods/null/ceiling story and the strategic fork
  resolves toward the memory-restoration phase.
- **hit80:** predicted identical between arms (0.982 both in the T-series) — no reliability
  claim is made or tested here.

## Statistical honesty (stated up front)

With 4 training seeds, a cross-seed t-test has df=3 — underpowered by construction. The
pre-registered gate is therefore pooled-eval significance + per-Brain directionality, with
per-Brain Δs and the cross-Brain sd reported verbatim in the paper. If cross-Brain variance
visibly swamps the mean effect, P-R1c applies regardless of the pooled p-value.

## Assumptions (explicit, with risk)

- **A1 — training-environment equivalence.** A rented-GPU training run (different hardware /
  torch build) draws from the same distribution as the March local run. Fresh randomness is
  the point; *systematic* environment differences are the risk. Mitigation: pinned deps per
  `docs/plans/cloud-setup.md`; no Pascal GPUs (or apply the cu126 downgrade); sanity-check one
  Brain's early training-reward curve (~ep 10–15) against the March curve shape before
  committing the full budget.
- **A2 — the selector generalizes.** avg10-training-reward picks a good checkpoint on fresh
  seeds as it did on seed 0. Risk: selector overfits seed-0's curve shape; mitigated by P-R1b
  reporting the dose trend, not just the selected point.
- **A3 — adapter fidelity.** `scripts/score_eval_dir.py` faithfully reproduces the ephemeral
  `score_t1.py`. Enforced by the re-scoring validation gate above.
- **A4 — budget.** ~130 eps ≈ 40–45 h per Brain on a 3060-class GPU (extrapolated from the
  measured ~15 h / 40–50 eps); 4 in parallel ≈ 2 days. If slower, the floor is ≥90 episodes
  (seed 0's ep91 already showed +0.025 at n=8) with the selector applied to what exists.
- **A5 — the March config is what config.txt says.** `config.txt` records the ep100→130
  *resume* invocation; verify episodes 1–100 used the same core config (check the episode dirs
  / March research-log) before launch.

## Decision matrix

- **Replicates (P-R1a):** the claim upgrades to "trained HP meta-control improves post-switch
  recovery *across training seeds*" — CoLLAs-viable; the write-up-vs-memory-phase fork becomes
  a scheduling choice, not a scientific one.
- **Fails (P-R1c):** the claim is withdrawn honestly; the paper is the methods + controls +
  ceiling contribution; the memory phase (LOOP-0010 brainstorm) becomes the only route to a
  positive result.
- **Either way** the 4 fresh Brains are reusable: they are the "trained Brain, ordinary
  interface" baseline arm of any future memory-lever experiment.

## Pre-launch verification (2026-07-08, before any fresh Brain exists)

The enforceable pre-launch gates in the design above were satisfied *before* launch — recorded
here so the pre-registration and its checks are one committed artifact. Predictions P-R1a–c are
untouched.

- **A3 — adapter fidelity: PASS.** `scripts/score_eval_dir.py` (the promoted form of the
  ephemeral `score_t1.py`) re-scores the archived `evals/t1_*` via the frozen `score_brain_run`
  and reproduces note 0006 **exactly**: trained ep130 **0.5577**, init **0.5285**, Δ **+0.0292**,
  t **2.96** (p=0.005), hit80 **0.982** both, n=32/arm. `--self-test` asserts this.
- **A5 — March config: VERIFIED.** The embedded `args` in the March run's `episode_50` and
  `episode_99` checkpoints match the resume `config.txt` core (brain_lr 1e-4, brain_num_envs 8
  async, decision_interval 1, inner 16 envs / 800k / 100k per regime, reward_mode recovery,
  pretrain_episodes 1); the config was consistent across episodes 1–130. Fresh Brains replicate
  exactly these params.
- **Seed/init plumbing (design "same seed's saved init checkpoint"): gap found & fixed.**
  `train_brain.py` previously seeded only the *inner* PPO and never saved an episode-0 init
  (March's `brain_init_seed0.pt` was reconstructed post-hoc). It now `seed_everything(args.seed)`
  before Brain init and saves `brain_init.pt` (the matched per-seed init control) before any
  training — verified reproducible-per-seed and distinct-across-seeds, smoke-tested end-to-end,
  148 tests green. So each fresh Brain's init is its *own* saved episode-0 snapshot, not a
  reconstruction.

## Results (2026-07-11)

4 fresh Brains trained on the box (seeds 1–4, `paper8x8_130` = March config exactly, ep130,
2026-07-08→11; survived a mid-run credit lapse + resume, incident in the LOOP-0009 note). The
avg10-reward selector picked **ep130 for all 4** (each Brain's training reward climbed to the end,
as the March Brain did). Each selected ep130 vs its **own saved `brain_init.pt`** (matched control),
evaluated frozen at the eval protocol (16 envs, `decision_interval=1`, 2×100k/800k), scored by the
validated `scripts/score_eval_dir.py`.

**Pooled contrast (trained ep130 vs matched init):**

| rung | trained | init | Δ | t | p |
| --- | --- | --- | --- | --- | --- |
| n=8/arm | 0.5799 | 0.5262 | **+0.0537** | 4.25 | 1.4e-4 |
| **n=16/arm** | **0.5678** (sd .066) | **0.5223** (sd .020) | **+0.0455** | 5.27 | **1.3e-6** |

**Convergence-vs-decay: PASSES** — the estimate is stable (+0.054 → +0.046), not decaying toward 0
(the signature that killed critic_code / auxcode_hi / redo). Not marginal → no n=32 needed.

**Per-seed Δ (n=16), reported verbatim per §Statistical honesty:** seed1 **+0.088** (p=2e-5),
seed2 **+0.045** (p=1.5e-3), seed4 **+0.076** (p=4e-6), seed3 **−0.027** (p=0.046). **3/4 positive.**
The init arm is consistent across all Brains (~0.522, matching the T-series control 0.5285). hit80
~0.97 both arms. seed3 is a genuinely below-init training draw — the honest cross-seed spread.

**Verdict — P-R1a met** (pooled Δ>0 at p≪0.01 **AND** ≥3/4 directionally positive, converged): the
trained-Brain effect is a property of *Brains trained this way*, not the single March seed. The
claim upgrades to **"trained HP meta-control improves post-switch recovery across training seeds."**
Notably the pooled +0.046 **exceeds** note 0006's March single-seed +0.029 — the March Brain was a
slightly *below*-median draw, not a favorable one. Per §Decision matrix (Replicates): CoLLAs-viable;
the write-up-vs-memory-phase fork is now a scheduling choice, not a scientific one (paper skeleton
[note 0008](./0008-paper-skeleton.md) C1 upgrades from existential to across-seed).

## Links

- Effect + selector + protocol being replicated: [note 0006](./0006-controls-axis-thesis-relocated.md).
- Untrained-controller finding (why the scout can't test this): [note 0005](./0005-untested-controller-bottom-rung-oracle.md) Finding 1.
- Campaign/ops: [LOOP-0009](../autoresearch-loops/LOOP-0009-trained-brain-replication.md) ·
  reference config: `runs/brain_2_regimes_8x8_neuromod_20260315-180839/config.txt`.
- Strategy decision (user, 2026-07-08): replicate + draft both; the final fork call happens at
  this note's gate.
