# LOOP-0009 — trained-Brain replication campaign (2026-07-08–OPEN)

**Goal:** Replicate note 0006's +0.0292 trained-vs-init contrast across ≥4 fresh Brain
training seeds — the mandatory fix for any above-workshop write-up, doubling as the baseline
arm for a future memory-lever phase.
**Verdict:** OPEN — pre-registered (note 0007), nothing launched yet.

## Hardware

Previous box (Vast 4× RTX 3060, `76.67.137.57:26061`) **DESTROYED 2026-07-08 by the user**
(archive-safe — everything was off-disk). New box to provision per
[cloud-setup.md](../plans/cloud-setup.md): 4× GPU ≥12 GB, **non-Pascal** (or apply the cu126
downgrade), ≥32 vCPU, ~2 days rental; 1 Brain-training run per GPU (async runs are
compute-bound at ~100% util — pack by utilization, not VRAM). Home 5070 available for the
eval arms (T-series precedent: 16 evals ≈ 2.2 h).

## Code state

`Auto-Research` @ `742ba8c` (run branch `autoresearch-run-20260706` MERGED — all
LOOP-0007/0008 conditions + notes 0005/0006 now in the main tree; run worktree removed; 148
tests pass post-merge).

**Pre-launch desk work — DONE 2026-07-08 (all box-independent):**
- ✅ **`scripts/score_eval_dir.py` recreated + validated (A3 gate).** Promotes the ephemeral
  `scratchpad/score_t1.py`: stages each eval's single `*_data.json` into the frozen scorer's
  run layout (`episode_00/ep0_env0/` + a synthesized `inner_steps_per_regime` config) and calls
  `score_brain_run` unmodified. `--self-test` **reproduces note 0006 exactly**: trained 0.5577,
  init 0.5285, Δ+0.0292, t=2.96 (p=0.005), hit80 0.982 both (32/32 arms). Reusable to score the
  fresh Brains' evals.
- ✅ **`train_brain.py` seed plumbing + init-checkpoint saving (2i) — gap found & fixed.** The
  outer Brain was never explicitly seeded (only the inner PPO was), and no episode-0 init was
  ever saved — March's `brain_init_seed0.pt` was reconstructed post-hoc (dated Jul 7). Fixed:
  fresh runs now `seed_everything(args.seed)` before Brain init and save `brain_init.pt` (the
  matched per-seed init control) before any training. Smoke-tested end-to-end (init saved
  pre-training, run completes, `brain_model.pt` intact); verified inits reproducible per seed &
  distinct across seeds; 148 tests still green.
- ✅ **March config (A5) verified.** Embedded `args` in `episode_50`/`episode_99` checkpoints
  match the resume `config.txt` core (brain_lr 1e-4, brain_num_envs 8 async, decision_interval
  1, inner 16 envs / 800k / 100k, reward_mode recovery, pretrain_episodes 1) — config was
  consistent across episodes 1–130. **Fresh Brains use exactly these params.**

Remaining (box-gated): cut a fresh run branch off **`70f2dd5`** (the tip carrying the seed/init
fix — launching from the pre-fix `742ba8c` would train without `brain_init.pt` + outer seeding and
break the matched-init-control design) before box bootstrap; provision box; launch.

## Runs

Planned (none launched): 4 Brain training runs — seeds 1–4, March config
(`runs/brain_2_regimes_8x8_neuromod_20260315-180839/config.txt`), 130 episodes, per-episode
checkpoints, 1/GPU — then the eval ladder n=8→16→(32) per arm per Brain against matched
per-seed init controls, at the eval protocol (16 inner envs, `decision_interval=1`).

## Statistics & verdicts

Pre-registered gates in [note 0007](../research-notes/0007-trained-brain-replication.md)
(P-R1a primary: pooled Δ>0 p<0.01 AND ≥3/4 Brains directionally positive; P-R1c failure
branch withdraws the positive claim) — committed before any run exists. Convergence-vs-decay
criterion applies at every ladder rung.

## Cost

Est. ~2 box-days (4 × ~40–45 h in parallel) + home eval hours; no agent-$ (master-implemented,
no trial agent).

## Sibling loops

[LOOP-0008](./LOOP-0008-controls-campaign.md) (closed; produced the effect + protocol being
replicated). If the fork resolves toward the memory phase, its brainstorm becomes
**LOOP-0010** (MoWM × Brain memory levers / interface redesign — see the 2026-07-08 hand-off).

## Pickup state

**Pre-launch verifications (was step 2) — ALL DONE 2026-07-08.** (i) seed plumbing + init
checkpoint saving: gap found, **fixed** in `train_brain.py` + smoke-tested; (ii) March config
A5: **verified** consistent across ep1–130; (iii) `scripts/score_eval_dir.py`: **recreated +
self-test reproduces note 0006 exactly**. See Code state above. Committed as the pre-launch
commit (pre-registration + scorer + fix), pushed to `Auto-Research`.

**Remaining — the box directive is the open decision (needs the user):**
1. **⏸ Provision the box** (spec above) — the old box was destroyed; a new 4-GPU rental is
   ~2 box-days of billing. **This is the outward-facing spend that must not be launched
   autonomously — awaiting the user's go / hold / spec call.** Bootstrap from a fresh run branch
   cut off `Auto-Research` @ **`70f2dd5`** (must include the seed/init fix — not `742ba8c`).
2. **Launch** 4 training runs (1/GPU) with thread caps + watcher per cloud-setup;
   sanity-check one training-reward curve against the March shape early (~ep 10–15) before
   committing the full 2 days (note 0007 A1).
3. **While training (desk work, can start before the box):** draft the paper skeleton (note
   0006 §Paper skeleton) and the LOOP-0010 memory-levers pre-registration draft (user curates
   before anything is implemented).
4. **Gate per note 0007.** The strategic fork (write up vs memory phase) is decided AT that
   gate — user decision 2026-07-08 ("replicate + draft both").

## Links

[Note 0007](../research-notes/0007-trained-brain-replication.md) (pre-registration) ·
[note 0006](../research-notes/0006-controls-axis-thesis-relocated.md) (effect) · hand-off
`docs/hand-offs/2026-07-08-loop-0008-findings-strategic-fork.md` ·
[AUTORESEARCH.md](../../AUTORESEARCH.md) register.
