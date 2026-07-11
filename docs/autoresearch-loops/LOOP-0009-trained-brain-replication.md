# LOOP-0009 — trained-Brain replication campaign (2026-07-08–OPEN)

**Goal:** Replicate note 0006's +0.0292 trained-vs-init contrast across ≥4 fresh Brain
training seeds — the mandatory fix for any above-workshop write-up, doubling as the baseline
arm for a future memory-lever phase.
**Verdict:** ✅ CONFIRMED — **P-R1a PASSED, the effect replicates across training seeds.** 4 fresh
Brains trained to ep130 (survived a credit-lapse + resume, incident log below); eval ladder on the
box gave pooled **Δ +0.0455 composite, p=1.3e-6, n=16/arm, 3/4 Brains positive**, converged
(n=8 +0.054 → n=16 +0.046). Larger than March's single-seed +0.029. Full result: [note 0007 §Results](../research-notes/0007-trained-brain-replication.md).
Remaining: pull eval evidence home + archive; box safe to destroy; C1 upgrade flows into the paper
skeleton (note 0008).

## Incident log

- **2026-07-10 — credit lapse at ep124/130.** Vast box stopped when credit ran out (host
  reachable again after top-up; disk persisted, training procs killed). Runs had reached ep124,
  last 5-ep checkpoint ep120. **Actions:** (1) backed up all 4 seeds' `brain_init.pt` +
  `brain_ep*.pt` + trends to `runs/loop9_backup/` (24 MB, off-box safety). (2) First resume
  crashed at startup on **every** seed: `torch.random.set_rng_state → TypeError: RNG state must
  be a torch.ByteTensor` — a torch 2.12+cu130 quirk where the checkpointed CPU RNG state
  round-trips as a non-ByteTensor. Fixed in `restore_brain_rng_state` (coerce to uint8 CPU
  tensor; best-effort with fresh-RNG fallback) — commit `fd51633`. (3) Pulled the fix on the box,
  re-resumed from ep120; all 4 healthy. **Durability gap noted:** `AUTO_PUSH=0` (no GH_TOKEN) +
  no periodic pull meant the near-final Brains were briefly single-copy on the box — mitigated by
  the manual backup; for future long runs, set a GH_TOKEN launch env var or periodic checkpoint pull.

## Hardware

**LAUNCHED 2026-07-08 on Vast `63.142.193.28:31406` — 4× RTX 4060 Ti 16 GB (Ada sm_89), 32
vCPU, 125 GB RAM, 130 GB disk, torch 2.12+cu130 (`/venv/main`).** All 4 seeds run in parallel,
1/GPU, ~3 GB VRAM + ~21 GB RAM each; ~48 h to completion (SPS ~600/stream, ~22 min/episode ×
130). Home 5070 reserved for the eval arms (T-series precedent: 16 evals ≈ 2.2 h).

**Provisioning history (the RAM lesson):** two earlier boxes were destroyed. The Vast 4× RTX
3060 box has **62 GB RAM**, which OOM-killed 3 of 4 runs during pretraining — each
`brain_num_envs=8` async run needs ~21 GB, so 4 × 21 ≈ 84 GB > 62 GB. `brain_num_envs` is fixed
by config fidelity, so the fix was a higher-RAM box (125 GB fits all 4). **Also: launch
staggered (~40 s apart)** to avoid a simultaneous 32-worker CUDA-init race. This workload is
**RAM-bound, not GPU-bound** — size boxes by RAM-per-GPU (≥~24 GB/GPU), not GPU class.

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

**RUNNING (launched 2026-07-08 19:57–19:59Z):** 4 Brain training runs, seeds 1–4, one per GPU,
via the `paper8x8_130` preset + `brain_neuromod` condition (= March config exactly, brain_episodes
130 — verified only-diff vs `paper8x8` is the episode count). Run dirs
`runs/paper8x8_130_brain_neuromod_seed{1..4}_*`; each has its `brain_init.pt` saved. Sweep out:
`sweeps/loop9_s{1..4}/`. Launch shape (documented for resume/repro):

```
CUDA_VISIBLE_DEVICES=$G MAX_PARALLEL=1 AUTO_PUSH=0 RESUME=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  bash scripts/cloud/run_sweep.sh paper8x8_130 "brain_neuromod" "$S" loop9_s$S    # staggered ~40s
```

RESUME=1 (spot-safe: `save_every_episodes=5`, re-run the same command to continue a killed cell).
AUTO_PUSH=0 (no GH_TOKEN on box) → **pull Brain checkpoints to home manually** (they are tiny —
MLP; only `brain_init.pt` + selected `brain_ep*.pt` + `brain_model.pt` are needed, NOT the inner
logs). Then the eval ladder n=8→16→(32) per arm per Brain vs matched per-seed inits, at the eval
protocol (16 inner envs, `decision_interval=1`), scored by `scripts/score_eval_dir.py`.

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

**Box provisioned + launched 2026-07-08** (`63.142.193.28:31406`, see Hardware). Bootstrapped
from run branch `autoresearch-run-20260708` @ `0e2f874` (carries the seed/init fix +
`paper8x8_130`). All 4 seeds training in parallel.

**Remaining:**
1. **⏳ ep10–15 sanity gate (note 0007 A1)** — compare one Brain's early training-reward curve
   against the March shape *before* trusting the full ~48 h. Data source: the
   `Episode N/130 | reward=...` lines in `sweeps/loop9_s*/logs/*.log`. ~3.5–4 h after launch.
2. **Monitor** liveness (4 `train_brain.py` alive, GPUs busy, RAM < 125 GB, disk < 130 GB);
   re-run the same `run_sweep.sh` command for any spot-killed cell (RESUME=1 continues it).
3. **On completion (~48 h):** pull each run's `brain_init.pt` + selected `brain_ep*.pt`
   (avg10-reward selector) + `brain_model.pt` to home; run the eval ladder (n=8→16→(32)/arm) on
   the 5070; score with `scripts/score_eval_dir.py`.
4. **Gate per note 0007** (P-R1a: pooled Δ>0 p<0.01 AND ≥3/4 Brains positive). The strategic
   fork (write up vs memory phase) is decided AT that gate — user decision 2026-07-08
   ("replicate + draft both"); the paper skeleton (note 0008) + LOOP-0010 draft (note 0009) are
   already drafted.

## Links

[Note 0007](../research-notes/0007-trained-brain-replication.md) (pre-registration) ·
[note 0006](../research-notes/0006-controls-axis-thesis-relocated.md) (effect) · hand-off
`docs/hand-offs/2026-07-08-loop-0008-findings-strategic-fork.md` ·
[AUTORESEARCH.md](../../AUTORESEARCH.md) register.
