# 0002 — `calib8x8` cloud sweep: execution plan

- **Date:** 2026-06-18
- **Status:** Accepted — execution gated on the local pilot ([0001](./0001-2026-06-18-calibration-scoring.md))
- **Touches:** infra/ops only (cloud box + `scripts/cloud/`). No source changes. Builds on
  [docs/plans/cloud-setup.md](../plans/cloud-setup.md) with `calib8x8`-specific corrections.

## Context

[0001](./0001-2026-06-18-calibration-scoring.md) re-scored the calibration to pure post-switch
success and added the harder `calib8x8` preset. We need to re-run the 4-condition × 3-seed matrix
(12 cells) on this preset to see whether 8×8 de-saturates the scorer and the conditions finally
separate (`oracle ≥ neuromod > random > no_neuromod`).

**Why not local:** the work is CPU-bound MiniGrid env-stepping, and `calib8x8` sets
`inner_num_envs=16`, so **one cell saturates the local 8-core / 16-thread box**. The full matrix
would be effectively serial (~3–6 days, machine pinned). Cloud adds cores → cells run concurrently.
GPU is *not* the constraint: the pilot measured ~0.5 GB VRAM and ~6% util per cell.

## Decision

Run the 12-cell `calib8x8` matrix on a **high-vCPU, cheap-GPU** cloud box, using the existing
[`scripts/cloud/run_sweep.sh`](../../scripts/cloud/run_sweep.sh) tooling, with one mandatory
override: **`CORES_PER_CELL=16`**.

### Gate (must pass before provisioning)

Read the pilot score (`sweeps/calib8x8_pilot/.../seed_0_score.json` or the run's score report):

- **De-saturation check:** `hit_rate_80 < ~0.95` **and** `mean_post_switch_window_success_rate`
  clearly off the ceiling (headroom exists). If instead 8×8 *floors* (success ≈ 0, no recovery),
  fix the preset (more inner steps / exploration) **before** cloud, not after.
- **Record `T`** = per-cell wall-clock. Drives box sizing + cost below.

### Prerequisite — DONE

Branch pushed: commit `72aac91` on `origin/Auto-Research`. Canonical remote (note capital **E** —
the repo moved): `https://github.com/Elijahtab/Non-Stationary-Meta-Learning.git`.

### Box spec (prioritize vCPU, never an H100)

> **⚠️ Superseded (2026-07-01).** This section assumed ~16 cores/cell (`max_parallel = vCPU/16`).
> Measured reality: a `calib8x8` cell uses **~1.5 cores**, so 12 cells burned only ~14–16 of 128
> vCPU. Provision **~32 vCPU + ~64 GB + 4× GPU** (GPU util, not vCPU, is the limit) and set
> `OMP_NUM_THREADS=8`. See [cloud-setup.md](../plans/cloud-setup.md) → "Operational rules" (absorbed from AUTORESEARCH v1) and
> `docs/multi_agent/0001`. The table below is kept for the record only.

| Spec | Pick | Why |
| --- | --- | --- |
| vCPU | **48–64** | the bottleneck; `max_parallel = vCPU / 16` for `calib8x8` |
| GPU | cheap 24 GB (3090 / A40 / 4090) | one GPU hosts all streams; ~0.5 GB/cell observed |
| RAM | ≥ 64 GB | 16 envs/cell × parallel cells |
| Disk | 30–50 GB | logs/runs; pull results before terminating |
| Image | PyTorch 2.x + CUDA template | reuse driver-matched torch |

Provider: **RunPod** (PyTorch template + high-CPU pod) or **Vast.ai** (filter high vCPU + cheap
24 GB GPU). Spot/interruptible is fine — the sweep is resume-safe (see below).

### Run commands

```bash
# 1. Clone (use the CAPITALIZED moved URL, not the redirect)
git clone https://github.com/Elijahtab/Non-Stationary-Meta-Learning.git Lifelong-Learning
cd Lifelong-Learning && git checkout Auto-Research

# 2. Bootstrap (reuses image torch, verifies imports, runs a tiny pilot smoke)
bash scripts/cloud/bootstrap.sh

# 3. Launch — CORES_PER_CELL=16 is MANDATORY for calib8x8 (see note)
#    15 cells: 4 neuromod conditions on recovery_v2 + brain_neuromod_recovery (reward A/B control).
CORES_PER_CELL=16 bash scripts/cloud/run_sweep.sh calib8x8 \
    "brain_neuromod brain_no_neuromod brain_random_code brain_oracle_code brain_neuromod_recovery" "0 1 2"

# 4. Watch
tail -f sweeps/calib8x8_*/sweep.out
```

> **`CORES_PER_CELL=16` is the must-not-forget flag.** `run_sweep.sh` defaults to 8, which was
> calibrated for the 5×5 `brain_num_envs=1` presets. `calib8x8` uses `inner_num_envs=16`, so the
> default would over-subscribe each cell 2× and slow the whole sweep. → `max_parallel = nproc/16`
> (e.g. 4 on a 64-vCPU box).

### Resume / spot-safety

`run_sweep.sh` passes `--resume` by default: forces periodic Brain checkpoints and, on re-launch,
reuses completed cells / continues partial ones / starts the rest. **To recover from a spot kill,
re-run the exact same command.** Caveat: a cell killed in its first <5 episodes restarts fresh.

### Pull results back (from your machine)

```bash
rsync -avz <host>:Lifelong-Learning/sweeps/ ./sweeps/        # small CSVs
# pull the brain_neuromod run dir only if you want the regime-decoding probe
```

## Cost model (parametrized on pilot `T`)

15 cells (pilot measured **`T` ≈ 3.83 h**), `max_parallel = vCPU/16`, waves = `ceil(15 / max_parallel)`:

| Box | max_parallel | Waves | Wall-clock (`T`=3.83 h) | Cost @ ~$1.5/hr |
| --- | --- | --- | --- | --- |
| 32 vCPU | 2 | 8 | ~30.6 h | ~$46 |
| 48 vCPU | 3 | 5 | ~19.2 h | ~$29 |
| 64 vCPU | 4 | 4 | ~15.3 h | ~$23 |

Total core-hours (≈ cost) is roughly constant across boxes; **more vCPUs just buy shorter
wall-clock.**

### Open choices (decide after pilot)

1. Provider — RunPod vs Vast.
2. Box size — 64 vCPU (~3 waves, fastest) vs 32 (cheaper/hr, ~6 waves). Same total cost.
3. Spot vs on-demand — spot ~2–3× cheaper; we're resume-safe.

## Outcome

**Pilot (2026-06-18, local, `brain_neuromod` seed 0, `recovery_v2`): de-saturation gate PASSED.**
`sweeps/calib8x8_pilot` / `runs/calib8x8_brain_neuromod_seed0_20260618-112612`:

| metric | calib5×5 (`recovery`) | calib8×8 (`recovery_v2`) |
| --- | --- | --- |
| composite (= post-switch success) | ~0.75 | **0.516** |
| hit_rate_80 | ~0.99 (pinned) | 0.905 |
| hit_rate_95 | ~0.85 | 0.481 |
| median_steps_to_80 | ~25k | 49k |
| median_steps_to_95 | ~52k | 75k |

8×8 opens real headroom in every term — benchmark is now discriminative. **Per-cell `T` ≈ 3.83 h**
(13,788 s). Cloud estimate: 64 vCPU / max_parallel=4 → ~11.5 h, ~$17–23; 32 vCPU / =2 → ~23 h, ~$35.

Oscillation diagnostic was directionally better (osc ratio 11.9×→4.3×) but **confounded**
(decision_interval 1 vs 10; the matched `recovery`+calib8×8 control was deleted). Deferring the
clean reward A/B to the sweep — see [0003](./0003-2026-06-18-recovery-reward-oscillation-fix.md).

_(Still to fill after the cloud sweep: chosen box/provider/cost and the per-condition
`composite_score` table. Success = `oracle ≥ neuromod > random > no_neuromod` with non-overlapping CIs.)_

## Follow-ups

- [ ] Pilot passes the de-saturation gate; record `T`.
- [ ] Provision box; bootstrap; launch with `CORES_PER_CELL=16`.
- [ ] Pull CSVs; record per-condition composite table in Outcome (here) and in [0001](./0001-2026-06-18-calibration-scoring.md).
- [ ] If signal confirmed → define the final-benchmark composite (post-switch + 0.95 facet) as a
      new record, and plan a `paper8x8` confirmation run for the surviving conditions.
