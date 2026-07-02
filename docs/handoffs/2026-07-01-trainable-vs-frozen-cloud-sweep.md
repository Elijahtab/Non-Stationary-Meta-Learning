# 0004 — Trainable-vs-Frozen Decoder Cloud Sweep: Handoff

- **Date:** 2026-07-01 · **Status:** ✅ COMPLETE — H1 not supported (trainable ≈ frozen; see §1)
- **Purpose:** hand a *separate* Claude session everything needed to (a) SSH into the Vast box,
  (b) read/pull the results, and (c) re-run or extend the sweep — plus the obstacles hit so they
  aren't rediscovered the hard way.
- **Experiment:** frozen vs **trainable** neuromodulation decoder — see the research note
  [docs/research-notes/0001-trainable-vs-frozen-decoder.md](../research-notes/0001-trainable-vs-frozen-decoder.md).

---

## 1. Findings — ✅ RESOLVED (2026-07-01)

**H1 (trainable ≥ frozen) is NOT supported.** Making the neuromodulation decoder trainable did
not improve post-switch recovery: on the primary metric it is statistically indistinguishable
from frozen, and it is *worse* on threshold reliability (`hit_rate_80`).

calib8x8, metric = `composite_score` (post-switch window success). **4 seeds** (cloud seeds 1–4;
the local seed-0 run **died** ~11 min in — machine slept at episode 1, log stale, excluded):

| Condition | composite mean [95% CI] | hit_rate_80 | n |
| --- | --- | --- | --- |
| frozen (`brain_neuromod`) | **0.5105** [0.499, 0.523] | **0.926** | 4 |
| trainable (`brain_neuromod_trainable`) | **0.5092** [0.505, 0.513] | **0.799** | 4 |

- **Δ(trainable − frozen) = −0.0013** on composite — within noise; the 95% CIs overlap heavily.
- **Trainable's `hit_rate_80` is lower (0.80 vs 0.93):** same *average* post-switch success but it
  reaches the 80% threshold *less reliably* — consistent with the two-learners-co-adapt
  instability flagged as R1 in [note 0001](../research-notes/0001-trainable-vs-frozen-decoder.md).
- **H3 confirmed** (pilot): decoder weight norm drifts only in the trainable condition.
- Per-seed composite — frozen: `[0.500, 0.526, 0.497, 0.519]`; trainable: `[0.510, 0.503, 0.514, 0.509]`.

**Interpretation (outcome branch 2/3 of note 0001):** even *fully learned*, this feature-gating
neuromodulation does not improve recovery on this benchmark — reinforcing that the Brain's
scalar-hyperparameter control, not the context-code routing, drives adaptation here. Clean
negative result: the frozen decoder is a fair baseline, and "just unfreeze it" is shown *not* to
be the missing ingredient. Next levers (per the plan) before declaring neuromod inert:
affine/gain masks, input-conditioned (FiLM) modulation, actor/critic-separate modulation, or a
stability fix (separate/smaller decoder LR).

---

## 2. The cloud box

| | |
| --- | --- |
| Provider | Vast.ai (on-demand), instance `C.43295752` / container `190ff3a03817` |
| Hardware | **128 vCPU** (EPYC 7B12), **251 GB RAM**, **4× RTX 3090** (24 GB each) |
| Cost | ~$0.724/hr |
| SSH | `root@209.33.172.149` port **12338** (container `22/tcp` maps to host 12338) |
| Image | Vast "PyTorch" template; venv at **`/venv/main`** (torch **2.12.0+cu130**, CUDA OK) |
| Repo | **`/workspace/Lifelong-Learning`**, branch **`trainable-vs-frozen-sweep`** @ `14f5b09` |

> **This box was over-sized.** Measured usage (12 cells): **~14–16 of 128 vCPU**, 31/251 GB RAM,
> ~1.6/24 GB VRAM per GPU. Next time provision **~32 vCPU + ~64 GB + 4× GPU** (a cheaper GPU class
> is fine — VRAM need is ~1.6 GB/cell) and set `OMP_NUM_THREADS=8`. Size vCPU ≈ 2× max concurrent
> cells. See `AUTORESEARCH.md` → "Right-sizing the box" for the full rationale.

## 3. SSH from a separate Claude session

The authorized private key is **`~/.ssh/id_ed25519` on the original local Windows machine**
(its `.pub` was appended to the box's `~/.ssh/authorized_keys`). A session on that same machine
can connect directly. From elsewhere, either copy that key or append a new pubkey to the box via
an existing session.

**Connection template (note the non-interactive flags + banner filter):**
```bash
SSHO="-p 12338 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
ssh $SSHO root@209.33.172.149 '<remote command>' 2>&1 | grep -vE "Welcome to vast|Have fun|AI agents"
```
- The box prints a 3-line vast banner on every login — **filter it** with the `grep -vE` above.
- **Non-interactive shells do NOT auto-activate the venv.** Prefix remote commands that use
  Python with `cd /workspace/Lifelong-Learning && source /venv/main/bin/activate && …`
  (after which `python`/`python3` → `/venv/main/bin/*`, which has torch).

## 4. Check progress / pull results

Four detached sweeps write to `sweeps/tvf_g{0,1,2,3}/`:

| dir | GPU | condition | seeds |
| --- | --- | --- | --- |
| `tvf_g0` | 0 | frozen (`brain_neuromod`) | 1, 2 |
| `tvf_g1` | 1 | frozen | 3, 4 |
| `tvf_g2` | 2 | **trainable** | 1, 2 |
| `tvf_g3` | 3 | **trainable** | 3, 4 |

```bash
# progress: per-cell latest update + SPS
ssh $SSHO root@209.33.172.149 'for d in /workspace/Lifelong-Learning/sweeps/tvf_g*; do for f in $d/logs/*.log; do echo "$(basename $f .log|sed s/calib8x8_//): $(grep -oE "update [0-9]+/390|SPS=[0-9]+" $f|tail -2|tr "\n" " ")"; done; done' 2>&1 | grep -vE "Welcome to vast|Have fun|AI agents"

# done? each group writes summary.csv at the end
ssh $SSHO root@209.33.172.149 'ls /workspace/Lifelong-Learning/sweeps/tvf_g*/summary.csv 2>/dev/null | wc -l'   # ==4 means complete

# pull the per-cell scores (composite_score column) from all 4 groups
ssh $SSHO root@209.33.172.149 'for d in /workspace/Lifelong-Learning/sweeps/tvf_g*; do echo "== $d =="; cat $d/runs.csv; done' 2>&1 | grep -vE "Welcome to vast|Have fun|AI agents"
```

## 5. The WORKING launch command (if you must re-run)

⚠️ Use **exactly** this shape — the plain `run_sweep.sh` defaults thrash on this box (see §6):
```bash
cd /workspace/Lifelong-Learning && source /venv/main/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=0 MAX_PARALLEL=2 RESUME=0 bash scripts/cloud/run_sweep.sh calib8x8 "brain_neuromod" "1 2" tvf_g0
CUDA_VISIBLE_DEVICES=1 MAX_PARALLEL=2 RESUME=0 bash scripts/cloud/run_sweep.sh calib8x8 "brain_neuromod" "3 4" tvf_g1
CUDA_VISIBLE_DEVICES=2 MAX_PARALLEL=2 RESUME=0 bash scripts/cloud/run_sweep.sh calib8x8 "brain_neuromod_trainable" "1 2" tvf_g2
CUDA_VISIBLE_DEVICES=3 MAX_PARALLEL=2 RESUME=0 bash scripts/cloud/run_sweep.sh calib8x8 "brain_neuromod_trainable" "3 4" tvf_g3
```
Healthy state: **8 cells, load < ~30, each GPU ~55–65% util, ~1300–1400 SPS/cell, ~5 h ETA.**

## 6. Obstacles hit (READ before touching the run)

1. **Torch is thread-UNCAPPED in sync mode** (`get_meta_env_runtime_cpu_threads("sync")→None`).
   Each cell spawned ~131 threads → 8 cells → **load 321, GPU idle, no progress**. **Fix:**
   `export OMP_NUM_THREADS=16` (+ MKL/OPENBLAS/NUMEXPR). Cells drop to ~39 threads, load sane.
2. **`nproc` honors `OMP_NUM_THREADS`.** After the cap, `nproc`→16, so `run_sweep.sh`'s
   `max_parallel = nproc/CORES_PER_CELL` collapsed to **1** (serial). **Fix:** set `MAX_PARALLEL`
   explicitly (it's an override in `run_sweep.sh`).
3. **`pkill -f run_seed_sweep` / `pkill -f train_brain.py` kills your own SSH shell** — the remote
   `bash -c '…'` command line *contains* those strings, so pkill matches itself. This looked like
   "the connection keeps dropping." **Fix:** the bracket trick →
   `pkill -9 -f "[r]un_seed_sweep"`, `pkill -9 -f "[t]rain_brain.py"` (also for `grep`/`pgrep`
   counts: `grep -c "[t]rain_brain.py"`).
4. **All cells default to `cuda:0`** (`--device cuda`) → **GPU 0 pegged at 99% while GPUs 1–3
   idle**, ~690 SPS/cell. **Fix:** run 4 sweeps, one per GPU via `CUDA_VISIBLE_DEVICES=0..3`,
   2 cells each → ~1360 SPS/cell (**2× faster**), all 4 GPUs ~58%.
5. **Vast login banner** (3 lines) pollutes every SSH stdout — filter with
   `grep -vE "Welcome to vast|Have fun|AI agents"`.
6. **`RESUME=0`** was used (fresh, no checkpoints) → a Vast interruption restarts from scratch.
   For long/interruptible runs use `RESUME=1`.
7. There's an `/etc/vast-agents-guide.md` (aka `./AGENTS.md`) on the box — the vast operating
   guide for agents; worth a read if doing anything unusual on the instance.

## 7. Aggregation + teardown

- Full 5-seed comparison = cloud `tvf_g0..g3` (seeds 1–4) **+** the **local** run
  `sweeps/local_trainable_vs_frozen/` (seed 0, on the RTX 5070) for both conditions.
- Aggregate: pool `composite_score` per condition across seeds, report mean ± bootstrap 95% CI
  (same logic as `run_seed_sweep.py`'s `_bootstrap_ci`).
- **Before terminating the Vast pod:** pull the 4 `runs.csv`/`summary.csv` locally (the pod disk
  is ephemeral). Then destroy the instance to stop billing.
