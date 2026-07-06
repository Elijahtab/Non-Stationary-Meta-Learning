# Cloud Setup — running sweeps on a many-core GPU box

**Why this exists:** the local calibration revealed each `scout5x5` cell takes **~15.5 hr**
(50 serial 800k-step inner runs), and `brain_num_envs=4 async` gave ~no speedup on a single
box — the work is **CPU-bound** MiniGrid env-stepping, so *total core count* is the throughput
lever, not GPU class. Cloud lets cells run **concurrently** across many cores. See the workshop
plan: [workshop-task-free-neuromodulation.md](./workshop-task-free-neuromodulation.md).

> **Key insight:** prioritize **vCPUs over GPU**. The GPU runs one tiny CNN per stream (~1.3 GB
> each), so a single modest GPU serves 8+ concurrent cells. Do **not** pay for an H100 here.

## Recommended instance

| Spec | Recommendation | Why |
| --- | --- | --- |
| vCPU | **32–64** | the actual bottleneck; `max_parallel ≈ vCPU / 8` |
| GPU | RTX 4090 / A40 / even 3090 (24 GB) | tiny models; one GPU serves many streams |
| RAM | ≥ 64 GB | ~2.5 GB per inner worker × concurrency |
| Disk | 30–50 GB | logs/runs; pull results off before terminating |
| Image | **"PyTorch 2.x + CUDA" template** | reuse preinstalled, CUDA-matched torch |

Providers (from the budget research): **RunPod** or **Vast.ai**. On Vast, filter for high
vCPU count + a cheap 24 GB GPU; on RunPod pick a PyTorch template and a high-CPU pod.

> **Old-GPU (Pascal/GTX 10xx) gotcha — verified 2026-07-03 on a 4× GTX 1080 box:** recent
> image torch builds (2.12+cu130) dropped `sm_61`/Pascal kernels — the first GPU op dies with
> `no kernel image is available for execution on the device` even though `nvidia-smi` and
> `torch.cuda.is_available()` look fine. Fix (in-container, ~5 min): downgrade to the last
> Pascal-capable wheel line, `uv pip install "torch==2.7.1" --index-url
> https://download.pytorch.org/whl/cu126`, **before** running `bootstrap.sh` (which reuses
> whatever torch is present). The cu126 wheel carries no sm_61 SASS either but ships PTX the
> (newer) driver JIT-compiles — expect a one-time per-kernel JIT pause, cached afterwards.
> Cheap Pascal boxes are otherwise fine for this workload (~1.6 GB VRAM/cell, CPU-bound).

## Throughput & cost model

- `max_parallel = nproc / CORES_PER_CELL` (default `CORES_PER_CELL=8`).
- Per-cell wall-clock (single stream): **`calib5x5` ≈ 3–4 hr**, **`scout5x5` ≈ 12–16 hr**,
  **`paper8x8` ≈ 20–40 hr**.

| Sweep | Box | Cells | max_parallel | Wall-clock | ~Cost @ $1.5/hr |
| --- | --- | --- | --- | --- | --- |
| `calib5x5` 4-cond × 5-seed | 64 vCPU | 20 | 8 | ~11 hr | ~$17 |
| `scout5x5` 4-cond × 5-seed | 64 vCPU | 20 | 8 | ~40 hr | ~$60 |
| `paper8x8` **confirmation** 2-cond × 3-seed | 64 vCPU | 6 | **2** | ~90 hr | ~$135 |

Strategy: **iterate on `calib5x5`** (cheap, parallel) → confirm survivors on `paper8x8` for
paper-comparable headline numbers. **`paper8x8` barely parallelizes** — `brain_num_envs=8` async
means one cell already uses ~8×16 envs, so set `MAX_PARALLEL=1` or `2` (override the auto value)
and run only the few surviving conditions with 3 seeds, not the full matrix. The full 4×5
`paper8x8` matrix would blow the monthly budget — don't run it.

```bash
# paper8x8 confirmation run: cap concurrency explicitly, few conditions/seeds
MAX_PARALLEL=2 bash scripts/cloud/run_sweep.sh paper8x8 "brain_neuromod brain_no_neuromod" "0 1 2"
```

## Step-by-step

```bash
# 1. SSH into the box, then clone
git clone <your-repo-url> Lifelong-Learning && cd Lifelong-Learning
git checkout Auto-Research   # or the branch with this tooling

# 2. Bootstrap (reuses image torch; skips tensorflow by default)
bash scripts/cloud/bootstrap.sh
#    -> verifies CUDA + imports + runs a tiny pilot smoke

# 3. Launch the sweep (detached; auto-sizes concurrency to nproc)
bash scripts/cloud/run_sweep.sh calib5x5 \
    "brain_neuromod brain_no_neuromod brain_random_code brain_oracle_code" "0 1 2 3 4"

# 4. Watch
tail -f sweeps/calib5x5_*/sweep.out
watch -n5 nvidia-smi

# 5. When done, pull results back to your machine (run locally):
#    scp/rsync the small CSVs (and optionally the run dirs for the probe)
rsync -avz <user>@<host>:Lifelong-Learning/sweeps/ ./sweeps/
#    runs/ are large; for the probe you only need the brain_neuromod run dir(s).
```

## Tuning concurrency

- Start with the default `CORES_PER_CELL=8`. If GPU memory is tight, raise it (fewer parallel
  cells); if CPU is underused (low `nproc` utilization), lower it.
- `MAX_PARALLEL=N bash scripts/cloud/run_sweep.sh ...` overrides the computed value.
- `calib5x5` uses `brain_num_envs=1` (sync) so each cell is a single clean stream — concurrency
  comes from running many cells at once, which is what actually saturates the cores.

## Resume / interruption (auto, spot-safe)

`run_sweep.sh` passes **`--resume` by default** (set `RESUME=0` to force all-fresh). In resume mode:

- The runner **forces periodic Brain checkpoints** (`--save_every_episodes 5`) so an interrupted
  cell is recoverable (each `episode_N/brain_epN.pt` carries the episode counter, RNG, optimizer,
  and MetaEnv normalizer state for a clean continue).
- On (re-)launch, each cell is classified against any prior run dir in `runs/`:
  - **complete** (`brain_model.pt` present) → skip training, just re-score (idempotent).
  - **partial** (has `episode_N/brain_epN.pt`) → continue from the latest checkpoint in the *same*
    folder, with the target `brain_episodes` held fixed (so it runs the remaining episodes only).
  - **empty / none** → run fresh.
- **To recover from a spot kill: just re-run the exact same `run_sweep.sh` command.** Completed
  cells are reused, the killed cell continues, and the rest run. (Resume keys off `runs/<run_name>`,
  so even a new `--out` dir resumes training — you just get fresh summary CSVs.)
- `runs.csv` records each cell's `status` and `resume_action` (`fresh` / `resume` /
  `reused_complete`) for auditability.

Caveat: the **first** run of a cell must have checkpointed at least once (≥5 episodes) before a
kill is recoverable; a cell killed in its first <5 episodes restarts fresh. Lower
`RESUME_CHECKPOINT_EVERY` in `run_seed_sweep.py` if you want finer granularity on `paper8x8`.

## After the sweep

`sweeps/<name>/summary.csv` has per-condition mean ± bootstrap 95% CI for `composite_score` and
the recovery metrics. Then run the headline analysis locally on the pulled `brain_neuromod` run:

```bash
PYTHONPATH=src python scripts/analyze_regime_decoding.py runs/<brain_neuromod_run> --features code
```


## Operational rules absorbed from AUTORESEARCH v1 (2026-07-05 restructure)

Measured, load-bearing — full history in [docs/archive/AUTORESEARCH-v1.md](../archive/AUTORESEARCH-v1.md).

- **Thread caps are mandatory** on many-core boxes: `OMP_NUM_THREADS=8` (+ `MKL`/`OPENBLAS`/
  `NUMEXPR`) per invocation. Uncapped torch → ~131 threads/cell → load 321, GPUs idle. The cap
  is a thread cap, NOT a per-cell core budget (cells use ~1.5 cores sync / ~6 cores async).
  Note: `nproc` respects `OMP_NUM_THREADS`, so `run_sweep.sh` will print `cores=8` — harmless
  with explicit `MAX_PARALLEL`.
- **Packing densities (measured):** sync calib8x8 cells ~1.6 GB VRAM → ~3/GPU on 24 GB (3090,
  util 23–61%). Async scoutv2 cells ~1.45 GB VRAM but **compute-bound at 99–100% util → 1
  cell/GPU regardless of VRAM** (RTX 3060, 2026-07-05; SPS ≈ 544/stream, ~9.9 h for 6
  sequential cells). Budget by GPU *utilization*, not VRAM or vCPU.
- **Durability:** assume `workspace_is_volume: false` — push results (`AUTO_PUSH=1` → `results`
  branch) at every wave; never accumulate unpushed results on an ephemeral disk; destroy idle
  boxes (recreation ≈ 20 min via bootstrap).
- **Launch shape:** one `run_sweep.sh` invocation per GPU via `CUDA_VISIBLE_DEVICES=N`,
  `MAX_PARALLEL` explicit, `RESUME=1` for spot safety. SSH gotchas: filter the 3-line vast
  banner; non-interactive shells don't activate `/venv/main`; kill with the bracket trick
  (`pkill -9 -f "[t]rain_brain.py"`).
- **Linux + async Brain configs require the spawn fix** (in code since `e756bad`): fork was
  never viable (`Cannot re-initialize CUDA in forked subprocess`); anything reverting
  `context="spawn"` breaks every Linux box run.
