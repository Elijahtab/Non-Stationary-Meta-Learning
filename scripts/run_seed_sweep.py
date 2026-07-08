#!/usr/bin/env python
"""
Seed-sweep runner for the task-free neuromodulation workshop study.

Runs a matrix of (condition x seed) cells, scores each finished run with the frozen
scorer (``score_brain_run``), and aggregates per-condition metrics with bootstrap 95%
confidence intervals. Produces a tidy per-run CSV plus a per-condition summary
(CSV + JSON), so headline tables/figures come straight out of the data.

This is research scaffolding, NOT part of the frozen benchmark surface.

Examples
--------
# Validate the whole pipeline fast (tiny config, 2 seeds, ~minutes):
python scripts/run_seed_sweep.py --preset pilot --seeds 0 1 --out sweeps/pilot

# Print the plan without running anything:
python scripts/run_seed_sweep.py --preset scout5x5 --seeds 0 1 2 3 4 --dry-run

# Real Phase-1 headline conditions:
python scripts/run_seed_sweep.py --preset scout5x5 --seeds 0 1 2 3 4 \
    --conditions brain_neuromod brain_no_neuromod brain_random_code brain_oracle_code \
    --out sweeps/phase1_headline
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lifelong_learning.research.benchmarking import score_brain_run  # noqa: E402

DEFAULT_RUN_ROOT = REPO_ROOT / "runs"

# When --resume is on, force periodic Brain checkpoints so an interrupted cell is recoverable.
RESUME_CHECKPOINT_EVERY = 5

# --- Shared base configs per preset (flags passed to the training script) ---------------
PRESETS: dict[str, dict] = {
    # Tiny config to validate the sweep end-to-end in minutes.
    "pilot": {
        "script": "train_brain",
        "base": {
            "env_id": "MiniGrid-MultiGoal-5x5-v0",
            "num_regimes": 2,
            "inner_total_timesteps": 40000,
            "inner_steps_per_regime": 20000,
            "brain_episodes": 2,
            "brain_num_envs": 1,
            "brain_vectorization": "sync",
            "pretrain_episodes": 0,
            "decision_interval": 10,
            "reward_mode": "recovery",
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
        },
        # scorer settings (match the tiny config so recovery can register)
        "score": {"sustained_points_required": 2, "post_switch_window_ratio": 0.5, "post_switch_buffer_steps": 200},
    },
    # The real Phase-1 5x5 study config (mirrors fast_switch_scout_v1).
    "scout5x5": {
        "script": "train_brain",
        "base": {
            "env_id": "MiniGrid-MultiGoal-5x5-v0",
            "num_regimes": 2,
            "inner_total_timesteps": 800000,
            "inner_steps_per_regime": 100000,
            "brain_episodes": 50,
            "brain_num_envs": 4,
            "brain_vectorization": "async",
            "pretrain_episodes": 0,
            "decision_interval": 10,
            "reward_mode": "recovery",
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
        },
        "score": {"sustained_points_required": 3, "post_switch_window_ratio": 0.5, "post_switch_buffer_steps": 500},
    },
    # Lighter config for fast iteration: half the inner steps, fewer Brain episodes, and
    # brain_num_envs=1 (sync) so cells parallelize cleanly via --max-parallel instead of
    # relying on async vectorization (which gave ~no speedup on a single box). ~3-4x faster
    # per cell than scout5x5; reserve scout5x5 for final headline numbers.
    "calib5x5": {
        "script": "train_brain",
        "base": {
            "env_id": "MiniGrid-MultiGoal-5x5-v0",
            "num_regimes": 2,
            "inner_total_timesteps": 400000,
            "inner_steps_per_regime": 100000,
            "brain_episodes": 30,
            "brain_num_envs": 1,
            "brain_vectorization": "sync",
            "pretrain_episodes": 0,
            "decision_interval": 10,
            "reward_mode": "recovery",
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
        },
        "score": {"sustained_points_required": 3, "post_switch_window_ratio": 0.5, "post_switch_buffer_steps": 500},
    },
    # Harder calibration: paper8x8's task + inner-exploration settings, but calib5x5's cheap
    # execution model (brain_num_envs=1 sync so cells parallelize via --max-parallel,
    # decision_interval=10, no pretrain, 30 brain episodes). The 5x5 calibration saturated the
    # scorer (everyone recovers; hit_rate_80~0.99) so conditions were indistinguishable; 8x8 is
    # meant to open post-switch headroom so the signal-detection sweep can actually separate
    # conditions before we trust the autoresearch loop. See docs/research-log/0001. NOTE: the
    # inner_total_timesteps / inner_num_envs here are a best-guess starting point — validate with
    # a single pilot cell (one condition, one seed) before committing a full 12-cell sweep, and
    # adjust if 8x8 over- or under-saturates.
    "calib8x8": {
        "script": "train_brain",
        "base": {
            "env_id": "MiniGrid-MultiGoal-8x8-v0",
            "num_regimes": 2,
            "inner_total_timesteps": 800000,
            "inner_steps_per_regime": 100000,
            "inner_num_envs": 16,
            "inner_num_steps": 128,
            "inner_mode": "dyna",
            "inner_intrinsic_coef": 0.015,
            "inner_imagined_horizon": 10,
            "inner_wm_lr": 0.0001,
            "brain_episodes": 30,
            "brain_num_envs": 1,
            "brain_vectorization": "sync",
            "pretrain_episodes": 0,
            "decision_interval": 10,
            "reward_mode": "recovery_v2",
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
        },
        "score": {"sustained_points_required": 3, "post_switch_window_ratio": 0.5, "post_switch_buffer_steps": 500},
    },
    # Paper-faithful config: matches the paper's neuromodulated Brain run
    # (runs/brain_2_regimes_8x8_neuromod) so headline numbers are directly comparable to the
    # paper (0.7498 reference). 8x8 grid, 800k/100k inner, inner_num_envs=16, brain_num_envs=8
    # async, decision_interval=1 (Brain acts every inner update), pretrain warm-start, 100k
    # episodic memory. EXPENSIVE: each cell ~20-40 hr and brain_num_envs=8 async means one cell
    # already uses ~8x16 envs, so keep --max-parallel low (1-2 even on a big box). The paper's
    # neuromod number used brain_episodes=130 (resumed); 50 is the pragmatic default here for a
    # multi-seed sweep — bump to 130 only for a final single-config confirmation.
    "paper8x8": {
        "script": "train_brain",
        "base": {
            "env_id": "MiniGrid-MultiGoal-8x8-v0",
            "num_regimes": 2,
            "inner_total_timesteps": 800000,
            "inner_steps_per_regime": 100000,
            "inner_num_envs": 16,
            "inner_num_steps": 128,
            "inner_mode": "dyna",
            "inner_intrinsic_coef": 0.015,
            "inner_imagined_horizon": 10,
            "inner_wm_lr": 0.0001,
            "brain_episodes": 50,
            "brain_num_envs": 8,
            "brain_vectorization": "async",
            "pretrain_episodes": 1,
            "pretrain_mode": "recovery",
            "decision_interval": 1,
            "reward_mode": "recovery",
            "episodic_memory_capacity": 100000,
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
        },
        "score": {"sustained_points_required": 3, "post_switch_window_ratio": 0.5, "post_switch_buffer_steps": 500},
    },
}

# LOOP-0009 replication config: paper8x8 at the paper's FULL 130-episode horizon (the March
# run's brain_episodes; paper8x8's base uses 50 as the cheaper multi-seed default). Derived from
# paper8x8 so it is provably identical except brain_episodes -> the config note 0007 requires for
# the fresh replication Brains (seeds 1-4). See docs/research-notes/0007 + LOOP-0009.
PRESETS["paper8x8_130"] = {
    **PRESETS["paper8x8"],
    "base": {**PRESETS["paper8x8"]["base"], "brain_episodes": 130},
}

# scoutv2: mirrors the frozen fast_switch_scout_v2 benchmark EXACTLY (copied
# programmatically so preset and spec cannot drift) for multi-seed confirmation sweeps of
# signals found by the autoresearch loop at n=1 — see docs/plans/2026-07-05. The seeds vary
# per sweep invocation; everything else is the benchmark's own config, so sweep cells sample
# the same distribution the n=1 signal came from.
from lifelong_learning.research.benchmarking import get_frozen_benchmark  # noqa: E402

_SCOUT_V2 = get_frozen_benchmark("fast_switch_scout_v2")
PRESETS["scoutv2"] = {
    "script": "train_brain",
    "base": {
        **_SCOUT_V2.fixed_train_args,
        "save_every_episodes": 0,
        "plot_every_episodes": 0,
    },
    "score": {
        "sustained_points_required": _SCOUT_V2.sustained_points_required,
        "post_switch_window_ratio": _SCOUT_V2.post_switch_window_ratio,
        "post_switch_buffer_steps": _SCOUT_V2.post_switch_buffer_steps,
    },
}

# --- Conditions: per-condition flag overrides on top of the preset base ------------------
# Scalar HP levers are always Brain-controlled; only the neuromodulation code is swapped.
CONDITIONS: dict[str, dict] = {
    "brain_neuromod": {},  # method: Brain controls levers + learned code
    "brain_no_neuromod": {"disable_neuromodulation": True},  # ablation: code ignored
    "brain_random_code": {"context_code_source": "random"},  # control: code content
    "brain_oracle_code": {"context_code_source": "oracle"},  # upper bound: hidden regime
    "brain_zero_code": {"context_code_source": "zero"},  # identity mask sanity check
    # Reward A/B: same as brain_neuromod but on the OLD recovery reward (preset base now uses
    # recovery_v2). Matched control to isolate the reward's effect. See docs/research-log/0003.
    "brain_neuromod_recovery": {"reward_mode": "recovery"},
    # Trainable decoder: the inner agent learns to interpret the Brain's code (decoder
    # co-adapts) instead of the frozen random projection. Head-to-head vs brain_neuromod.
    # See docs/research-notes/0001-trainable-vs-frozen-decoder.md.
    "brain_neuromod_trainable": {"trainable_neuromod": True},
    # --- Stabilizing the trainable decoder (docs/multi_agent/0002) -----------------------
    # Trainable ≈ frozen on composite but UNSTABLE (late collapse; growing variance = R1
    # co-adaptation). These test fixes, all compared against the existing brain_neuromod
    # (frozen) + brain_neuromod_trainable baselines already on the `results` branch.
    #   slowbrain : slow the controller so the code is a more stationary target for the decoder
    #   long      : just run longer (control for the "needs more time" hypothesis)
    #   slow_long : stabilize AND extend (predicted winner)
    #   declr     : give the decoder its own small LR, decoupled from the Brain-controlled inner LR
    "brain_neuromod_trainable_slowbrain": {"trainable_neuromod": True, "brain_lr": 3e-5},
    "brain_neuromod_trainable_long": {"trainable_neuromod": True, "brain_episodes": 60},
    "brain_neuromod_trainable_slow_long": {"trainable_neuromod": True, "brain_lr": 3e-5, "brain_episodes": 60},
    "brain_neuromod_trainable_declr": {"trainable_neuromod": True, "neuromod_decoder_lr": 1e-5},
    # Combined stabilizer: decouple the decoder LR AND slow the controller — tests whether the two
    # cleanest levers stack (the "belt-and-suspenders" targeted fix). Run in spare box capacity.
    "brain_neuromod_trainable_declr_slowbrain": {"trainable_neuromod": True, "neuromod_decoder_lr": 1e-5, "brain_lr": 3e-5},
    # Frozen decoder at the 60-ep horizon: the CONTROL for the trainable_long climb (note 0002
    # preview). If frozen also keeps rising past ep 30, the climb is about horizon, not the
    # trainable decoder.
    "brain_neuromod_long": {"brain_episodes": 60},
    # --- Mechanism-family confirmation arms (docs/plans/2026-07-05) ----------------------
    # Both showed the hit_rate_80 signature at n=1 on fast_switch_scout_v2 (composite flat,
    # threshold reliability up: actor-only +8.9pts incl. a +1.2pt 3-seed holdout echo;
    # gain-a0.5 +10.7pts). Run with --preset scoutv2 against brain_neuromod as control.
    "brain_neuromod_actor_only": {"neuromod_actor_only": True},
    "brain_neuromod_gain05": {"neuromod_gain_alpha": 0.5},
    # --- LOOP-0006 learning-dynamics family (queue entries 1-4; note 0003 + loop note) ----
    # Code -> learning dynamics instead of forward-path modulation. Screen at n=1 (seed 1)
    # vs the LOOP-0004 n=8 frozen control, extend seeds per research-log 0007 decision rules.
    "brain_neuromod_gradgate": {"neuromod_grad_gate": True},
    "brain_neuromod_critic_code": {"neuromod_critic_code": True},
    "brain_neuromod_auxcode": {"neuromod_aux_code_coef": 0.05},
    "brain_neuromod_adamflush": {"neuromod_adam_flush_threshold": 0.25},
    # --- Idle-capacity exploration variants (log 0007 addendum; explore_* out dirs) -------
    # Flag combos of the mechanisms above — no new code paths. n=1 screens only.
    #   gradgate_gain: two-sided plasticity gate in [0.5, 1.5] (gain decode feeds the
    #                  backward gate) — the Brain can AMPLIFY learning, not just protect.
    #   critic_code_gradgate: queue entries 1+2 stacked (independent by construction).
    #   auxcode_hi / adamflush_lo: dose variants of entries 3-4.
    "brain_neuromod_gradgate_gain": {"neuromod_grad_gate": True, "neuromod_gain_alpha": 0.5},
    "brain_neuromod_critic_code_gradgate": {"neuromod_grad_gate": True, "neuromod_critic_code": True},
    "brain_neuromod_auxcode_hi": {"neuromod_aux_code_coef": 0.15},
    "brain_neuromod_adamflush_lo": {"neuromod_adam_flush_threshold": 0.1},
    # --- Reliability-fork follow-ups (LOOP-0006 gate 2026-07-06; RUN-20260706 living doc) ----
    # The two DURABLE leads were reliability-signature wins (composite flat, hit_80 > baseline
    # CI): critic_code (holdout-CONFIRMED) and auxcode_hi (n=3 signature). Their n=1 "composite
    # leads" were noise; gradgate_gain's composite lead FAILED holdout. So we deepen the
    # reliability axis, not composite. Screen n=1 per fork; predictions in log 0007 addendum.
    #   critic_code_auxhi: do the two reliability winners STACK? (independent code paths:
    #     critic reads code as input; aux adds a code-prediction loss). Predict hit_80 ≥ the
    #     better single (~0.86) with composite still flat — additive reliability if mechanisms
    #     are complementary, sub-additive if they tap the same signal.
    #   auxcode coef curve: 0.05 was KILLED, 0.15 wins on reliability. Map the dose-response
    #     between/above to locate the reliability optimum (0.10 interpolates, 0.25 pushes).
    "brain_neuromod_critic_code_auxhi": {"neuromod_critic_code": True, "neuromod_aux_code_coef": 0.15},
    "brain_neuromod_auxcode_010": {"neuromod_aux_code_coef": 0.10},
    "brain_neuromod_auxcode_025": {"neuromod_aux_code_coef": 0.25},
    # --- LOOP-0007: code-free plasticity & stability (research note 0004) ---------------------
    # Nothing routes the regime code into the inner agent (dead across two families). These
    # attack the two MEASURED pathologies directly. Screen at n≥8 from the start (the n=3 lesson).
    #   critic_lr_lo (cand 1): critic head in its own optimizer group at LR = main_lr * 0.5, to
    #     DAMP the measured post-switch critic whiplash (|ΔV|≈0.92 vs policy-KL≈0.001). The
    #     Brain's proven LR lever still moves the critic, halved. Predict composite ↑ via
    #     less-biased GAE + hit_80 ↑. Code-free analogue of the null critic_code.
    #   encoder_lr_lo (cand 5): shared encoder in its own group at LR = main_lr * 0.5, so
    #     features stay stable across regimes while heads re-map fast. Code-free, uniform
    #     analogue of the null gradgate — isolates whether the two-timescale idea itself carries
    #     value once the (dead) code-gating is removed. Predict composite ↑; null retires it.
    #   plasticity_norm (cand 3): static LayerNorm on the shared encoder representation (Lyle
    #     2023), no code/lever. Predict composite ↑ / hit_80 ↑ via sustained cross-regime
    #     adaptability; also the A1 test — if it does nothing, plasticity loss probably isn't the
    #     bottleneck on this 2-regime task. Adds params (not baseline-interchangeable).
    #   surprise_spike (cand 4): change-point detector on the agent's own per-update TD-error
    #     surprise (value_loss) transiently spikes ent/intrinsic at DETECTED switches — faster
    #     than the Brain's decision_interval. Predict faster post-switch recovery (composite ↑);
    #     guard: over-exploration could depress within-regime hit_80.
    "brain_neuromod_critic_lr_lo": {"neuromod_critic_lr_scale": 0.5},
    "brain_neuromod_encoder_lr_lo": {"neuromod_encoder_lr_scale": 0.5},
    #   redo (cand 2): ReDo dormant-neuron reset every 50 updates (Sokar 2023); reset dormant
    #     actor/critic head units to restore plasticity. Predict composite ↑ via faster
    #     re-adaptation; registered mechanism probe = dormant-fraction (logged) must move, else
    #     it fails cleanly. Trigger = generic activation stat, not the code.
    "brain_neuromod_plasticity_norm": {"neuromod_plasticity_norm": True},
    "brain_neuromod_surprise_spike": {"neuromod_surprise_spike": 0.5},
    "brain_neuromod_redo": {"neuromod_redo_interval": 50},
    # Bottom-rung + oracle-rung controls (research note 0005) — the strategic-fork tests.
    #   constant_action (D0): Brain severed — zero action every decision = exact mid-bound
    #     HPs (the levels trained scout Brains empirically settle at, note 0005) with no
    #     sampling noise, Brain updates skipped. Directly measures what the Brain — an
    #     effectively untrained, noisy controller on this instrument (note 0005 A1 probe) —
    #     adds vs tuned static HPs.
    #   critic_lr_oracle (O1): ground-truth-timed critic damp (×0.5, detection update + 15
    #     following). Upper-bounds ANY Brain-learned critic damp — the adaptive A2 test the
    #     static cand-1 null could not answer. If this fails at n=8, A2 is dead.
    #   policy_swap (O2): per-regime learner snapshot/restore on revisit — the
    #     zero-forgetting CEILING of the instrument (diagnostic, never a method); sizes the
    #     headroom above the frozen control (reading ①).
    "brain_constant_action": {"constant_brain_action": True},
    "brain_critic_lr_oracle": {"neuromod_critic_lr_oracle": 0.5},
    "brain_oracle_policy_swap": {"neuromod_policy_swap": True},
}

# Metrics pulled from each scored run into the per-run table.
SCORE_METRICS = [
    "composite_score",
    "mean_post_switch_window_success_rate",
    "mean_episode_avg_success_rate",
    "median_steps_to_80",
    "median_steps_to_95",
    "hit_rate_80",
    "hit_rate_95",
    "mean_post_switch_policy_kl",
    "mean_post_switch_value_delta_abs",
    "mean_post_switch_neuromod_activity",
]


def _flag_args(flags: dict) -> list[str]:
    """Turn a {flag: value} dict into CLI tokens (store_true flags use bool values)."""
    args: list[str] = []
    for key, value in flags.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _script_path(script: str) -> Path:
    return REPO_ROOT / "scripts" / f"{script}.py"


def _find_new_run_dir(run_name: str, before: set[Path]) -> Path | None:
    after = {p.resolve() for p in DEFAULT_RUN_ROOT.glob(f"{run_name}_*") if p.is_dir()}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new:
        return None
    return new[-1]


def _existing_run_dir(run_name: str) -> Path | None:
    """Most-recent run dir for this run_name (a prior, possibly interrupted, attempt)."""
    dirs = sorted((p for p in DEFAULT_RUN_ROOT.glob(f"{run_name}_*") if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def _resume_target(run_dir: Path) -> tuple[str, Path | None]:
    """Classify a prior run dir for resume.

    Returns one of:
      ("complete", brain_model.pt)  -> training finished (brain_model.pt only exists at the end)
      ("resume",   episode_N/brain_epN.pt) -> latest periodic checkpoint to continue from
      ("empty",    None)            -> dir exists but no usable checkpoint (start fresh)
    """
    final = run_dir / "brain_model.pt"
    if final.exists():
        return ("complete", final)
    ckpts: list[tuple[int, Path]] = []
    for ep_dir in run_dir.glob("episode_*"):
        for ck in ep_dir.glob("brain_ep*.pt"):
            try:
                ckpts.append((int(ck.stem.replace("brain_ep", "")), ck))
            except ValueError:
                continue
    if ckpts:
        return ("resume", max(ckpts, key=lambda t: t[0])[1])
    return ("empty", None)


def _bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Mean and percentile bootstrap CI over a small set of seed values."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return (None, None, None, 0)
    arr = np.asarray(clean, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (mean, lo, hi, len(arr))


# Inner-run scalar prefix that marks a mechanism probe (e.g. the ReDo dormant-fraction —
# note 0004's registered probe — or the O1/O2 oracle activity flags).
PROBE_PREFIX = "brain_neuromod/"


def _extract_probes(run_dir: Path, out_dir: Path, run_name: str) -> str | None:
    """Copy every mechanism-probe series (brain_neuromod/* scalars) out of the ephemeral
    episode_*/ inner logs into the sweep dir, so probes land on origin/results alongside
    the scores instead of dying with the box disk. The LOOP-0007 redo gate was probe-blind
    because these series lived only in the inner logs (research note 0005) — this makes
    pre-registered probe discipline enforceable from the pushed artifacts."""
    probes: dict[str, dict] = {}
    for data_json in sorted(run_dir.glob("episode_*/**/*_data.json")):
        try:
            with open(data_json, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        series = {k: v for k, v in data.items() if k.startswith(PROBE_PREFIX)}
        if series:
            probes[str(data_json.parent.relative_to(run_dir))] = series
    if not probes:
        return None
    dest = out_dir / "probes" / f"{run_name}_probes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(probes, fh, separators=(",", ":"))
    return str(dest)


def run_cell(*, preset_name: str, preset: dict, condition: str, seed: int, device: str,
             out_dir: Path, dry_run: bool, resume: bool = False) -> dict:
    base = dict(preset["base"])
    overrides = CONDITIONS[condition]
    flags = {**base, **overrides, "seed": seed, "device": device}
    run_name = f"{preset_name}_{condition}_seed{seed}"
    flags["run_name"] = run_name

    record = {"preset": preset_name, "condition": condition, "seed": seed, "run_name": run_name}

    # --- resume bookkeeping -------------------------------------------------------------
    # With --resume: force periodic checkpoints (so an interrupted cell is recoverable) and,
    # if a prior attempt exists, either reuse it (complete) or continue it (partial).
    resume_action = "fresh"
    known_dir: Path | None = None
    if resume:
        if int(flags.get("save_every_episodes", 0) or 0) == 0:
            flags["save_every_episodes"] = RESUME_CHECKPOINT_EVERY
        existing = _existing_run_dir(run_name)
        if existing is not None:
            action, ckpt = _resume_target(existing)
            if action == "complete":
                resume_action, known_dir = "reused_complete", existing
            elif action == "resume":
                resume_action, known_dir = "resume", existing
                # explicit --brain_episodes (already in flags) keeps the target fixed; train_brain
                # reads the start episode from the checkpoint and continues in the same folder.
                flags = {**flags, "resume_path": str(ckpt)}
            # action == "empty" -> fall through to a fresh run

    cmd = [sys.executable, str(_script_path(preset["script"])), *_flag_args(flags)]
    record["resume_action"] = resume_action
    record["command"] = " ".join(cmd)

    if dry_run:
        record["status"] = "planned"
        return record

    if resume_action == "reused_complete":
        record["status_note"] = "already complete; scoring without re-running"
        run_dir = known_dir
    else:
        # For a fresh run, snapshot existing dirs to detect the new one; for resume, the dir is known.
        before = set() if resume_action == "resume" else {
            p.resolve() for p in DEFAULT_RUN_ROOT.glob(f"{run_name}_*") if p.is_dir()
        }
        log_path = out_dir / "logs" / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        import os
        # PYTHONUNBUFFERED so long-run per-cell logs stream live instead of block-buffering.
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONUNBUFFERED": "1"}

        started = time.perf_counter()
        log_mode = "a" if resume_action == "resume" else "w"
        with open(log_path, log_mode, encoding="utf-8") as handle:
            if resume_action == "resume":
                handle.write(f"\n===== RESUME {run_name} from {flags['resume_path']} =====\n")
                handle.flush()
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, text=True)
        record["duration_seconds"] = round(time.perf_counter() - started, 1)
        record["returncode"] = proc.returncode
        record["log_path"] = str(log_path)

        if proc.returncode != 0:
            record["status"] = "failed"
            return record

        run_dir = known_dir if resume_action == "resume" else _find_new_run_dir(run_name, before)
        if run_dir is None:
            record["status"] = "no_run_dir"
            return record
    record["run_dir"] = str(run_dir)

    # Probe extraction runs regardless of scoring outcome — a probe-blind gate is worse
    # than a score-blind one for probe-gated candidates.
    try:
        probes_path = _extract_probes(run_dir, out_dir, run_name)
        if probes_path:
            record["probes_path"] = probes_path
    except Exception as exc:
        record["probes_error"] = str(exc)

    try:
        score = score_brain_run(run_dir, **preset["score"])
        payload = score.to_dict()
        for metric in SCORE_METRICS:
            record[metric] = payload.get(metric)
        record["inner_run_count"] = payload.get("inner_run_count")
        record["switch_count"] = payload.get("switch_count")
        record["status"] = "scored"
    except Exception as exc:  # scoring can fail on too-short pilot traces; don't abort the sweep
        record["status"] = "score_error"
        record["error"] = str(exc)
    return record


def aggregate(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    for condition, cond_rows in by_condition.items():
        scored = [r for r in cond_rows if r.get("status") == "scored"]
        summary = {
            "condition": condition,
            "n_cells": len(cond_rows),
            "n_scored": len(scored),
            "seeds": sorted({r["seed"] for r in cond_rows}),
        }
        for metric in SCORE_METRICS:
            mean, lo, hi, n = _bootstrap_ci([r.get(metric) for r in scored])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci_lo"] = lo
            summary[f"{metric}_ci_hi"] = hi
            summary[f"{metric}_n"] = n
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="pilot")
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS),
                        default=["brain_neuromod", "brain_no_neuromod", "brain_random_code", "brain_oracle_code"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Number of cells to run concurrently (each cell is its own training "
                             "process). On a many-core box set to ~cores//8. Use 1 with async "
                             "presets to avoid oversubscription.")
    parser.add_argument("--out", default=None, help="Output dir (default sweeps/<preset>_<timestamp>)")
    parser.add_argument("--resume", action="store_true",
                        help="Resumable mode: force periodic checkpoints, and on re-launch continue "
                             "interrupted cells from their latest checkpoint (and skip-rescore "
                             "completed ones). Recommended for cloud/spot instances.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned cells and exit")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "sweeps" / f"{args.preset}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = [(c, s) for c in args.conditions for s in args.seeds]
    max_parallel = max(1, args.max_parallel)
    print(f"[sweep] preset={args.preset} conditions={args.conditions} seeds={args.seeds}")
    print(f"[sweep] {len(cells)} cells, max_parallel={max_parallel} -> {out_dir}"
          f"{' (DRY RUN)' if args.dry_run else ''}")

    rows: list[dict] = []
    if args.dry_run:
        for condition, seed in cells:
            rows.append(run_cell(preset_name=args.preset, preset=preset, condition=condition,
                                 seed=seed, device=args.device, out_dir=out_dir, dry_run=True,
                                 resume=args.resume))
        for row in rows:
            tag = "" if row.get("resume_action", "fresh") == "fresh" else f" [{row['resume_action']}]"
            print(f"  {row['run_name']}{tag}: {row['command']}")
        return

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_cell, preset_name=args.preset, preset=preset, condition=condition,
                        seed=seed, device=args.device, out_dir=out_dir, dry_run=False,
                        resume=args.resume): (condition, seed)
            for condition, seed in cells
        }
        for fut in concurrent.futures.as_completed(futures):
            condition, seed = futures[fut]
            row = fut.result()
            done += 1
            status = row.get("status")
            action = row.get("resume_action", "fresh")
            tag = "" if action == "fresh" else f" ({action})"
            extra = f" composite={row.get('composite_score')}" if status == "scored" else ""
            print(f"[sweep] ({done}/{len(cells)}) {condition} seed={seed}{tag} -> {status}{extra}", flush=True)
            rows.append(row)

    run_columns = ["preset", "condition", "seed", "run_name", "status", "returncode",
                   "duration_seconds", *SCORE_METRICS, "inner_run_count", "switch_count",
                   "run_dir", "probes_path", "log_path", "error", "command"]
    _write_csv(out_dir / "runs.csv", rows, run_columns)
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summaries = aggregate(rows)
    summary_columns = ["condition", "n_cells", "n_scored", "seeds"] + [
        f"{m}_{suffix}" for m in SCORE_METRICS for suffix in ("mean", "ci_lo", "ci_hi", "n")
    ]
    _write_csv(out_dir / "summary.csv", summaries, summary_columns)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"\n[sweep] done. per-run -> {out_dir/'runs.csv'} | summary -> {out_dir/'summary.csv'}")
    print("\n[sweep] composite_score (mean [95% CI], n):")
    for s in summaries:
        m, lo, hi, n = s["composite_score_mean"], s["composite_score_ci_lo"], s["composite_score_ci_hi"], s["composite_score_n"]
        if m is None:
            print(f"  {s['condition']:<20} no scored runs")
        else:
            print(f"  {s['condition']:<20} {m:.4f} [{lo:.4f}, {hi:.4f}]  (n={n})")


if __name__ == "__main__":
    main()
