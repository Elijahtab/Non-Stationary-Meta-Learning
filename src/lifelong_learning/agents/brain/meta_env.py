"""
MetaEnv: Gymnasium environment wrapping the inner Dyna-PPO training loop.

The Brain meta-agent interacts with this environment. Each "step" runs
N inner PPO updates and returns training signals as observations.
"""
from __future__ import annotations

import copy
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    InnerTrainState,
    init_inner_training,
    run_inner_update,
    close_inner_training,
    apply_inner_lr,
)
from lifelong_learning.agents.brain.neuromod import (
    BRAIN_ACTION_DIM,
    BRAIN_CONTEXT_SLICE,
    CONTEXT_CODE_DIM,
    log_neuromodulation_snapshot,
)

CONTEXT_CODE_SOURCES = ("brain", "random", "oracle", "zero")
from lifelong_learning.agents.brain.signals import SignalExtractor, NUM_SIGNALS


class MetaEnv(gym.Env):
    """
    Gymnasium environment where:
      - Observation: 19-dim vector of normalized training signals
      - Action: continuous vector controlling hyperparameter adjustments
        [0]  lr scale             in [-1, 1] -> mapped linearly to [lr_min, lr_max]
        [1]  ent_coef scale       in [-1, 1] -> mapped linearly to [ent_min, ent_max]
        [2]  intrinsic_coef       in [-1, 1] -> mapped linearly to [intr_min, intr_max]
        [3]  imagined_horizon     in [-1, 1] -> mapped linearly to [1, 30]
        [4]  replay_ratio         in [-1, 1] -> mapped linearly to [0.0, 0.5]
        [5]  replay_prioritization in [-1, 1] -> mapped linearly to [0.0, 1.0]
        [6]  anchoring_weight     in [-1, 1] -> mapped linearly to [0.0, 0.5]
        [7:] context_code         in [-1, 1]^8 -> neuromodulation mask via ContextDecoder
      - Reward: recovery-based metric (delta success_rate + alpha * delta return - beta * failure_rate)      - Episode: one full inner training run
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: str = "MiniGrid-MultiGoal-8x8-v0",
        inner_cfg: PPOConfig | None = None,
        decision_interval: int = 10,
        steps_per_regime: int | None = 15000,
        episodes_per_regime: int | None = None,
        start_regime: int = 0,
        randomize_start_regime: bool = False,
        num_regimes: int = 2,
        reward_alpha: float = 0.1,
        reward_beta: float = 0.5,
        reward_mode: str = "auc",
        inner_run_name: str | None = None,
        save_checkpoints: bool = False,
        anneal_lr: bool = False,
        intrinsic_coef: float = 0.015,
        intrinsic_reward_clip: float = 0.1,
        imagined_horizon: int = 10,
        wm_lr: float = 1e-4,
        inner_log_dir: str | None = None,
        env_index: int = 0,
        episodic_memory_capacity: int = 50000,
        max_inner_lr: float = 0.003,
        min_inner_lr: float = 1e-4,
        min_ent_coef: float = 0.001,
        max_ent_coef: float = 0.1,
        min_intrinsic_coef: float = 0.001,
        max_intrinsic_coef: float = 0.5,
        start_episode: int = 1,
        disable_neuromodulation: bool = False,
        context_code_source: str = "brain",
        trainable_neuromod: bool = False,
        neuromod_decoder_lr: float | None = None,
        actor_only_neuromod: bool = False,
        neuromod_gain_alpha: float = 0.0,
        grad_gate_neuromod: bool = False,
        critic_code_neuromod: bool = False,
        aux_code_coef: float = 0.0,
        neuromod_adam_flush_threshold: float = 0.0,
        critic_lr_scale: float = 1.0,
        encoder_lr_scale: float = 1.0,
        plasticity_norm: bool = False,
        runtime_cpu_threads: int | None = None,
    ):
        super().__init__()

        if context_code_source not in CONTEXT_CODE_SOURCES:
            raise ValueError(
                f"context_code_source must be one of {CONTEXT_CODE_SOURCES}, got {context_code_source!r}"
            )

        self.start_episode = start_episode
        self.env_index = env_index
        self.env_id = env_id
        self.inner_cfg = copy.deepcopy(inner_cfg or PPOConfig())
        self.decision_interval = decision_interval
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.start_regime = start_regime
        self.randomize_start_regime = randomize_start_regime
        self.num_regimes = num_regimes
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_mode = reward_mode
        self.inner_run_name = inner_run_name
        self.save_checkpoints = save_checkpoints
        self.anneal_lr = anneal_lr
        self.intrinsic_coef_init = intrinsic_coef
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.imagined_horizon_init = imagined_horizon
        self.wm_lr = wm_lr
        self.inner_log_dir = inner_log_dir
        self.episodic_memory_capacity = episodic_memory_capacity
        self.max_inner_lr = max_inner_lr
        self.min_inner_lr = min_inner_lr
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.min_intrinsic_coef = min_intrinsic_coef
        self.max_intrinsic_coef = max_intrinsic_coef
        self.disable_neuromodulation = disable_neuromodulation
        self.context_code_source = context_code_source
        self.trainable_neuromod = trainable_neuromod
        self.neuromod_decoder_lr = neuromod_decoder_lr
        self.actor_only_neuromod = actor_only_neuromod
        self.neuromod_gain_alpha = neuromod_gain_alpha
        self.grad_gate_neuromod = grad_gate_neuromod
        self.critic_code_neuromod = critic_code_neuromod
        self.aux_code_coef = aux_code_coef
        # LOOP-0006 hypothesis 4: when > 0, reset the inner Adam optimizer state whenever
        # the context-code strength (‖code‖/√dim, clamped to [0,1]) jumps by at least this
        # much between Brain decisions — a Brain-directed "flush stale curvature" signal.
        self.neuromod_adam_flush_threshold = neuromod_adam_flush_threshold
        # LOOP-0007 cands 1/5 (research note 0004): critic head / shared encoder in
        # their own optimizer group at LR = main_lr * scale — cand 1 damps critic
        # whiplash, cand 5 makes the encoder learn slower than the heads (stable
        # features, fast readout re-map). 1.0 == off (baseline-identical).
        self.critic_lr_scale = critic_lr_scale
        self.encoder_lr_scale = encoder_lr_scale
        # LOOP-0007 cand 3: static LayerNorm on the shared encoder representation
        # (plasticity preservation, Lyle 2023). No code, no lever; also the A1 test.
        self.plasticity_norm = plasticity_norm
        self._prev_context_strength: float | None = None
        self._random_context_code = None
        self.runtime_cpu_threads = runtime_cpu_threads

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(NUM_SIGNALS,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(BRAIN_ACTION_DIM,), dtype=np.float32
        )

        # HP bounds (absolute min/max)
        self.lr_bounds = (self.min_inner_lr, self.max_inner_lr)
        self.ent_coef_bounds = (self.min_ent_coef, self.max_ent_coef)
        self.intrinsic_coef_bounds = (self.min_intrinsic_coef, self.max_intrinsic_coef)
        self.imagined_horizon_bounds = (1, 30)
        self.replay_ratio_bounds = (0.0, 0.5)
        self.replay_prioritization_bounds = (0.0, 1.0)
        self.anchoring_weight_bounds = (0.0, 0.5)

        # Will be initialized on reset()
        self._state: InnerTrainState | None = None
        self._signal_extractor: SignalExtractor | None = None
        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0
        self._prev_failure_rate = 0.0
        self._episode_counter = self.start_episode - 1
        self._episode_prefix = "episode"
        self._regime_exposures = {}
        self._current_regime = None
        self._pending_resume_state = None

        # Switch tracking for reward_mode="recovery_v2": count Brain steps since the last
        # regime switch so the recovery bonus can be gated to a post-switch window. The window
        # mirrors the scorer's 0.5*steps_per_regime, expressed in Brain steps.
        self._steps_since_switch = None
        self._last_switch_regime = None
        inner_steps_per_brain_step = max(
            1, self.inner_cfg.num_envs * self.inner_cfg.num_steps * self.decision_interval
        )
        if self.steps_per_regime:
            brain_steps_per_regime = self.steps_per_regime / inner_steps_per_brain_step
            self._post_switch_window_steps = max(1, int(round(0.5 * brain_steps_per_regime)))
        else:
            self._post_switch_window_steps = 10

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        pending_resume_state = self._pending_resume_state or {}
        np_random_state = pending_resume_state.get("np_random_state")
        if np_random_state is not None and getattr(self, "_np_random", None) is not None:
            self.np_random.bit_generator.state = copy.deepcopy(np_random_state)

        if self.randomize_start_regime:
            start_regime = int(self.np_random.integers(0, self.num_regimes))
        else:
            start_regime = self.start_regime

        # For the "random" code ablation, draw one fixed, regime-agnostic code per episode.
        # It controls for code *content*: the Brain still sets levers, but its learned code
        # is replaced by uninformative noise of comparable magnitude.
        if self.context_code_source == "random":
            self._random_context_code = self.np_random.uniform(
                -1.0, 1.0, size=CONTEXT_CODE_DIM
            ).astype(np.float32)

        # Clean up any previous inner training
        if self._state is not None:
            close_inner_training(self._state)

        # Adam-flush spike detector starts fresh each episode (new optimizer, new inner run).
        self._prev_context_strength = None

        self._episode_counter += 1
        run_name = self.inner_run_name or f"ep{self._episode_counter}_env{self.env_index}"

        init_kwargs = dict(
            env_id=self.env_id,
            cfg=self.inner_cfg,
            steps_per_regime=self.steps_per_regime,
            episodes_per_regime=self.episodes_per_regime,
            start_regime=start_regime,
            num_regimes=self.num_regimes,
            run_name=run_name,
            save_every_updates=9999 if not self.save_checkpoints else 50,
            anneal_lr=self.anneal_lr,
            intrinsic_coef=self.intrinsic_coef_init,
            intrinsic_reward_clip=self.intrinsic_reward_clip,
            imagined_horizon=self.imagined_horizon_init,
            wm_lr=self.wm_lr,
            episodic_memory_capacity=self.episodic_memory_capacity,
            cpu_threads=self.runtime_cpu_threads,
            trainable_neuromod=self.trainable_neuromod,
            neuromod_decoder_lr=self.neuromod_decoder_lr,
            actor_only_neuromod=self.actor_only_neuromod,
            neuromod_gain_alpha=self.neuromod_gain_alpha,
            grad_gate_neuromod=self.grad_gate_neuromod,
            critic_code_neuromod=self.critic_code_neuromod,
            aux_code_coef=self.aux_code_coef,
            critic_lr_scale=self.critic_lr_scale,
            encoder_lr_scale=self.encoder_lr_scale,
            plasticity_norm=self.plasticity_norm,
        )
        if self.inner_log_dir is not None:
            ep_log_dir = os.path.join(self.inner_log_dir, f"{self._episode_prefix}_{self._episode_counter}")
            init_kwargs["log_dir"] = ep_log_dir
            init_kwargs["save_dir"] = os.path.join(ep_log_dir, "inner_checkpoints")
            
        self._state = init_inner_training(**init_kwargs)

        # Reuse SignalExtractor across episodes to preserve normalizer stats;
        # only create it on the very first reset.
        if self._signal_extractor is None:
            self._signal_extractor = SignalExtractor(
                max_inner_lr=self.max_inner_lr,
                min_inner_lr=self.min_inner_lr,
                min_ent_coef=self.min_ent_coef,
                max_ent_coef=self.max_ent_coef,
                min_intrinsic_coef=self.min_intrinsic_coef,
                max_intrinsic_coef=self.max_intrinsic_coef,
            )
        else:
            self._signal_extractor.reset()

        signal_extractor_state = pending_resume_state.get("signal_extractor")
        if signal_extractor_state is not None:
            self._signal_extractor.load_state_dict(signal_extractor_state)
        self._pending_resume_state = None

        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0
        self._prev_failure_rate = 0.0
        self._regime_exposures = {}
        self._current_regime = None
        self._steps_since_switch = None
        self._last_switch_regime = None

        # Run initial updates to get a meaningful first observation
        stats = self._run_n_updates(self.decision_interval)
        obs = self._signal_extractor.extract(stats)
        info = {"inner_stats": stats}

        return obs, info

    def step(self, action: np.ndarray):
        assert self._state is not None, "Must call reset() first"

        # Apply hyperparameter adjustments
        self._apply_action(action)

        # Run N inner updates
        stats = self._run_n_updates(self.decision_interval)

        # Compute observation
        obs = self._signal_extractor.extract(stats)

        # Compute reward
        success_rate = stats.get("success_rate", 0.0)
        mean_return = stats.get("mean_episodic_return", 0.0)
        failure_rate = stats.get("failure_rate", 0.0)

        # Track Brain steps since the last regime switch (used by recovery_v2).
        regime_now = stats.get("current_mode_regime", 0)
        if self._last_switch_regime is None:
            self._last_switch_regime = regime_now
        elif regime_now != self._last_switch_regime:
            self._last_switch_regime = regime_now
            self._steps_since_switch = 0
        elif self._steps_since_switch is not None:
            self._steps_since_switch += 1

        if self.reward_mode == "recovery_v2":
            # Fixes the oscillation-farming pathology of "recovery" (see docs/research-log/0003):
            # the progress term is SYMMETRIC (a potential-based delta that telescopes to net
            # change), so up/down noise nets to zero and within-regime forgetting is penalized
            # rather than free. Fast recovery is incentivized by up-weighting that same symmetric
            # progress inside a post-switch window — not by an asymmetric bonus that rewards any
            # upswing. A maintenance (level) term anchors the optimum at "high and stable", and a
            # failure penalty is retained.
            delta_sr = success_rate - self._prev_success_rate
            in_post_switch = (
                self._steps_since_switch is not None
                and self._steps_since_switch <= self._post_switch_window_steps
            )
            progress_weight = 5.0 if in_post_switch else 1.5
            progress = delta_sr * progress_weight        # symmetric: drops cost as much as gains
            maintenance = success_rate * 0.5
            failure_penalty = -failure_rate * 0.5
            reward = progress + maintenance + failure_penalty
        elif self.reward_mode == "recovery":
            # Hybrid recovery reward: incentivizes fast recovery after regime switches
            delta_sr = success_rate - self._prev_success_rate
            recovery = max(0.0, delta_sr) * 5.0
            maintenance = success_rate * 0.3
            shortfall = max(0.0, 0.8 - success_rate)
            urgency_penalty = -shortfall * 3.0
            failure_penalty = -failure_rate * 0.5
            reward = recovery + maintenance + urgency_penalty + failure_penalty
        elif self.reward_mode == "curriculum":
            # Tracks exposure to regimes. Re-visiting an old regime grants a massive multiplier.
            current_regime = stats.get("current_mode_regime", 0)
            if self._current_regime != current_regime:
                self._regime_exposures[current_regime] = self._regime_exposures.get(current_regime, 0) + 1
                self._current_regime = current_regime
            
            exposures = self._regime_exposures.get(current_regime, 1)
            # Give a 10x multiplier if this is the 2nd+ time we've seen this regime
            multiplier = 10.0 if exposures > 1 else 1.0
            
            base_reward = success_rate + self.reward_alpha * mean_return - self.reward_beta * failure_rate
            reward = base_reward * multiplier
        else:
            # Original AUC reward
            reward = success_rate + self.reward_alpha * mean_return - self.reward_beta * failure_rate

        self._prev_success_rate = success_rate
        self._prev_mean_return = mean_return
        self._prev_failure_rate = failure_rate

        terminated = stats.get("done", False)
        truncated = False
        info = {"inner_stats": stats}

        return obs, float(reward), terminated, truncated, info

    def _run_n_updates(self, n: int) -> dict:
        """Run n inner PPO updates and summarize the full decision interval."""
        interval_stats = []
        for _ in range(n):
            stats = run_inner_update(self._state)
            interval_stats.append(stats)
            if stats.get("done", False):
                break
        return self._summarize_interval_stats(interval_stats)

    def _summarize_interval_stats(self, interval_stats: list[dict]) -> dict:
        """Collapse multiple inner updates into one Brain-facing meta-step summary."""
        if not interval_stats:
            return {}

        last = dict(interval_stats[-1])
        mean_keys = [
            "mean_surprise",
            "wm_loss_state",
            "wm_loss_reward",
            "policy_entropy",
            "policy_loss",
            "value_loss",
        ]
        for key in mean_keys:
            last[key] = float(np.mean([stats.get(key, 0.0) for stats in interval_stats]))

        completed_episodes = int(sum(stats.get("completed_episodes", 0) for stats in interval_stats))
        successes = int(sum(stats.get("successful_episodes", 0) for stats in interval_stats))
        failures = int(sum(stats.get("failed_episodes", 0) for stats in interval_stats))
        timeouts = int(sum(stats.get("timeout_episodes", 0) for stats in interval_stats))
        total_return = float(sum(stats.get("episodic_return_sum", 0.0) for stats in interval_stats))

        last["overall_success_rate"] = float(last.get("success_rate", 0.0))
        last["overall_failure_rate"] = float(last.get("failure_rate", 0.0))

        if completed_episodes > 0:
            last["mean_episodic_return"] = total_return / completed_episodes
            last["success_rate"] = successes / completed_episodes
            last["failure_rate"] = failures / completed_episodes
            last["timeout_rate"] = timeouts / completed_episodes
        else:
            last["success_rate"] = float(last.get("recent_success_rate", last.get("success_rate", 0.0)))
            last["failure_rate"] = float(last.get("recent_failure_rate", last.get("failure_rate", 0.0)))
            last["timeout_rate"] = float(last.get("recent_timeout_rate", 1.0 - last["success_rate"] - last["failure_rate"]))
            last["mean_episodic_return"] = float(np.mean([stats.get("mean_episodic_return", 0.0) for stats in interval_stats]))

        last["interval_completed_episodes"] = completed_episodes
        last["interval_successes"] = successes
        last["interval_failures"] = failures
        last["interval_timeouts"] = timeouts
        last["interval_updates"] = len(interval_stats)
        return last

    def _apply_action(self, action: np.ndarray):
        """Map Brain action [-1, 1]^15 to absolute HP values and apply to inner state."""
        s = self._state

        # Helper to map [-1, 1] to [min_val, max_val]
        def map_to_range(a: float, bounds: tuple[float, float]) -> float:
            return float(bounds[0] + (a + 1.0) / 2.0 * (bounds[1] - bounds[0]))

        # Action[0]: lr scale
        new_lr = map_to_range(action[0], self.lr_bounds)
        apply_inner_lr(s.optimizer, new_lr, s.cfg)

        # Action[1]: ent_coef
        s.cfg.ent_coef = map_to_range(action[1], self.ent_coef_bounds)

        # Action[2]: intrinsic_coef
        s.intrinsic_coef = map_to_range(action[2], self.intrinsic_coef_bounds)

        # Action[3]: imagined_horizon (integer mapping)
        horizon_float = map_to_range(action[3], self.imagined_horizon_bounds)
        s.imagined_horizon = int(np.clip(round(horizon_float), *self.imagined_horizon_bounds))

        # Action[4]: replay_ratio
        s.replay_ratio = map_to_range(action[4], self.replay_ratio_bounds)

        # Action[5]: replay_prioritization
        s.replay_prioritization = map_to_range(action[5], self.replay_prioritization_bounds)

        # Action[6]: anchoring_weight
        s.cfg.anchoring_weight = map_to_range(action[6], self.anchoring_weight_bounds)

        # Action[7:]: neuromodulation context code (source selectable for ablations)
        if not self.disable_neuromodulation and len(action) >= BRAIN_CONTEXT_SLICE.stop:
            import torch
            code = self._resolve_context_code(action)
            context_code = torch.tensor(code, dtype=torch.float32, device=s.device)
            s.model.set_context_code(context_code)
            self._maybe_flush_optimizer_state(code)
            self._log_neuromodulation_snapshot(context_code)

    def _maybe_flush_optimizer_state(self, code: np.ndarray) -> None:
        """LOOP-0006 hypothesis 4: reset stale Adam moments on a context-strength spike.

        Strength = ‖code‖/√dim clamped to [0,1] (same scalar that scales the decoded
        mask). A jump ≥ threshold between consecutive Brain decisions clears the inner
        optimizer's per-param state (exp_avg, exp_avg_sq, step) — Adam re-estimates
        curvature from post-spike gradients instead of mis-scaling updates with
        pre-switch second moments. Default threshold 0.0 = disabled (baseline).
        """
        if self.neuromod_adam_flush_threshold <= 0.0:
            return
        strength = min(1.0, float(np.linalg.norm(code)) / float(np.sqrt(CONTEXT_CODE_DIM)))
        prev = self._prev_context_strength
        self._prev_context_strength = strength
        if prev is None or abs(strength - prev) < self.neuromod_adam_flush_threshold:
            return
        s = self._state
        s.optimizer.state.clear()
        if s.logger is not None:
            s.logger.scalar("brain_neuromod/adam_flush_event", 1.0, s.global_step)

    def _resolve_context_code(self, action: np.ndarray) -> np.ndarray:
        """Select the neuromodulation context code per ``context_code_source``.

        Scalar HP levers (action[0:7]) are always applied by the caller regardless of
        source; only the 8-d code feeding the neuromodulator is swapped, which isolates
        the contribution of the *code* from the contribution of the levers.

        - ``brain``  : the Brain's emitted code (default; current behavior)
        - ``zero``   : all-zeros → identity mask (modulation off, code path still logged)
        - ``random`` : a per-episode fixed random code (controls for code *content*)
        - ``oracle`` : a fixed one-hot of the otherwise-hidden current regime (upper bound)
        """
        source = self.context_code_source
        if source == "zero":
            return np.zeros(CONTEXT_CODE_DIM, dtype=np.float32)
        if source == "random":
            if self._random_context_code is None:
                self._random_context_code = np.zeros(CONTEXT_CODE_DIM, dtype=np.float32)
            return self._random_context_code
        if source == "oracle":
            code = np.zeros(CONTEXT_CODE_DIM, dtype=np.float32)
            regime = int(getattr(self._state, "current_mode_regime", 0) or 0)
            code[regime % CONTEXT_CODE_DIM] = 1.0
            return code
        # default: "brain"
        return np.asarray(action[BRAIN_CONTEXT_SLICE], dtype=np.float32)

    def _log_neuromodulation_snapshot(self, context_code):
        """Record the Brain context code, decoded mask, and its effect on the current batch."""
        s = self._state
        if s is None:
            return

        log_neuromodulation_snapshot(
            logger=getattr(s, "logger", None),
            model=s.model,
            obs_t=getattr(s, "obs_t", None),
            context_code=context_code,
            step=int(getattr(s, "global_step", 0)),
        )

    def get_resume_state(self) -> dict:
        """Return per-env state needed to smooth Brain checkpoint resumes."""
        signal_extractor_state = None
        if self._signal_extractor is not None:
            signal_extractor_state = self._signal_extractor.state_dict()

        np_random_state = None
        if getattr(self, "_np_random", None) is not None:
            np_random_state = copy.deepcopy(self.np_random.bit_generator.state)

        return {
            "env_index": int(self.env_index),
            "signal_extractor": signal_extractor_state,
            "np_random_state": np_random_state,
        }

    def load_resume_state(self, resume_state):
        """Queue a per-env resume payload to apply on the next reset."""
        if isinstance(resume_state, (list, tuple)):
            if 0 <= self.env_index < len(resume_state):
                resume_state = resume_state[self.env_index]
            else:
                resume_state = None
        self._pending_resume_state = copy.deepcopy(resume_state)

    def close(self):
        if self._state is not None:
            close_inner_training(self._state)
            self._state = None
        super().close()
