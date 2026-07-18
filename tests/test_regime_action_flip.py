"""
Tests for LOOP-0018 / IP-1 — the action_flip regime mode (research-log 0016; note 0015).

Contract of RegimeGoalSwapWrapper(regime_effect="action_flip"):
  - Odd regimes swap actions 0<->1 before they reach the wrapped env; action 2 and even
    regimes pass through unchanged.
  - The reward mapping is FIXED at regime 0's (hit_goal_index==0 is good in EVERY regime)
    — regimes differ in dynamics only.
  - Regime clocking and info["regime_id"] are unchanged.
  - Default mode ("goal_swap") stays byte-identical to the frozen benchmark's behavior.
"""
import gymnasium as gym
import numpy as np
import pytest

from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper


class _RecorderEnv(gym.Env):
    """Discrete(3) stub that records the actions it receives and terminates on demand."""

    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.received: list[int] = []
        self.next_hit_goal: int | None = None  # set to terminate with that goal index

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.received.append(int(action))
        if self.next_hit_goal is not None:
            info = {"hit_goal_index": self.next_hit_goal}
            self.next_hit_goal = None
            return np.zeros(1, dtype=np.float32), 1.0, True, False, info
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _wrap(mode: str, steps_per_regime: int = 4):
    inner = _RecorderEnv()
    return inner, RegimeGoalSwapWrapper(
        inner, steps_per_regime=steps_per_regime, num_regimes=2, regime_effect=mode)


def test_invalid_mode_rejected_at_init():
    with pytest.raises(ValueError):
        _wrap("reward_flip")


def test_action_flip_swaps_turns_in_odd_regime_only():
    inner, env = _wrap("action_flip", steps_per_regime=3)
    env.reset()
    for a in (0, 1, 2):          # regime 0 after these steps' clocking: steps 1,2,3
        env.step(a)
    # step counter is now 3 -> regime 1 for the next steps
    for a in (0, 1, 2):
        env.step(a)
    assert inner.received[:3] == [0, 1, 2]       # regime 0: pass-through
    assert inner.received[3:] == [1, 0, 2]       # regime 1: 0<->1 swapped, 2 unchanged


def test_action_flip_reward_mapping_is_fixed_at_goal_zero():
    inner, env = _wrap("action_flip", steps_per_regime=100)
    env.reset()
    # Regime 0: goal 0 good, goal 1 bad.
    inner.next_hit_goal = 0
    _, r, _, _, info = env.step(2)
    assert r > 0 and info["reached_good_goal"] == 1.0
    inner.next_hit_goal = 1
    _, r, _, _, info = env.step(2)
    assert r < 0 and info["reached_bad_goal"] == 1.0

    # Force regime 1: goal 0 must STILL be the good goal (dynamics-only regimes).
    env.cumulative_steps = 150
    inner.next_hit_goal = 0
    _, r, _, _, info = env.step(2)
    assert info["regime_id"] == 1
    assert r > 0 and info["reached_good_goal"] == 1.0
    inner.next_hit_goal = 1
    _, r, _, _, info = env.step(2)
    assert r < 0 and info["reached_bad_goal"] == 1.0


def test_goal_swap_default_unchanged():
    inner, env = _wrap("goal_swap", steps_per_regime=100)
    env.reset()
    # Regime 0: goal 0 good.
    inner.next_hit_goal = 0
    _, r, _, _, info = env.step(1)
    assert r > 0 and info["reached_good_goal"] == 1.0
    # Regime 1: goal 1 good (the frozen benchmark's swap), actions pass through.
    env.cumulative_steps = 150
    inner.next_hit_goal = 1
    _, r, _, _, info = env.step(1)
    assert info["regime_id"] == 1
    assert r > 0 and info["reached_good_goal"] == 1.0
    assert inner.received == [1, 1]              # never permuted in goal_swap mode
