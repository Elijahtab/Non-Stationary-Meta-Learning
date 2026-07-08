"""
Regression tests for LOOP-0007 candidate 4 — surprise-triggered exploration spike
(research note 0004). A generic change-point detector on the inner agent's own
per-update TD-error surprise (value_loss) arms a transient ent/intrinsic boost at
detected regime switches — never reading the Brain code.

We test the pure detector (`_update_surprise_spike`) that carries the mechanism, plus
its off-path no-op. The transient-boost application in run_inner_update is a plain
scale-and-restore around the existing update body (byte-identical when off).
"""
from types import SimpleNamespace

from lifelong_learning.agents.ppo.train import (
    _update_surprise_spike,
    SURPRISE_SPIKE_UPDATES,
    SURPRISE_EMA_DECAY,
)


def _state(threshold=0.5, ema=None, remaining=0):
    return SimpleNamespace(
        surprise_spike_threshold=threshold,
        surprise_value_ema=ema,
        surprise_spike_remaining=remaining,
    )


def test_off_is_noop():
    s = _state(threshold=0.0, ema=None)
    _update_surprise_spike(s, 5.0)
    assert s.surprise_value_ema is None
    assert s.surprise_spike_remaining == 0


def test_first_observation_only_seeds_baseline():
    s = _state(ema=None)
    _update_surprise_spike(s, 3.0)
    assert s.surprise_value_ema == 3.0
    assert s.surprise_spike_remaining == 0  # never triggers on the first sample


def test_jump_above_threshold_arms_spike():
    s = _state(threshold=0.5, ema=2.0)
    # 3.1 > 2.0 * 1.5 == 3.0 -> triggers
    _update_surprise_spike(s, 3.1)
    assert s.surprise_spike_remaining == SURPRISE_SPIKE_UPDATES
    # EMA moves toward the new value
    assert abs(s.surprise_value_ema - (SURPRISE_EMA_DECAY * 2.0 + 0.1 * 3.1)) < 1e-9


def test_small_rise_does_not_trigger():
    s = _state(threshold=0.5, ema=2.0)
    # 2.9 < 3.0 threshold -> no trigger
    _update_surprise_spike(s, 2.9)
    assert s.surprise_spike_remaining == 0
    assert s.surprise_value_ema != 2.0  # baseline still updates


def test_stable_surprise_never_triggers():
    s = _state(threshold=0.5, ema=None)
    for _ in range(20):
        _update_surprise_spike(s, 1.0)
    assert s.surprise_spike_remaining == 0
    assert abs(s.surprise_value_ema - 1.0) < 1e-6


def test_ema_tracks_gradual_drift_without_triggering():
    # A slow upward drift stays under the relative threshold -> no false positives.
    s = _state(threshold=0.5, ema=None)
    v = 1.0
    triggered = False
    for _ in range(30):
        _update_surprise_spike(s, v)
        if s.surprise_spike_remaining > 0:
            triggered = True
        v *= 1.05  # +5% per step, well under the +50% trigger
    assert not triggered
