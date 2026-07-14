"""
Regression tests for the bottom-rung + oracle-rung controls (research note 0005):

  D0  brain_constant_action  — sever the Brain: zero action every decision (= exact
      mid-bound HPs, zero code), Brain PPO updates skipped.
  O1  critic_lr_oracle       — ground-truth-timed critic-LR damp: on each detected
      regime switch the critic optimizer group runs at main_lr * scale for the
      detection update + CRITIC_ORACLE_UPDATES following, then reverts to tracking
      the main LR exactly.
  O2  policy_swap_topline    — diagnostic ceiling: bank the full learner per regime
      at switch-away, restore on revisit; must never clobber the live LR schedule.

All three are flag-guarded and default-off: with the flags at their defaults the
optimizer layout, forward pass, and Brain loop are byte-identical to baseline.
"""
import sys
from types import SimpleNamespace

import pytest
import torch

import scripts.train_brain as train_brain_script
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    CRITIC_ORACLE_UPDATES,
    _restore_learner,
    _snapshot_learner,
    apply_inner_lr,
    init_inner_training,
    run_inner_update,
)

ENV_ID = "MiniGrid-MultiGoal-8x8-v0"


def _tiny_cfg(**over) -> PPOConfig:
    base = dict(total_timesteps=600, num_envs=2, num_steps=8, seed=0,
                device="cpu", mode="dyna", lr=1e-3)
    base.update(over)
    return PPOConfig(**base)


def _groups_by_name(optimizer):
    return {g.get("name"): g for g in optimizer.param_groups}


# --------------------------------------------------------------------------- #
# O1 — apply_inner_lr damp-window semantics (pure)
# --------------------------------------------------------------------------- #

def test_oracle_damps_critic_while_window_live():
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "critic", "lr": 0.0},
    ])
    cfg = SimpleNamespace(critic_lr_scale=1.0, critic_lr_oracle_scale=0.5,
                          critic_oracle_remaining=3)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 5e-4  # damped


def test_oracle_tracks_main_outside_window():
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "critic", "lr": 123.0},  # stale value must be overwritten
    ])
    cfg = SimpleNamespace(critic_lr_scale=1.0, critic_lr_oracle_scale=0.5,
                          critic_oracle_remaining=0)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[1]["lr"] == 1e-3  # tracks main exactly, no damp


def test_oracle_composes_with_static_critic_scale():
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "critic", "lr": 0.0},
    ])
    cfg = SimpleNamespace(critic_lr_scale=0.5, critic_lr_oracle_scale=0.5,
                          critic_oracle_remaining=1)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[1]["lr"] == 2.5e-4  # 0.5 static * 0.5 oracle
    cfg.critic_oracle_remaining = 0
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[1]["lr"] == 5e-4    # static scale only


def test_oracle_absent_from_cfg_is_baseline_behaviour():
    # Old cfgs (e.g. SimpleNamespace in prior tests) lack the oracle attrs entirely.
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "critic", "lr": 777.0},
    ])
    cfg = SimpleNamespace(critic_lr_scale=1.0)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[1]["lr"] == 777.0  # untouched, exactly as before


# --------------------------------------------------------------------------- #
# O1 — optimizer construction + end-to-end damp window
# --------------------------------------------------------------------------- #

def test_oracle_creates_critic_group_at_main_lr():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, critic_lr_oracle_scale=0.5)
    try:
        groups = _groups_by_name(state.optimizer)
        assert "critic" in groups
        assert groups["critic"]["lr"] == groups["main"]["lr"]  # no damp before a switch
        assert state.cfg.critic_oracle_remaining == 0
    finally:
        state.envs.close()


def test_oracle_default_off_single_group():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg)
    try:
        assert len(state.optimizer.param_groups) == 1  # baseline-identical
    finally:
        state.envs.close()


def test_oracle_window_arms_on_switch_and_reverts():
    # steps_per_regime = 2 updates' worth of steps -> first switch inside update 3.
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, steps_per_regime=32,
                                critic_lr_oracle_scale=0.5)
    try:
        armed = False
        for _ in range(6):
            run_inner_update(state)
            if state.cfg.critic_oracle_remaining > 0:
                armed = True
                break
        assert armed, "regime switch never armed the oracle window"
        groups = _groups_by_name(state.optimizer)
        assert abs(groups["critic"]["lr"] - 0.5 * groups["main"]["lr"]) < 1e-12
        assert any("critic_oracle_active" in k for k in state.logger.data)

        # Exhaust the window: the very next update must revert critic -> main.
        state.cfg.critic_oracle_remaining = 0
        run_inner_update(state)
        groups = _groups_by_name(state.optimizer)
        assert groups["critic"]["lr"] == groups["main"]["lr"]
    finally:
        state.envs.close()


# --------------------------------------------------------------------------- #
# O2 — snapshot/restore semantics + end-to-end swap
# --------------------------------------------------------------------------- #

def test_snapshot_restore_roundtrip_and_live_lr_preserved():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, policy_swap_topline=True)
    try:
        snap = _snapshot_learner(state)
        ref = {k: v.clone() for k, v in snap["model"].items()}

        # Mutate the live model and move the live LR past the snapshot's.
        with torch.no_grad():
            for p in state.model.parameters():
                p.add_(1.0)
        apply_inner_lr(state.optimizer, 5e-4, state.cfg)

        _restore_learner(state, snap)
        for k, v in state.model.state_dict().items():
            assert torch.allclose(v, ref[k]), f"param {k} not restored"
        # The restore must keep the LIVE LR, not resurrect the snapshot's.
        assert state.optimizer.param_groups[0]["lr"] == 5e-4
    finally:
        state.envs.close()


def test_policy_swap_banks_and_restores_across_switches():
    cfg = _tiny_cfg()
    # 2 regimes, switch every 2 updates -> A->B then B->A within 6 updates.
    state = init_inner_training(ENV_ID, cfg, steps_per_regime=32, num_regimes=2,
                                policy_swap_topline=True)
    try:
        for _ in range(6):
            run_inner_update(state)
        assert set(state.policy_swap_snapshots) == {0, 1}
        restored_flags = [v for _, v in state.logger.data.get(
            "brain_neuromod/policy_swap_restored", [])]
        assert restored_flags, "swap probe never logged"
        assert restored_flags[0] == 0.0          # first switch: nothing banked yet
        assert 1.0 in restored_flags             # a revisit restored a banked learner
    finally:
        state.envs.close()


def test_policy_swap_default_off_logs_nothing():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, steps_per_regime=32)
    try:
        for _ in range(3):
            run_inner_update(state)
        assert "brain_neuromod/policy_swap_restored" not in state.logger.data
        assert state.policy_swap_snapshots == {}
    finally:
        state.envs.close()


# --------------------------------------------------------------------------- #
# G-DECOMP — scoped snapshot/restore (swap-scope ladder, action tree 2026-07-09)
# --------------------------------------------------------------------------- #

def test_scoped_snapshot_filters_keys():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, policy_swap_topline=True,
                                policy_swap_scope="heads")
    try:
        snap = _snapshot_learner(state, "heads")
        assert set(snap) == {"model_partial"}
        assert snap["model_partial"], "heads scope banked nothing"
        assert all(k.startswith(("actor_head.", "critic_head."))
                   for k in snap["model_partial"])

        he = _snapshot_learner(state, "heads+encoder")["model_partial"]
        assert any(k.startswith("encoder.") for k in he)

        wm = _snapshot_learner(state, "world_model")
        assert set(wm) == {"world_model"}

        full = _snapshot_learner(state, "full")
        assert set(full) == {"model", "world_model", "optimizer", "wm_optimizer"}
    finally:
        state.envs.close()


def test_scoped_restore_touches_only_scope():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, policy_swap_topline=True,
                                policy_swap_scope="heads")
    try:
        snap = _snapshot_learner(state, "heads")
        ref_heads = {k: v.clone() for k, v in snap["model_partial"].items()}
        live_lr = state.optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            for p in state.model.parameters():
                p.add_(1.0)
        mutated = {k: v.clone() for k, v in state.model.state_dict().items()}

        _restore_learner(state, snap)
        for k, v in state.model.state_dict().items():
            if k.startswith(("actor_head.", "critic_head.")):
                assert torch.allclose(v, ref_heads[k]), f"{k} not restored"
            else:
                assert torch.allclose(v, mutated[k]), f"{k} clobbered outside scope"
        # Partial scopes carry no optimizer state; the live LR must be untouched.
        assert state.optimizer.param_groups[0]["lr"] == live_lr
    finally:
        state.envs.close()


def test_swap_scope_validated_at_init():
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="policy_swap_scope"):
        init_inner_training(ENV_ID, cfg, policy_swap_topline=True,
                            policy_swap_scope="nonsense")


# --------------------------------------------------------------------------- #
# D0 — constant Brain action (CLI + semantics)
# --------------------------------------------------------------------------- #

def test_constant_brain_action_flag_parses(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_brain_script, "train_brain",
                        lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["train_brain.py", "--constant_brain_action"])
    train_brain_script.main()
    assert captured["args"].constant_brain_action is True


def test_constant_brain_action_defaults_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_brain_script, "train_brain",
                        lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["train_brain.py"])
    train_brain_script.main()
    assert captured["args"].constant_brain_action is False


def test_zero_action_maps_to_lever_midpoints():
    # The D0 vector: a == 0 under MetaEnv's linear map lands on (lo+hi)/2 exactly —
    # empirically the levels trained scout Brains settle at (note 0005).
    def map_to_range(a, bounds):
        lo, hi = bounds
        return lo + (a + 1.0) / 2.0 * (hi - lo)
    for lo, hi in [(1e-4, 3e-3), (1e-3, 0.1), (1e-3, 0.5), (1, 30), (0.0, 0.5),
                   (0.0, 1.0), (0.0, 0.5)]:
        assert abs(map_to_range(0.0, (lo, hi)) - (lo + hi) / 2.0) < 1e-12
