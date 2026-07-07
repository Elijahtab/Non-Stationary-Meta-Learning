"""
Regression tests for LOOP-0007 candidate 1 — decoupled critic learning rate
(research note 0004, code-free plasticity/stability family).

The mechanism's contract:
  - critic_lr_scale == 1.0 (default): single-group optimizer, byte-identical to
    the prior baseline (no "critic" group exists).
  - critic_lr_scale != 1.0: the critic head lives in its own optimizer group
    ("critic") whose LR *tracks* the main LR scaled by critic_lr_scale — so the
    Brain's proven LR lever (which writes the main group) still reaches the
    critic, damped, rather than being silently bypassed.
  - apply_inner_lr is the single write path for the main LR; it must keep the
    critic group at main * scale and must leave the decoder group (fixed LR) alone.
"""
from types import SimpleNamespace

import torch

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import apply_inner_lr, init_inner_training

ENV_ID = "MiniGrid-MultiGoal-8x8-v0"


def _tiny_cfg(lr: float = 2.5e-4) -> PPOConfig:
    return PPOConfig(
        total_timesteps=2_000,
        num_envs=2,
        num_steps=16,
        seed=0,
        device="cpu",
        mode="dyna",
        lr=lr,
    )


def _groups_by_name(optimizer):
    return {g.get("name"): g for g in optimizer.param_groups}


# --------------------------------------------------------------------------- #
# apply_inner_lr — the pure LR write path
# --------------------------------------------------------------------------- #

def test_apply_inner_lr_mirrors_critic_group():
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "critic", "lr": 0.0},
    ])
    cfg = SimpleNamespace(critic_lr_scale=0.5)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 5e-4  # main * 0.5


def test_apply_inner_lr_leaves_decoder_group_fixed():
    opt = SimpleNamespace(param_groups=[
        {"name": "main", "lr": 0.0},
        {"name": "decoder", "lr": 7e-6},
        {"name": "critic", "lr": 0.0},
    ])
    cfg = SimpleNamespace(critic_lr_scale=0.5)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 7e-6  # decoder untouched
    assert opt.param_groups[2]["lr"] == 5e-4


def test_apply_inner_lr_noop_when_scale_off():
    # scale == 1.0 and no critic group -> only the main group is written, no crash.
    opt = SimpleNamespace(param_groups=[{"name": "main", "lr": 0.0}])
    cfg = SimpleNamespace(critic_lr_scale=1.0)
    apply_inner_lr(opt, 1e-3, cfg)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_apply_inner_lr_tolerates_missing_cfg():
    opt = SimpleNamespace(param_groups=[{"name": "main", "lr": 0.0}])
    apply_inner_lr(opt, 1e-3, None)
    assert opt.param_groups[0]["lr"] == 1e-3


# --------------------------------------------------------------------------- #
# init_inner_training — optimizer construction
# --------------------------------------------------------------------------- #

def test_default_is_single_group_baseline():
    state = init_inner_training(ENV_ID, _tiny_cfg())
    try:
        assert len(state.optimizer.param_groups) == 1
        assert "critic" not in _groups_by_name(state.optimizer)
        assert state.cfg.critic_lr_scale == 1.0
    finally:
        state.envs.close()


def test_decoupled_critic_builds_scaled_group():
    lr = 2.5e-4
    scale = 0.5
    state = init_inner_training(ENV_ID, _tiny_cfg(lr=lr), critic_lr_scale=scale)
    try:
        groups = _groups_by_name(state.optimizer)
        assert set(groups) == {"main", "critic"}
        assert abs(groups["main"]["lr"] - lr) < 1e-12
        assert abs(groups["critic"]["lr"] - lr * scale) < 1e-12

        # The critic group holds EXACTLY the critic-head params; main holds none of them.
        critic_ids = {id(p) for p in state.model.critic_head.parameters()}
        assert {id(p) for p in groups["critic"]["params"]} == critic_ids
        assert not (critic_ids & {id(p) for p in groups["main"]["params"]})
    finally:
        state.envs.close()


def test_brain_lever_scales_critic_via_apply_inner_lr():
    lr = 2.5e-4
    scale = 0.5
    state = init_inner_training(ENV_ID, _tiny_cfg(lr=lr), critic_lr_scale=scale)
    try:
        new_lr = 9e-4  # a Brain lever setting distinct from the init LR
        apply_inner_lr(state.optimizer, new_lr, state.cfg)
        groups = _groups_by_name(state.optimizer)
        assert abs(groups["main"]["lr"] - new_lr) < 1e-12
        assert abs(groups["critic"]["lr"] - new_lr * scale) < 1e-12
    finally:
        state.envs.close()


def test_decoupled_critic_composes_with_decoder_group():
    lr = 2.5e-4
    state = init_inner_training(
        ENV_ID,
        _tiny_cfg(lr=lr),
        trainable_neuromod=True,
        neuromod_decoder_lr=1e-5,
        critic_lr_scale=0.5,
    )
    try:
        groups = _groups_by_name(state.optimizer)
        assert set(groups) == {"main", "decoder", "critic"}
        assert abs(groups["decoder"]["lr"] - 1e-5) < 1e-12
        assert abs(groups["critic"]["lr"] - lr * 0.5) < 1e-12

        # A Brain lever write mirrors critic but NOT the fixed decoder LR.
        apply_inner_lr(state.optimizer, 9e-4, state.cfg)
        groups = _groups_by_name(state.optimizer)
        assert abs(groups["main"]["lr"] - 9e-4) < 1e-12
        assert abs(groups["critic"]["lr"] - 9e-4 * 0.5) < 1e-12
        assert abs(groups["decoder"]["lr"] - 1e-5) < 1e-12
    finally:
        state.envs.close()
