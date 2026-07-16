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
# G3 head-bank memory (LOOP-0012) — trigger/selection rungs
# --------------------------------------------------------------------------- #

def test_head_bank_oracle_banks_and_restores_across_switches():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, steps_per_regime=32, num_regimes=2,
                                head_bank_slots=2)
    try:
        for _ in range(6):
            run_inner_update(state)
        assert set(state.head_bank) == {0, 1}
        restored = [v for _, v in state.logger.data.get(
            "brain_neuromod/head_bank_restored", [])]
        assert restored, "head bank never fired on oracle switches"
        assert restored[0] == 0.0            # first switch: warm start, nothing banked
        assert 1.0 in restored               # a revisit restored a banked slot
    finally:
        state.envs.close()


def test_head_bank_surprise_trigger_flips_slot():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="surprise", head_bank_select="other",
                                head_bank_surprise_threshold=1.0)
    try:
        from lifelong_learning.agents.ppo.train import _head_bank_surprise_check
        run_inner_update(state)              # buffer + logger populated
        _head_bank_surprise_check(state, 1.0)     # seeds the EMA baseline
        assert state.head_bank_active == 0
        _head_bank_surprise_check(state, 10.0)    # >2x baseline -> fires
        assert state.head_bank_active == 1
        assert 0 in state.head_bank               # outgoing slot banked
        assert state.head_bank_cooldown > 0
        _head_bank_surprise_check(state, 100.0)   # inside cooldown -> must NOT fire
        assert state.head_bank_active == 1
    finally:
        state.envs.close()


def test_head_bank_value_error_selects_best_critic():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=3,
                                head_bank_trigger="surprise", head_bank_select="value_error")
    try:
        from lifelong_learning.agents.ppo.train import (
            _head_bank_select_value_error,
            _head_bank_snapshot,
        )
        run_inner_update(state)
        # Craft slot 1: same heads but the critic's final bias shifted far positive.
        shifted = _head_bank_snapshot(state)
        bias_key = next(k for k in shifted["policy"] if k.startswith("critic_head")
                        and k.endswith("bias") and shifted["policy"][k].numel() == 1)
        shifted["policy"][bias_key] = shifted["policy"][bias_key] + 50.0
        state.head_bank[1] = shifted
        live_heads = {k: v.clone() for k, v in _head_bank_snapshot(state)["policy"].items()}

        # Returns near the live critic's range -> live (active) slot wins.
        state.buffer.returns = torch.zeros_like(state.buffer.returns)
        assert _head_bank_select_value_error(state) == 0
        # Returns near +50 -> the shifted slot wins.
        state.buffer.returns = torch.full_like(state.buffer.returns, 50.0)
        assert _head_bank_select_value_error(state) == 1
        # Selection must not mutate the live weights.
        for k, v in _head_bank_snapshot(state)["policy"].items():
            assert torch.allclose(v, live_heads[k]), f"{k} mutated by selection"
    finally:
        state.envs.close()


def test_head_bank_reward_error_selects_matching_regime_head():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="surprise", head_bank_select="reward_error")
    try:
        from lifelong_learning.agents.ppo.train import (
            _head_bank_select_reward_error,
            _head_bank_snapshot,
        )
        run_inner_update(state)
        # Craft slot 1: same policy heads, but a WM reward head biased far positive.
        other = _head_bank_snapshot(state)
        assert "wm_reward" in other, "reward_error slots must bank the WM reward head"
        bias_key = next(k for k in other["wm_reward"] if k.endswith("bias"))
        other["wm_reward"][bias_key] = other["wm_reward"][bias_key] + 50.0
        state.head_bank[1] = other
        live_wm = {k: v.clone() for k, v in state.world_model.state_dict().items()}

        # Observed rewards near zero -> the live (active) head fits better.
        state.buffer.extrinsic_rewards = torch.zeros_like(state.buffer.extrinsic_rewards)
        assert _head_bank_select_reward_error(state) == 0
        # Observed rewards near +50 -> the shifted head fits better.
        state.buffer.extrinsic_rewards = torch.full_like(state.buffer.extrinsic_rewards, 50.0)
        assert _head_bank_select_reward_error(state) == 1
        # Selection must leave the live world model untouched.
        for k, v in state.world_model.state_dict().items():
            assert torch.allclose(v, live_wm[k]), f"WM {k} mutated by selection"
    finally:
        state.envs.close()


def test_head_bank_content_addressing_spawns_then_selects():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="surprise", head_bank_select="reward_error")
    try:
        from lifelong_learning.agents.ppo.train import _head_bank_surprise_check
        run_inner_update(state)
        _head_bank_surprise_check(state, 1.0)      # seed EMA
        # First fire: no banked fingerprints -> must SPAWN slot 1, not stay on 0.
        _head_bank_surprise_check(state, 10.0)
        assert state.head_bank_active == 1
        assert 0 in state.head_bank
        # Second fire (cooldown expired): all slots live -> content addressing runs.
        state.head_bank_cooldown = 0
        state.buffer.extrinsic_rewards = torch.zeros_like(state.buffer.extrinsic_rewards)
        _head_bank_surprise_check(state, 100.0)
        assert set(state.head_bank) == {0, 1}      # both fingerprints banked
    finally:
        state.envs.close()


def test_head_bank_reward_fp_scores_in_frozen_feature_space():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="surprise", head_bank_select="reward_fp")
    try:
        from lifelong_learning.agents.ppo.train import (
            _head_bank_select_reward_fp,
            _head_bank_snapshot,
            _head_bank_load,
        )
        run_inner_update(state)
        # Craft slot 1: a full-WM fingerprint whose reward head is biased far positive.
        other = _head_bank_snapshot(state)
        assert "wm_full" in other, "reward_fp slots must bank the full WM"
        other["wm_full"]["reward_head.bias"] = other["wm_full"]["reward_head.bias"] + 50.0
        state.head_bank[1] = other
        live_wm = {k: v.clone() for k, v in state.world_model.state_dict().items()}
        live_model = {k: v.clone() for k, v in state.model.state_dict().items()}

        # Rewards near zero: the live WM (active slot) fits better -> stay.
        state.buffer.extrinsic_rewards = torch.zeros_like(state.buffer.extrinsic_rewards)
        assert _head_bank_select_reward_fp(state) == 0
        # Rewards near +50: slot 1's banked fingerprint fits better -> flip.
        state.buffer.extrinsic_rewards = torch.full_like(state.buffer.extrinsic_rewards, 50.0)
        assert _head_bank_select_reward_fp(state) == 1
        # Selection must never touch the live world model (scratch module only).
        for k, v in state.world_model.state_dict().items():
            assert torch.allclose(v, live_wm[k]), f"live WM {k} mutated by selection"

        # Restoring an fp slot loads policy heads only — the live WM stays live.
        _head_bank_load(state, other)
        for k, v in state.world_model.state_dict().items():
            assert torch.allclose(v, live_wm[k]), f"restore touched live WM {k}"
        changed = any(
            not torch.allclose(v, live_model[k])
            for k, v in state.model.state_dict().items()
            if k.startswith(("actor_head.", "critic_head."))
        ) or all(
            torch.allclose(other["policy"][k], live_model[k]) for k in other["policy"]
        )
        assert changed, "restore loaded neither heads nor was a no-op copy"
    finally:
        state.envs.close()


def test_head_bank_step_trigger_fires_on_success_collapse():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="step_surprise", head_bank_select="other")
    try:
        import numpy as np
        from lifelong_learning.agents.ppo.train import _head_bank_step_check
        state.global_step = 60000                       # past warmup
        one_done = np.array([True, False])
        r_success = np.array([0.9, 0.0])
        for _ in range(50):                             # a learned regime: slow EMA ~1.0
            _head_bank_step_check(state, r_success, one_done)
        assert state.head_bank_active == 0

        r_fail = np.array([0.0, 0.0])                   # goal swap: successes stop
        fired_at = None
        for k in range(30):
            _head_bank_step_check(state, r_fail, one_done)
            if state.head_bank_active == 1:
                fired_at = k
                break
        assert fired_at is not None, "collapse never fired"
        assert fired_at >= 5                            # several failures, not one blip
        assert 0 in state.head_bank                     # outgoing heads banked

        # Step cooldown blocks an immediate re-fire under continued failure.
        for _ in range(10):
            _head_bank_step_check(state, r_fail, one_done)
        assert state.head_bank_active == 1
    finally:
        state.envs.close()


def test_head_bank_step_shadow_logs_but_never_switches():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                                head_bank_trigger="step_surprise", head_bank_select="other",
                                head_bank_step_shadow=True)
    try:
        import numpy as np
        from lifelong_learning.agents.ppo.train import _head_bank_step_check
        state.global_step = 60000
        one_done = np.array([True, False])
        for _ in range(50):
            _head_bank_step_check(state, np.array([0.9, 0.0]), one_done)
        for _ in range(20):
            _head_bank_step_check(state, np.array([0.0, 0.0]), one_done)
        assert "brain_neuromod/head_bank_step_fire" in state.logger.data
        assert state.head_bank_active == 0              # never switched
        assert state.head_bank == {}                    # nothing banked
    finally:
        state.envs.close()


def test_head_bank_combo_validation():
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="oracle trigger"):
        init_inner_training(ENV_ID, cfg, head_bank_slots=2, head_bank_select="other")
    with pytest.raises(ValueError, match="regime id"):
        init_inner_training(ENV_ID, cfg, head_bank_slots=2,
                            head_bank_trigger="surprise", head_bank_select="oracle")
    with pytest.raises(ValueError, match="K=2|head_bank_slots=2"):
        init_inner_training(ENV_ID, cfg, head_bank_slots=3,
                            head_bank_trigger="surprise", head_bank_select="other")
    with pytest.raises(ValueError, match="mutually exclusive"):
        init_inner_training(ENV_ID, cfg, head_bank_slots=2, policy_swap_topline=True)


def test_head_bank_default_off_logs_nothing():
    cfg = _tiny_cfg()
    state = init_inner_training(ENV_ID, cfg, steps_per_regime=32)
    try:
        for _ in range(3):
            run_inner_update(state)
        assert "brain_neuromod/head_bank_restored" not in state.logger.data
        assert state.head_bank == {}
    finally:
        state.envs.close()


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
