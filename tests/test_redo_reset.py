"""
Regression tests for LOOP-0007 candidate 2 — ReDo dormant-neuron reset (research note
0004; Sokar 2023).

Contract of redo_reset_heads():
  - A hidden unit whose normalized mean-abs activation over the batch is <= tau is
    "dormant" and gets reset: incoming weights re-initialised, bias zeroed, outgoing
    weights zeroed, and its Adam moment estimates cleared.
  - Non-dormant units are left untouched.
  - Returns the dormant fraction per head (the registered mechanism probe).

Dormancy is made deterministic by driving every hidden unit purely from its bias:
zero the incoming weights and set a large +bias (always active) or a large -bias
(always dead through the ReLU), independent of the observation.
"""
import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    redo_reset_heads,
    init_inner_training,
    run_inner_update,
)

OBS_SHAPE = (21, 8, 8)
N_ACTIONS = 3
ENV_ID = "MiniGrid-MultiGoal-8x8-v0"


def _obs(batch=8):
    obs = torch.zeros(batch, *OBS_SHAPE)
    obs[:, 0] = 1.0
    return obs


def _force_all_live(net):
    """Every hidden unit active via a large positive bias, weights zeroed (activation == bias)."""
    with torch.no_grad():
        for head in (net.actor_head, net.critic_head):
            head[0].weight.zero_()
            head[0].bias.fill_(100.0)


def _kill(head, idx):
    """Force hidden unit `idx` permanently dead (pre-activation << 0 -> ReLU 0)."""
    with torch.no_grad():
        head[0].bias[idx] = -1e3


def test_no_dormant_units_leaves_weights_unchanged():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_all_live(net)
    before = net.actor_head[0].weight.detach().clone()
    fractions = redo_reset_heads(net, opt, _obs(), tau=0.0)
    assert fractions["actor"] == 0.0 and fractions["critic"] == 0.0
    assert torch.allclose(net.actor_head[0].weight, before)


def test_dormant_unit_is_reset():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_all_live(net)
    idx = 5
    _kill(net.actor_head, idx)
    in_w_before = net.actor_head[0].weight[idx].detach().clone()  # all zeros
    with torch.no_grad():
        net.actor_head[2].weight[:, idx] = 0.7  # nonzero outgoing to observe zeroing

    fractions = redo_reset_heads(net, opt, _obs(), tau=0.0)

    hidden = net.actor_head[0].weight.shape[0]
    assert fractions["actor"] == 1.0 / hidden  # exactly one dormant unit
    assert not torch.allclose(net.actor_head[0].weight[idx], in_w_before)
    assert net.actor_head[0].weight[idx].abs().sum() > 0  # re-initialised
    assert float(net.actor_head[0].bias[idx]) == 0.0       # bias zeroed
    assert torch.allclose(net.actor_head[2].weight[:, idx], torch.zeros(N_ACTIONS))  # outgoing zeroed


def test_adam_state_cleared_for_reset_unit():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_all_live(net)
    idx = 3
    _kill(net.actor_head, idx)

    # a step so Adam moments become nonzero
    logits, value = net(_obs())
    (logits.sum() + value.sum()).backward()
    opt.step()

    in_w = net.actor_head[0].weight
    redo_reset_heads(net, opt, _obs(), tau=0.0)
    assert torch.count_nonzero(opt.state[in_w]["exp_avg"][idx]) == 0
    assert torch.count_nonzero(opt.state[in_w]["exp_avg_sq"][idx]) == 0


def test_non_dormant_units_untouched():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_all_live(net)
    _kill(net.actor_head, 7)
    live_before = net.actor_head[0].weight[0].detach().clone()
    live_bias_before = float(net.actor_head[0].bias[0])

    redo_reset_heads(net, opt, _obs(), tau=0.0)
    assert torch.allclose(net.actor_head[0].weight[0], live_before)
    assert float(net.actor_head[0].bias[0]) == live_bias_before


def test_redo_runs_inside_update_loop_and_off_is_noop():
    cfg = PPOConfig(total_timesteps=2_000, num_envs=2, num_steps=16, seed=0,
                    device="cpu", mode="dyna")
    # ReDo ON every update: run a couple of updates, must not crash and must log the probe.
    state = init_inner_training(ENV_ID, cfg, redo_interval=1)
    try:
        for _ in range(2):
            out = run_inner_update(state)
        assert "value_loss" in out
        assert any("redo_dormant_fraction" in k for k in state.logger.data)
    finally:
        state.envs.close()

    # ReDo OFF (default): probe is never logged.
    state = init_inner_training(ENV_ID, cfg)
    try:
        run_inner_update(state)
        assert not any("redo_dormant_fraction" in k for k in state.logger.data)
    finally:
        state.envs.close()
