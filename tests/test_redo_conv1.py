"""
Tests for LOOP-0017 / branch F — conv1-targeted ReDo (research-log 0015).

Contract of redo_reset_conv1():
  - A conv1 channel whose normalized mean-abs post-ReLU activation is <= tau is reset:
    incoming filter re-initialised, bias zeroed, conv2's incoming weights for that channel
    zeroed, Adam moments cleared for the touched slices.
  - Non-dormant channels and every other layer are untouched.
  - Returns the pre-reset dormant fraction (the mechanism probe).
  - Flag-guarded default-off in the update loop (redo_conv1_interval=0 logs nothing).

Determinism trick from test_redo_reset.py: drive channels purely from bias — weights
zeroed, large +bias (always active) or large -bias (dead through the ReLU).
"""
import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    redo_reset_conv1,
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


def _force_conv1_live(net):
    with torch.no_grad():
        net.encoder[0].weight.zero_()
        net.encoder[0].bias.fill_(100.0)


def test_no_dormant_channels_is_pure_noop():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_conv1_live(net)
    w_before = net.encoder[0].weight.detach().clone()
    conv2_before = net.encoder[2].weight.detach().clone()
    frac = redo_reset_conv1(net, opt, _obs(), tau=0.0)
    assert frac == 0.0
    assert torch.equal(net.encoder[0].weight, w_before)
    assert torch.equal(net.encoder[2].weight, conv2_before)


def test_dormant_channel_is_reset_and_outgoing_zeroed():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_conv1_live(net)
    idx = 4
    with torch.no_grad():
        net.encoder[0].bias[idx] = -1e3          # dead through the ReLU
        net.encoder[2].weight[:, idx] = 0.7      # nonzero outgoing to observe zeroing

    frac = redo_reset_conv1(net, opt, _obs(), tau=0.0)

    n_channels = net.encoder[0].out_channels
    assert frac == 1.0 / n_channels
    assert net.encoder[0].weight[idx].abs().sum() > 0        # re-initialised
    assert float(net.encoder[0].bias[idx]) == 0.0            # bias zeroed
    assert torch.count_nonzero(net.encoder[2].weight[:, idx]) == 0  # outgoing zeroed


def test_adam_state_cleared_and_live_channels_untouched():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    _force_conv1_live(net)
    idx = 7
    with torch.no_grad():
        net.encoder[0].bias[idx] = -1e3

    logits, value = net(_obs())
    (logits.sum() + value.sum()).backward()
    opt.step()

    live_w = net.encoder[0].weight[0].detach().clone()
    live_b = float(net.encoder[0].bias[0])
    redo_reset_conv1(net, opt, _obs(), tau=0.0)

    w = net.encoder[0].weight
    assert torch.count_nonzero(opt.state[w]["exp_avg"][idx]) == 0
    assert torch.count_nonzero(opt.state[w]["exp_avg_sq"][idx]) == 0
    c2 = net.encoder[2].weight
    assert torch.count_nonzero(opt.state[c2]["exp_avg"][:, idx]) == 0
    assert torch.equal(net.encoder[0].weight[0], live_w)
    assert float(net.encoder[0].bias[0]) == live_b


def test_runs_inside_update_loop_and_off_is_noop():
    cfg = PPOConfig(total_timesteps=2_000, num_envs=2, num_steps=16, seed=0,
                    device="cpu", mode="dyna")
    state = init_inner_training(ENV_ID, cfg, redo_conv1_interval=1)
    try:
        run_inner_update(state)
        assert "brain_neuromod/redo_conv1_dormant_fraction" in state.logger.data
    finally:
        state.envs.close()

    state = init_inner_training(ENV_ID, cfg)
    try:
        run_inner_update(state)
        assert "brain_neuromod/redo_conv1_dormant_fraction" not in state.logger.data
    finally:
        state.envs.close()
