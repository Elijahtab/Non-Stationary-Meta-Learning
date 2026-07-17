"""
Tests for LOOP-0016 / W0c — the probe-only dormancy measurement (research-log 0014).

Contract of dormancy_probe():
  - Measures dormant fractions at five sites: encoder conv1/conv2/conv3 (per-channel,
    post-ReLU) and actor/critic head hidden layers (per-unit, the redo_reset_heads
    statistic) at the primary tau and a descriptive secondary tau.
  - Is a PURE measurement: model params, optimizer state, and the torch RNG state are
    byte-identical before and after (the P-W0c2 composite-inertness gate depends on this).
  - Flag-guarded default-off in the update loop (dormancy_probe_interval=0 logs nothing).

Determinism trick borrowed from test_redo_reset.py: drive units purely from their bias —
zero incoming weights, large +bias (always active) or large -bias (dead through the ReLU).
"""
import copy

import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    dormancy_probe,
    init_inner_training,
    run_inner_update,
)

OBS_SHAPE = (21, 8, 8)
N_ACTIONS = 3
ENV_ID = "MiniGrid-MultiGoal-8x8-v0"

SITES = ("conv1", "conv2", "conv3", "actor", "critic")


def _obs(batch=8):
    obs = torch.zeros(batch, *OBS_SHAPE)
    obs[:, 0] = 1.0
    return obs


def test_probe_returns_all_sites_with_valid_fractions():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    fractions = dormancy_probe(net, _obs())
    for site in SITES:
        for key in (f"dormancy_{site}", f"dormancy10_{site}"):
            assert key in fractions
            assert 0.0 <= fractions[key] <= 1.0


def test_probe_detects_planted_dead_conv_channel():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    conv1 = net.encoder[0]
    with torch.no_grad():
        # All conv1 channels live via bias (activation == bias, equal scores)...
        conv1.weight.zero_()
        conv1.bias.fill_(100.0)
        # ...except one forced dead through the ReLU.
        conv1.bias[4] = -1e3
    fractions = dormancy_probe(net, _obs(), tau=0.0)
    n_channels = conv1.out_channels
    assert fractions["dormancy_conv1"] == 1.0 / n_channels


def test_probe_is_pure_measurement():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # A step so optimizer state is nonzero (a mutation would be visible).
    logits, value = net(_obs())
    (logits.sum() + value.sum()).backward()
    opt.step()

    params_before = copy.deepcopy(net.state_dict())
    opt_before = copy.deepcopy(opt.state_dict())
    rng_before = torch.get_rng_state().clone()

    dormancy_probe(net, _obs())

    for k, v in net.state_dict().items():
        assert torch.equal(v, params_before[k]), f"param {k} mutated by the probe"
    after = opt.state_dict()
    for pid, st in opt_before["state"].items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(after["state"][pid][k], v), "optimizer state mutated"
            else:
                assert after["state"][pid][k] == v
    assert torch.equal(torch.get_rng_state(), rng_before), "probe consumed RNG"


def test_probe_runs_inside_update_loop_and_off_is_noop():
    cfg = PPOConfig(total_timesteps=2_000, num_envs=2, num_steps=16, seed=0,
                    device="cpu", mode="dyna")
    # Probe ON every update: logs all five sites at both taus.
    state = init_inner_training(ENV_ID, cfg, dormancy_probe_interval=1)
    try:
        run_inner_update(state)
        for site in SITES:
            assert f"brain_neuromod/dormancy_{site}" in state.logger.data
            assert f"brain_neuromod/dormancy10_{site}" in state.logger.data
    finally:
        state.envs.close()

    # Probe OFF (default): nothing logged.
    state = init_inner_training(ENV_ID, cfg)
    try:
        run_inner_update(state)
        assert not any(k.startswith("brain_neuromod/dormancy") for k in state.logger.data)
    finally:
        state.envs.close()
