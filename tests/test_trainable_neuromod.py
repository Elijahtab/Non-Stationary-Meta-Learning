"""
Regression tests for the trainable-vs-frozen neuromodulation decoder.

The whole point of the `trainable_neuromod` flag is a single, checkable property:
does inner-training gradient reach the code->mask decoder?

  - frozen (default): NO gradient -> decoder stays a fixed random projection (paper behavior)
  - trainable:        YES gradient -> decoder co-adapts with the rest of the inner network

Both modes must compute the *same* mask for the same code (only gradient flow differs),
and the zero-code identity property must hold in both.
"""
import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic

OBS_SHAPE = (21, 8, 8)
N_ACTIONS = 3
CODE = torch.tensor([0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3])


def _decoder_grad_after_backward(trainable: bool):
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=trainable)
    net.set_context_code(CODE)
    obs = torch.zeros(2, *OBS_SHAPE)
    obs[:, 0] = 1.0
    logits, value = net(obs)  # the exact path inner PPO uses (forward -> active_mask)
    (logits.sum() + value.sum()).backward()
    return net.neuromodulator.decoder[0].weight.grad


def test_frozen_decoder_receives_no_gradient():
    grad = _decoder_grad_after_backward(trainable=False)
    assert grad is None, "frozen decoder must not receive gradient (stays random)"


def test_trainable_decoder_receives_gradient():
    grad = _decoder_grad_after_backward(trainable=True)
    assert grad is not None, "trainable decoder must receive gradient"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "gradient should be non-zero for a non-zero code"


def test_encoder_and_heads_train_in_both_modes():
    # Sanity: the rest of the network always trains, regardless of decoder mode.
    for trainable in (False, True):
        net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=trainable)
        net.set_context_code(CODE)
        obs = torch.zeros(2, *OBS_SHAPE)
        obs[:, 0] = 1.0
        logits, value = net(obs)
        (logits.sum() + value.sum()).backward()
        assert net.encoder[0].weight.grad is not None
        assert net.actor_head[0].weight.grad is not None


def test_both_modes_compute_same_mask_for_same_code():
    frozen = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=False)
    trainable = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=True)
    trainable.load_state_dict(frozen.state_dict())  # identical decoder weights
    frozen.set_context_code(CODE)
    trainable.set_context_code(CODE)
    with torch.no_grad():
        m_frozen = frozen.neuromodulator.active_mask()
        m_trainable = trainable.neuromodulator.active_mask()
    assert torch.allclose(m_frozen, m_trainable, atol=1e-6)


def test_zero_code_is_identity_mask_in_both_modes():
    for trainable in (False, True):
        net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=trainable)
        net.set_context_code(torch.zeros(8))
        with torch.no_grad():
            mask = net.neuromodulator.active_mask()
        assert torch.allclose(mask, torch.ones_like(mask), atol=1e-6)


def test_decoder_weight_norm_is_reported():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, trainable_neuromod=True)
    norm = net.neuromodulator.decoder_weight_norm()
    assert isinstance(norm, float) and norm > 0
