"""
Regression tests for LOOP-0007 candidate 3 — plasticity-preserving LayerNorm on the
shared encoder representation (research note 0004; Lyle 2023).

Contract:
  - plasticity_norm=False (default): no feature_norm module, no extra params, the
    forward path is byte-identical to the prior baseline.
  - plasticity_norm=True: a static LayerNorm(flat_size) is applied to the flattened
    encoder output before the neuromod mask and both heads; it adds params, receives
    gradient, and is shared by every consumer (forward, aux head).
  - The mask's zero-code identity property (mask == 1 at code 0) is unaffected — the
    normalization sits *before* the multiplicative mask.
"""
import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic

OBS_SHAPE = (21, 8, 8)
N_ACTIONS = 3
CODE = torch.tensor([0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3])


def _obs(batch=2):
    obs = torch.zeros(batch, *OBS_SHAPE)
    obs[:, 0] = 1.0
    return obs


def test_default_has_no_feature_norm():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    assert net.feature_norm is None
    assert not any("feature_norm" in k for k in net.state_dict())


def test_plasticity_norm_adds_layernorm():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=True)
    assert isinstance(net.feature_norm, torch.nn.LayerNorm)
    assert net.feature_norm.normalized_shape == (net.flat_size,)
    assert any("feature_norm" in k for k in net.state_dict())


def test_off_path_is_identical_to_baseline():
    # plasticity_norm=False must not perturb the forward pass at all.
    a = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=False)
    b = CNNActorCritic(OBS_SHAPE, N_ACTIONS)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        la, va = a(_obs())
        lb, vb = b(_obs())
    assert torch.allclose(la, lb, atol=1e-7)
    assert torch.allclose(va, vb, atol=1e-7)


def test_norm_changes_representation():
    # With the same encoder weights, enabling the norm must change the features the
    # heads see (LayerNorm standardises them), i.e. it is actually in the path.
    base = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=False)
    normed = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=True)
    # copy the shared/base params so only the LayerNorm differs
    normed.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        raw = base._encode(_obs())
        norm = normed._encode(_obs())
    assert not torch.allclose(raw, norm, atol=1e-4)
    # LayerNorm output is ~zero-mean unit-var per row
    assert norm.mean(dim=-1).abs().max() < 1e-4
    assert (norm.std(dim=-1, unbiased=False) - 1.0).abs().max() < 1e-2


def test_layernorm_receives_gradient():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=True)
    net.set_context_code(CODE)
    logits, value = net(_obs())
    (logits.sum() + value.sum()).backward()
    assert net.feature_norm.weight.grad is not None
    assert net.feature_norm.weight.grad.abs().sum() > 0
    # heads/encoder still train too
    assert net.encoder[0].weight.grad is not None
    assert net.actor_head[0].weight.grad is not None


def test_zero_code_still_identity_mask_with_norm():
    net = CNNActorCritic(OBS_SHAPE, N_ACTIONS, plasticity_norm=True)
    net.set_context_code(torch.zeros(8))
    with torch.no_grad():
        mask = net.neuromodulator.active_mask()
    assert torch.allclose(mask, torch.ones_like(mask), atol=1e-6)
