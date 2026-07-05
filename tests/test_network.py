import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic, CONTEXT_CODE_DIM


def test_zero_context_code_keeps_neuromodulation_neutral():
    model = CNNActorCritic((21, 8, 8), 3)
    zero_code = torch.zeros(CONTEXT_CODE_DIM)

    model.set_context_code(zero_code)

    assert torch.allclose(model.neuro_mask, torch.ones_like(model.neuro_mask))


def test_zero_context_matches_unmasked_forward_pass():
    torch.manual_seed(0)
    model = CNNActorCritic((21, 8, 8), 3)
    obs = torch.randn(2, 21, 8, 8)

    model.set_context_code(torch.zeros(CONTEXT_CODE_DIM))
    zero_logits, zero_value = model.forward(obs)
    clear_logits, clear_value = model.forward_with_mask(obs, torch.ones_like(model.neuro_mask))

    assert torch.allclose(zero_logits, clear_logits)
    assert torch.allclose(zero_value, clear_value)


def test_actor_only_mask_leaves_critic_invariant():
    torch.manual_seed(0)
    model = CNNActorCritic((21, 5, 5), 3, actor_only_neuromod=True)
    model.eval()
    obs = torch.rand(4, 21, 5, 5)

    model.set_context_code(torch.zeros(CONTEXT_CODE_DIM))
    with torch.no_grad():
        logits_zero, value_zero = model(obs)

    model.set_context_code(torch.ones(CONTEXT_CODE_DIM))
    with torch.no_grad():
        logits_code, value_code = model(obs)

    assert torch.allclose(value_zero, value_code), "critic must ignore the mask in actor-only mode"
    assert not torch.allclose(logits_zero, logits_code), "actor must still be modulated"


def test_shared_mask_still_moves_critic_by_default():
    torch.manual_seed(0)
    model = CNNActorCritic((21, 5, 5), 3)
    model.eval()
    obs = torch.rand(4, 21, 5, 5)

    model.set_context_code(torch.zeros(CONTEXT_CODE_DIM))
    with torch.no_grad():
        _, value_zero = model(obs)
    model.set_context_code(torch.ones(CONTEXT_CODE_DIM))
    with torch.no_grad():
        _, value_code = model(obs)

    assert not torch.allclose(value_zero, value_code)


def test_gain_alpha_mask_range_and_zero_code_identity():
    from lifelong_learning.agents.brain.neuromod import FeatureMaskNeuromodulator

    torch.manual_seed(0)
    mod = FeatureMaskNeuromodulator(feature_dim=64, gain_alpha=0.5)

    identity = mod.decode_context_code(torch.zeros(1, mod.context_dim))
    assert torch.allclose(identity, torch.ones_like(identity))

    mask = mod.decode_context_code(torch.ones(1, mod.context_dim))
    assert float(mask.min()) >= 0.5 - 1e-6
    assert float(mask.max()) <= 1.5 + 1e-6
    assert float(mask.min()) < 1.0 < float(mask.max()), "two-sided: both damping and gain"


def test_gain_alpha_zero_is_byte_identical_to_suppress_mask():
    from lifelong_learning.agents.brain.neuromod import FeatureMaskNeuromodulator

    torch.manual_seed(1)
    a = FeatureMaskNeuromodulator(feature_dim=32)
    torch.manual_seed(1)
    b = FeatureMaskNeuromodulator(feature_dim=32, gain_alpha=0.0)

    code = torch.randn(1, a.context_dim)
    assert torch.equal(a.decode_context_code(code), b.decode_context_code(code))
