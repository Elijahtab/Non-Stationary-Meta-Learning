# 05 — Neuromodulation

Neuromodulation is the mechanism that lets the Brain change *how the inner agent computes*,
not just how it is trained. The Brain emits an 8-dim **context code**; a small decoder turns it
into a **feature-wise mask** that gates the inner CNN's shared representation before both the
actor and the critic. This is the "learned, context-dependent routing" of paper §3.7, and it
is the primary **editable surface** for autoresearch.

Sources:
[`agents/brain/neuromod.py`](../../src/lifelong_learning/agents/brain/neuromod.py),
[`agents/ppo/network.py`](../../src/lifelong_learning/agents/ppo/network.py).

> **📊 Visual companion:** [neuromodulation-box-diagram.md](./neuromodulation-box-diagram.md) —
> a numbered box diagram of the whole forward pass (context code → frozen decoder → mask →
> gated features), with a per-box reference table.

## The action layout

The last 8 of the Brain's 15 action dims are the context code
([neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py#L11-L17)):

```python
SCALAR_BRAIN_ACTION_DIM = 7
CONTEXT_CODE_DIM = 8
BRAIN_ACTION_DIM = 15
BRAIN_CONTEXT_SLICE = slice(7, 15)
```

[`MetaEnv._apply_action`](../../src/lifelong_learning/agents/brain/meta_env.py#L356-L361) takes
`action[7:15]`, makes it a tensor, and calls `model.set_context_code(code)` each Brain decision
(unless `disable_neuromodulation`).

## The decoder & mask — [`FeatureMaskNeuromodulator`](../../src/lifelong_learning/agents/brain/neuromod.py#L80)

The decoder is `Linear(context_dim=8 → 256) → ReLU → Linear(256 → feature_dim)`
([neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py#L99-L103)), where
`feature_dim` is the flattened CNN size (`4096` for an 8×8 grid). The mask is computed by
[`decode_context_code`](../../src/lifelong_learning/agents/brain/neuromod.py#L106):

> **⚠️ The decoder is frozen at initialization — it never trains.** `set_context_code`
> decodes the mask under `torch.no_grad()` and stores it as a detached buffer
> (`current_mask`), and `forward` multiplies that buffer in. So no gradient ever reaches the
> decoder's weights (verified 2026-06-16: `decoder.weight.grad is None` after an inner
> backward pass, while encoder/head grads are populated). The decoder is therefore a **fixed
> random (orthogonal-init) projection**, and the *only* learned part of the modulation
> pathway is the Brain's context code. Making the decoder trainable (recompute the mask
> in-graph, or train it under a separate objective) is an obvious, well-motivated extension —
> see the workshop plan.

```python
suppression_template = sigmoid(decoder(code))                 # ∈ (0,1)^feature_dim
context_strength     = ‖code‖₂ / sqrt(context_dim)            # scalar, clamped to [0,1]
mask                 = 1 - context_strength * suppression_template   # ∈ (0,1]
```

Properties this guarantees:

- **`code = 0` ⇒ `mask = 1` everywhere** (identity). The network reduces exactly to its
  unmodulated baseline, so neuromodulation can never *hurt* a zero-code policy.
- **Larger `‖code‖` ⇒ stronger suppression.** The code magnitude is a global "how much to
  modulate" knob; its direction selects *which* features.
- The mask is **suppressive** (in `(0,1]`): it can damp features but not amplify them past 1.

## How the mask is applied — [`CNNActorCritic.forward_with_mask`](../../src/lifelong_learning/agents/ppo/network.py#L119)

```python
features = self.encoder(obs)              # (B, flat_size)
features = features * expand_mask(...)    # element-wise gate (broadcast over batch)
return self.actor_head(features), self.critic_head(features).squeeze(-1)
```

The **same gated features feed both heads**, so a context code changes both the action
distribution and the value estimate. The current mask is stored on the module
(`set_context_code` writes `current_mask` under `no_grad`) and used by the default `forward`
([network.py](../../src/lifelong_learning/agents/ppo/network.py#L99-L154)).

## Diagnostics — [`describe_neuromodulation`](../../src/lifelong_learning/agents/ppo/network.py#L124)

On each Brain decision, the inner observation batch is run through the network **twice** — once
with the current mask, once with an all-ones mask — to measure the causal effect of the code:

| Diagnostic | Meaning |
| --- | --- |
| `mask_mean`, `mask_std`, `mask_min`, `mask_max` | mask statistics |
| `channel_means` | per-channel mask strength, averaged over space (mask reshaped to `(B, 64, H, W)` → mean over `(B,H,W)`), i.e. **64 shared CNN channels** |
| `policy_kl_vs_unmasked` | `KL(π_masked ‖ π_unmasked)` |
| `entropy_delta_vs_unmasked` | `H(π_masked) − H(π_unmasked)` |
| `value_delta_abs_vs_unmasked` | `E[|V_masked − V_unmasked|]` |

These are logged by
[`log_neuromodulation_snapshot`](../../src/lifelong_learning/agents/brain/neuromod.py#L48)
under two prefixes ([neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py#L19-L20)):

- `brain_context/context_{i}` — the 8 raw context-code entries.
- `brain_neuromod/{mask_mean,…,policy_kl_vs_unmasked,value_delta_abs_vs_unmasked}` and
  `brain_neuromod/channel_mean_{i}`.

Two of these traces feed the scorer's neuromodulation-activity metric:
`brain_neuromod/policy_kl_vs_unmasked` and `brain_neuromod/value_delta_abs_vs_unmasked`
(see [07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md)). They are
also the four panels of the paper's §4.3.1 diagnostic dashboard (context code → decoded mask →
policy/value effect), rendered per inner run as `*_neuromodulation_dashboard.png`.

## Why this is the research surface

The decoder architecture and masking scheme are *deliberately isolated* in `neuromod.py` so
that autoresearch can iterate on them without touching the benchmark, scorer, or environment.
The research brief ([program_neuromod.md](../../config/program_neuromod.md)) directs the agent to
explore exactly this space: decoder width/depth, mask parameterization, where modulation is
applied (actor-only vs critic-only vs shared), suppressive vs gain-based masking, and larger
context dimensions. The **editable surface** for trials is `neuromod.py` + `network.py` (see
[07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md)).
