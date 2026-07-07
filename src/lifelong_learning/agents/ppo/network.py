from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lifelong_learning.agents.brain.neuromod import (
    CONTEXT_CODE_DIM,
    FeatureMaskNeuromodulator,
    gradient_gate,
)


class CNNActorCritic(nn.Module):
    """
    Actor-Critic network for MiniGrid with one-hot encoded observations.

    Architecture:
        - Shared 3-layer CNN encoder (input channels -> 32 -> 64 -> 64)
        - Neuromodulation gating via ContextDecoder (8-dim code -> sigmoid mask)
        - Decoupled Actor head (policy logits) and Critic head (state value)

    Input:  (B, C, H, W) one-hot tensor from OneHotPartialObsWrapper
    Output: (logits, value)
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        *,
        trainable_neuromod: bool = False,
        actor_only_neuromod: bool = False,
        neuromod_gain_alpha: float = 0.0,
        grad_gate_neuromod: bool = False,
        critic_code_neuromod: bool = False,
        aux_code_head: bool = False,
        plasticity_norm: bool = False,
    ):
        super().__init__()
        self.c, self.h, self.w = obs_shape
        self.feature_channels = 64
        # Mask scope: shared (default, paper-faithful) applies the mask before both heads;
        # actor-only leaves the critic reading raw encoder features (autoresearch trials
        # 20260704-022146/001 + 20260704-203740/001; confirmation sweep docs/plans/2026-07-05).
        # Neither flag adds parameters, so checkpoints are interchangeable across modes.
        self.actor_only_neuromod = actor_only_neuromod
        # LOOP-0006 learning-dynamics family (research note 0003 + loop note):
        #   grad_gate_neuromod  — the decoded mask gates the BACKWARD pass into the encoder
        #                         (forward untouched); mutually exclusive with the forward
        #                         mask modes above by construction (it replaces the multiply).
        #   critic_code_neuromod — the raw 8-D context code is concatenated to the critic
        #                          head input (adds params: critic first layer widens).
        #   aux_code_head        — small head predicting the current context code from
        #                          encoder features (adds params; loss hooked via ppo.py).
        self.grad_gate_neuromod = grad_gate_neuromod
        self.critic_code_neuromod = critic_code_neuromod

        # Shared CNN feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.c, self.h, self.w)
            flat_size = self.encoder(dummy).shape[1]
        self.flat_size = flat_size

        # LOOP-0007 cand 3 (research note 0004; Lyle 2023): static LayerNorm on the shared
        # flattened encoder representation to mitigate plasticity loss across regimes. No code,
        # no lever — a fixed architectural change applied to every forward before both the
        # neuromod mask and the heads. Adds params (LayerNorm weight/bias), so checkpoints are
        # NOT interchangeable with baseline. Also the A1 test: if it does nothing, plasticity
        # loss probably isn't the bottleneck on this 2-regime task.
        self.plasticity_norm = plasticity_norm
        self.feature_norm = nn.LayerNorm(flat_size) if plasticity_norm else None

        # Keep the neuromodulation implementation isolated from the rest of the
        # actor-critic so future research can iterate on it in one place.
        self.neuromodulator = FeatureMaskNeuromodulator(
            feature_dim=flat_size,
            context_dim=CONTEXT_CODE_DIM,
            trainable=trainable_neuromod,
            gain_alpha=neuromod_gain_alpha,
        )

        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        # Critic head (value function); with critic_code_neuromod the first layer also
        # reads the 8-D context code, so value can re-fit per regime through a small,
        # fast-adapting pathway without touching the policy path.
        critic_in_dim = flat_size + (CONTEXT_CODE_DIM if critic_code_neuromod else 0)
        self.critic_head = nn.Sequential(
            nn.Linear(critic_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Auxiliary regime-inference head: predict the Brain's context code from features
        # (code as teaching signal, not modulator). Only built when the aux loss is active.
        self.code_prediction_head = None
        if aux_code_head:
            self.code_prediction_head = nn.Sequential(
                nn.Linear(flat_size, 64),
                nn.ReLU(),
                nn.Linear(64, CONTEXT_CODE_DIM),
            )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Orthogonal init with role-specific gains for output layers."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Actor output: gain=0.01 -> near-uniform initial policy
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                gain = 0.01 if layer == self.actor_head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Critic output: gain=1.0 -> standard for value function
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                gain = 1.0 if layer == self.critic_head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @property
    def neuro_mask(self) -> torch.Tensor:
        """Compatibility view of the current neuromodulation mask."""
        return self.neuromodulator.current_mask

    def set_context_code(self, code: torch.Tensor):
        """
        Decode an 8-dim context code into a gating mask and store it.

        Args:
            code: Tensor of shape (CONTEXT_CODE_DIM,) with values in [-1, 1]
        """
        self.neuromodulator.set_context_code(code)

    def decode_context_code(self, code: torch.Tensor) -> torch.Tensor:
        """Map a context code to a suppressive mask, keeping zero-context neutral."""
        return self.neuromodulator.decode_context_code(code)

    def clear_context(self):
        """Reset the neuro mask to all-ones (no gating)."""
        self.neuromodulator.clear_context()

    def _expand_mask(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.neuromodulator.expand_mask(features, mask)

    def _batch_context_code(self, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        """The current 8-D context code, detached and broadcast over the batch."""
        code = self.neuromodulator.current_code.detach().to(dtype=ref.dtype)
        return code.expand(batch_size, -1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared encoder features, with optional static plasticity LayerNorm (cand 3).

        Single choke-point so every consumer (forward, aux head, diagnostics) sees the
        same representation. When plasticity_norm is off this is exactly self.encoder(obs).
        """
        features = self.encoder(obs)
        if self.feature_norm is not None:
            features = self.feature_norm(features)
        return features

    def forward_with_mask(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        features = self._encode(obs)
        if self.grad_gate_neuromod:
            # Plasticity gating: the mask leaves the forward pass entirely and instead
            # gates the gradient flowing back into the encoder (identity forward). Head
            # weights keep full plasticity; only encoder learning is code-directed.
            # code=0 => mask=1 => exact baseline in both directions.
            gated = gradient_gate(features, self._expand_mask(features, mask))
            critic_in = gated
            if self.critic_code_neuromod:
                critic_in = torch.cat(
                    [gated, self._batch_context_code(gated.shape[0], gated)], dim=-1
                )
            return self.actor_head(gated), self.critic_head(critic_in).squeeze(-1)
        masked = features * self._expand_mask(features, mask)
        critic_in = features if self.actor_only_neuromod else masked
        if self.critic_code_neuromod:
            critic_in = torch.cat(
                [critic_in, self._batch_context_code(critic_in.shape[0], critic_in)], dim=-1
            )
        return self.actor_head(masked), self.critic_head(critic_in).squeeze(-1)

    def aux_code_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """MSE between the code-prediction head's output and the current context code.

        Shapes the shared representation to make regime information (as carried by the
        Brain's code) linearly decodable — the code acts as a teaching signal. The target
        is the detached current code, constant over a minibatch.
        """
        assert self.code_prediction_head is not None, "aux_code_head was not enabled"
        features = self._encode(obs)
        pred = self.code_prediction_head(features)
        target = self._batch_context_code(pred.shape[0], pred)
        return torch.nn.functional.mse_loss(pred, target)

    def describe_neuromodulation(self, obs: torch.Tensor) -> dict[str, torch.Tensor | float]:
        """Summarize the current context mask and its effect on a reference batch."""
        with torch.no_grad():
            # active_mask(), not the snapshot buffer: in trainable mode the decoder keeps
            # learning between snapshots, so the buffer describes a mask that forward()
            # is no longer using (review 2026-07-03). Frozen mode: identical tensors.
            current_mask = self.neuromodulator.active_mask().detach().clone()
            masked_logits, masked_value = self.forward_with_mask(obs, current_mask)
            unmasked_mask = torch.ones_like(current_mask)
            unmasked_logits, unmasked_value = self.forward_with_mask(obs, unmasked_mask)

            masked_dist = torch.distributions.Categorical(logits=masked_logits)
            unmasked_dist = torch.distributions.Categorical(logits=unmasked_logits)

            channel_means = current_mask.view(
                current_mask.shape[0],
                self.feature_channels,
                self.h,
                self.w,
            ).mean(dim=(0, 2, 3))

            return {
                'mask_mean': float(current_mask.mean().cpu()),
                'mask_std': float(current_mask.std(unbiased=False).cpu()),
                'mask_min': float(current_mask.min().cpu()),
                'mask_max': float(current_mask.max().cpu()),
                'channel_means': channel_means.cpu(),
                'policy_kl_vs_unmasked': float(torch.distributions.kl_divergence(masked_dist, unmasked_dist).mean().cpu()),
                'entropy_delta_vs_unmasked': float((masked_dist.entropy().mean() - unmasked_dist.entropy().mean()).cpu()),
                'value_delta_abs_vs_unmasked': float((masked_value - unmasked_value).abs().mean().cpu()),
            }

    def forward(self, obs: torch.Tensor):
        # active_mask() returns the frozen snapshot buffer (default) or, when the
        # neuromodulator is trainable, re-decodes the code in-graph so gradient reaches
        # the decoder. Frozen mode is byte-identical to using self.neuro_mask directly.
        return self.forward_with_mask(obs, self.neuromodulator.active_mask())

    def act(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions and return (log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
