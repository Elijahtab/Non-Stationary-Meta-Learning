from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lifelong_learning.agents.brain.neuromod import (
    CONTEXT_CODE_DIM,
    FeatureMaskNeuromodulator,
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

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        self.c, self.h, self.w = obs_shape
        self.feature_channels = 64

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

        # Keep the neuromodulation implementation isolated from the rest of the
        # actor-critic so future research can iterate on it in one place.
        self.neuromodulator = FeatureMaskNeuromodulator(
            feature_dim=flat_size,
            context_dim=CONTEXT_CODE_DIM,
        )

        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
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

    def forward_with_mask(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        features = self.encoder(obs)
        features = features * self._expand_mask(features, mask)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)

    def describe_neuromodulation(self, obs: torch.Tensor) -> dict[str, torch.Tensor | float]:
        """Summarize the current context mask and its effect on a reference batch."""
        with torch.no_grad():
            current_mask = self.neuro_mask.detach().clone()
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
        return self.forward_with_mask(obs, self.neuro_mask)

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
