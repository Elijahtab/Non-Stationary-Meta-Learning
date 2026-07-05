from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


# Brain action layout:
#   0:7   -> scalar inner-loop levers
#   7:15  -> neuromodulation context code
SCALAR_BRAIN_ACTION_DIM = 7
CONTEXT_CODE_DIM = 8
BRAIN_ACTION_DIM = SCALAR_BRAIN_ACTION_DIM + CONTEXT_CODE_DIM
BRAIN_CONTEXT_SLICE = slice(SCALAR_BRAIN_ACTION_DIM, BRAIN_ACTION_DIM)

CONTEXT_LOG_PREFIX = "brain_context"
NEUROMOD_LOG_PREFIX = "brain_neuromod"


def compose_brain_action(
    scalar_levers: Sequence[float],
    context_code: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Build a full Brain action vector from the scalar levers and optional context code.

    Keeping this layout centralized makes future neuromodulation changes easier to
    test without editing every call site that constructs actions manually.
    """
    if len(scalar_levers) != SCALAR_BRAIN_ACTION_DIM:
        raise ValueError(
            f"Expected {SCALAR_BRAIN_ACTION_DIM} scalar levers, got {len(scalar_levers)}"
        )

    if context_code is None:
        context_code = [0.0] * CONTEXT_CODE_DIM
    elif len(context_code) != CONTEXT_CODE_DIM:
        raise ValueError(
            f"Expected {CONTEXT_CODE_DIM} context values, got {len(context_code)}"
        )

    return np.asarray([*scalar_levers, *context_code], dtype=np.float32)


def log_neuromodulation_snapshot(
    *,
    logger,
    model,
    obs_t: torch.Tensor,
    context_code: torch.Tensor,
    step: int,
) -> None:
    """Record the Brain context code, decoded mask, and its effect on a reference batch."""
    if logger is None or obs_t is None or not hasattr(model, "describe_neuromodulation"):
        return

    for idx, value in enumerate(context_code.detach().cpu().tolist()):
        logger.scalar(f"{CONTEXT_LOG_PREFIX}/context_{idx}", float(value), step)

    summary = model.describe_neuromodulation(obs_t)
    for key in (
        "mask_mean",
        "mask_std",
        "mask_min",
        "mask_max",
        "policy_kl_vs_unmasked",
        "entropy_delta_vs_unmasked",
        "value_delta_abs_vs_unmasked",
    ):
        logger.scalar(f"{NEUROMOD_LOG_PREFIX}/{key}", float(summary[key]), step)

    channel_means = summary.get("channel_means", [])
    for idx, value in enumerate(channel_means):
        logger.scalar(f"{NEUROMOD_LOG_PREFIX}/channel_mean_{idx}", float(value), step)

    # Decoder weight norm: flat over training ⇒ frozen; drifting ⇒ trainable decoder is learning.
    neuromodulator = getattr(model, "neuromodulator", None)
    if neuromodulator is not None and hasattr(neuromodulator, "decoder_weight_norm"):
        logger.scalar(
            f"{NEUROMOD_LOG_PREFIX}/decoder_weight_norm",
            neuromodulator.decoder_weight_norm(),
            step,
        )


class FeatureMaskNeuromodulator(nn.Module):
    """
    Decode a Brain context code into a suppressive feature mask.

    The current implementation preserves existing behavior: a zero context code
    yields an all-ones mask, while larger-magnitude codes increase suppression.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        context_dim: int = CONTEXT_CODE_DIM,
        hidden_dim: int = 256,
        trainable: bool = False,
        gain_alpha: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.trainable = trainable
        # gain_alpha > 0 switches decode_context_code from the paper-faithful suppress-only
        # mask to a two-sided gain mask in [1-alpha, 1+alpha] (autoresearch trials
        # 20260704-133654/001 alpha=1.0, 20260704-203740/002 alpha=0.5; confirmation sweep
        # docs/plans/2026-07-05). 0.0 (default) is byte-identical to the original behavior.
        self.gain_alpha = float(gain_alpha)

        self.decoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.register_buffer("current_mask", torch.ones(1, feature_dim))
        # Raw code retained so the mask can be RE-decoded in-graph when trainable=True
        # (so gradient reaches the decoder). Non-persistent: excluded from state_dict, so
        # frozen behavior and loading of pre-existing checkpoints are unaffected.
        self.register_buffer("current_code", torch.zeros(1, context_dim), persistent=False)

    def decode_context_code(self, code: torch.Tensor) -> torch.Tensor:
        """Map a context code to a mask, keeping zero-context neutral.

        Default (gain_alpha == 0): suppressive mask `1 - s * sigmoid(decoder(code))` in (0,1].
        Gain mode (gain_alpha > 0): two-sided mask `1 + alpha * s * tanh(decoder(code))` in
        [1-alpha, 1+alpha] — the code's direction selects which features are amplified vs
        damped instead of a common-mode downscale. Zero code => identity mask in both modes.
        """
        if code.dim() == 1:
            code = code.unsqueeze(0)

        context_strength = torch.linalg.vector_norm(code, dim=-1, keepdim=True)
        context_strength = context_strength / math.sqrt(self.context_dim)
        context_strength = context_strength.clamp(0.0, 1.0)
        if self.gain_alpha > 0.0:
            gain_template = torch.tanh(self.decoder(code))
            return 1.0 + self.gain_alpha * context_strength * gain_template
        suppression_template = torch.sigmoid(self.decoder(code))
        return 1.0 - context_strength * suppression_template

    def set_context_code(self, code: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            code_2d = code.unsqueeze(0) if code.dim() == 1 else code
            self.current_code = code_2d.detach().to(
                device=self.current_mask.device, dtype=self.current_mask.dtype
            )
            self.current_mask = self.decode_context_code(self.current_code)
        return self.current_mask

    def active_mask(self) -> torch.Tensor:
        """Return the mask consumed by the live forward pass.

        Frozen (default): the detached snapshot buffer, so no gradient reaches the decoder
        and the code->mask converter stays at its random initialization (paper behavior).

        Trainable: re-decode the stored code IN-GRAPH so the inner optimizer trains the
        decoder alongside the encoder/heads. The code itself is detached, so only the
        decoder learns to interpret the Brain's signal — the Brain is unaffected.
        """
        if self.trainable:
            return self.decode_context_code(self.current_code)
        return self.current_mask

    def decoder_weight_norm(self) -> float:
        """L2 norm of all decoder parameters (diagnostic: is the decoder actually moving?)."""
        with torch.no_grad():
            total = sum(float(p.detach().pow(2).sum().cpu()) for p in self.decoder.parameters())
        return math.sqrt(total)

    def clear_context(self) -> None:
        self.current_mask = torch.ones(1, self.feature_dim, device=self.current_mask.device)
        self.current_code = torch.zeros(1, self.context_dim, device=self.current_mask.device)

    def expand_mask(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = self.current_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and features.shape[0] != 1:
            mask = mask.expand(features.shape[0], -1)
        return mask
