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
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim

        self.decoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.register_buffer("current_mask", torch.ones(1, feature_dim))

    def decode_context_code(self, code: torch.Tensor) -> torch.Tensor:
        """Map a context code to a suppressive mask, keeping zero-context neutral."""
        if code.dim() == 1:
            code = code.unsqueeze(0)

        suppression_template = torch.sigmoid(self.decoder(code))
        context_strength = torch.linalg.vector_norm(code, dim=-1, keepdim=True)
        context_strength = context_strength / math.sqrt(self.context_dim)
        context_strength = context_strength.clamp(0.0, 1.0)
        return 1.0 - context_strength * suppression_template

    def set_context_code(self, code: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.current_mask = self.decode_context_code(code)
        return self.current_mask

    def clear_context(self) -> None:
        self.current_mask = torch.ones(1, self.feature_dim, device=self.current_mask.device)

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
