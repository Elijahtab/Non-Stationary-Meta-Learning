from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    num_envs: int = 16
    num_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    seed: int = 0
    device: str = "cuda"
    mode: str = "dyna"
    anchoring_weight: float = 0.0
    # LOOP-0006 hypothesis 3: weight of the auxiliary regime-inference loss (predict the
    # Brain's context code from encoder features). 0.0 (default) = off, byte-identical
    # baseline; requires the model to be built with aux_code_head=True.
    aux_code_coef: float = 0.0


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    minibatches,
    cfg: PPOConfig,
    anchor_logprobs_list: list[torch.Tensor] | None = None,
):
    """
    Performs one round of PPO clipped updates over the given minibatches.
    If anchor_logprobs_list is provided, computes a KL penalty against the anchor logprobs.
    Returns averaged logging scalars.
    """
    total_pg, total_v, total_ent, total_kl, total_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    total_aux_code = 0.0
    n = 0

    for i, (obs, actions, old_logprobs, advantages, returns, old_values) in enumerate(minibatches):
        new_logprobs, entropy, new_values = model.evaluate_actions(obs, actions)
        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)

        # Clipped policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Clipped value loss (prevents large value function updates)
        new_values = new_values.view(-1)
        if cfg.clip_coef > 0:
            v_clipped = old_values + torch.clamp(
                new_values - old_values,
                -cfg.clip_coef,
                cfg.clip_coef,
            )
            v_loss_unclipped = (new_values - returns).pow(2)
            v_loss_clipped = (v_clipped - returns).pow(2)
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * (returns - new_values).pow(2).mean()

        # Entropy bonus (encourages exploration)
        ent_loss = entropy.mean()

        loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

        # Auxiliary regime-inference loss (LOOP-0006 hypothesis 3): press the Brain's
        # context code into the shared representation as a prediction target.
        aux_code_loss = torch.tensor(0.0, device=loss.device)
        if getattr(cfg, "aux_code_coef", 0.0) > 0.0 and hasattr(model, "aux_code_loss"):
            aux_code_loss = model.aux_code_loss(obs)
            loss = loss + cfg.aux_code_coef * aux_code_loss

        # Distillation / Anchoring loss
        kl_penalty = torch.tensor(0.0, device=loss.device)
        if cfg.anchoring_weight > 0.0 and anchor_logprobs_list is not None:
             anchor_logprobs = anchor_logprobs_list[i]
             # KL(P || Q) for discrete where we only have logprobs of the chosen action 
             # is approximated well enough by just encouraging matched logprobs: MSE or absolute difference
             # Alternatively, for continuous or discrete where we don't have full dist: 
             # expected divergence is approx: 0.5 * (new_logprobs - anchor_logprobs)^2
             kl_penalty = 0.5 * (new_logprobs - anchor_logprobs).pow(2).mean()
             loss = loss + cfg.anchoring_weight * kl_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_pg += float(pg_loss.detach().cpu())
        total_v += float(v_loss.detach().cpu())
        total_ent += float(ent_loss.detach().cpu())
        total_kl += float(kl_penalty.detach().cpu()) if cfg.anchoring_weight > 0.0 else 0.0
        total_aux_code += float(aux_code_loss.detach().cpu())
        total_loss += float(loss.detach().cpu())
        n += 1

    stats = {
        "loss/policy": total_pg / max(n, 1),
        "loss/value": total_v / max(n, 1),
        "loss/entropy": total_ent / max(n, 1),
        "loss/total": total_loss / max(n, 1),
    }
    if cfg.anchoring_weight > 0.0:
        stats["loss/kl_anchor"] = total_kl / max(n, 1)
    if getattr(cfg, "aux_code_coef", 0.0) > 0.0:
        stats["loss/aux_code"] = total_aux_code / max(n, 1)

    return stats
