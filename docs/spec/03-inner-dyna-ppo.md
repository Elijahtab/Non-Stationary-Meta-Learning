# 03 — Inner Dyna-PPO Continual Learner

The inner agent is the thing that actually plays MiniGrid and that *forgets*. It is a PPO
actor-critic augmented Dyna-style with a learned world model, intrinsic curiosity, imagined
("dream") rollouts, episodic replay, and policy anchoring. The outer Brain controls its
plasticity (see [04-outer-brain-and-metaenv.md](./04-outer-brain-and-metaenv.md)); this doc
is about the learner itself.

Source: [`src/lifelong_learning/agents/ppo/`](../../src/lifelong_learning/agents/ppo/).

## Components

### Actor-critic network — [`CNNActorCritic`](../../src/lifelong_learning/agents/ppo/network.py#L13)

- **Shared encoder**: `Conv(C→32) → Conv(32→64) → Conv(64→64)` (all `3×3`, `padding=1`, ReLU)
  then `Flatten` ([network.py](../../src/lifelong_learning/agents/ppo/network.py#L32-L40)).
  For `(21,8,8)` input the flattened feature size is `64·8·8 = 4096`.
- **Neuromodulation gate**: a [`FeatureMaskNeuromodulator`](../../src/lifelong_learning/agents/brain/neuromod.py#L80)
  multiplies the flattened features by a mask before the heads — see
  [05-neuromodulation.md](./05-neuromodulation.md). Both heads consume the gated features.
- **Heads**: `actor_head = Linear(flat,256)→ReLU→Linear(256,n_actions)`;
  `critic_head = Linear(flat,256)→ReLU→Linear(256,1)`
  ([network.py](../../src/lifelong_learning/agents/ppo/network.py#L55-L66)).
- **Init**: orthogonal; actor output gain `0.01` (near-uniform initial policy), critic output
  gain `1.0` ([network.py](../../src/lifelong_learning/agents/ppo/network.py#L71-L92)).
- Discrete `Categorical` policy. `act` / `evaluate_actions` return `(logprob, entropy, value)`.

### World model — [`SimpleWorldModel`](../../src/lifelong_learning/agents/ppo/world_model.py#L9)

A lightweight feed-forward model predicting next-state and reward from `(state, action)`:

- Same 3-layer CNN encoder as the actor-critic, plus an `Embedding(n_actions, 32)` fused
  late (concatenated after the CNN) → `MLP(cnn_out+32 → 256 → 256)`
  ([world_model.py](../../src/lifelong_learning/agents/ppo/world_model.py#L31-L56)).
- **Heads**: `next_state_head` produces `C·H·W` logits reshaped to `(C,H,W)` (trained with
  cross-entropy against the true next one-hot); `reward_head` produces a scalar (trained with
  MSE) ([world_model.py](../../src/lifelong_learning/agents/ppo/world_model.py#L58-L91)).
- [`discretize_state`](../../src/lifelong_learning/agents/ppo/world_model.py#L93): argmax over
  channels → re-one-hot, to stop "blurry dreams" from compounding.
- [`generate_imagined_trajectories`](../../src/lifelong_learning/agents/ppo/world_model.py#L110):
  rolls the policy forward through the WM for `horizon` steps under `no_grad`, discretizing each
  step. Dones are forced to `0` (fixed horizon — a learned terminal predictor is unreliable
  early).

### Rollout buffer & episodic memory

- [`RolloutBuffer`](../../src/lifelong_learning/agents/ppo/buffers.py) stores the on-policy
  rollout and computes GAE returns/advantages. It tracks both the **total** reward
  (extrinsic + intrinsic) used for PPO and the **extrinsic-only** reward used for replay
  archival.
- [`EpisodicMemory`](../../src/lifelong_learning/agents/ppo/episodic_memory.py) is a
  capacity-bounded rehearsal store. `store_from_rollout` archives transitions tagged with the
  current `regime_id` (using extrinsic rewards), and `sample(..., prioritization, current_regime)`
  biases sampling toward **older** regimes when `prioritization > 0`.

### PPO update math — [`ppo_update`](../../src/lifelong_learning/agents/ppo/ppo.py#L31)

Standard clipped PPO: clipped policy-gradient loss + clipped value loss + entropy bonus,
`loss = pg_loss - ent_coef·entropy + vf_coef·v_loss`, gradient-clipped to `max_grad_norm`
([ppo.py](../../src/lifelong_learning/agents/ppo/ppo.py#L51-L88)). When
`anchoring_weight > 0`, it adds an **anchoring penalty** `0.5·(new_logprob - anchor_logprob)²`
that pulls the policy toward the snapshot taken at the last regime switch
([ppo.py](../../src/lifelong_learning/agents/ppo/ppo.py#L75-L84)).

### Default config — [`PPOConfig`](../../src/lifelong_learning/agents/ppo/ppo.py#L8)

| Field | Default | | Field | Default |
| --- | --- | --- | --- | --- |
| `total_timesteps` | `500_000` | | `gamma` | `0.99` |
| `num_envs` | `16` | | `gae_lambda` | `0.95` |
| `num_steps` | `128` | | `clip_coef` | `0.2` |
| `update_epochs` | `4` | | `ent_coef` | `0.01` |
| `minibatch_size` | `256` | | `vf_coef` | `0.5` |
| `lr` | `3e-4` | | `max_grad_norm` | `0.5` |
| `mode` | `"dyna"` | | `anchoring_weight` | `0.0` |

> Note: when launched through `MetaEnv`, several of these are overwritten online by the Brain
> (see below), and `num_envs`/`num_steps`/`total_timesteps` come from the training CLI, not
> the dataclass defaults.

## The update cycle — [`run_inner_update`](../../src/lifelong_learning/agents/ppo/train.py#L321)

This is the atomic unit the Brain steps. It mutates an
[`InnerTrainState`](../../src/lifelong_learning/agents/ppo/train.py#L76) in place and returns
a stats dict. One call does:

1. **LR anneal** (if `anneal_lr`) — note `MetaEnv` runs with `anneal_lr=False` so the Brain
   owns the LR ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L346-L351)).
2. **Phase A — collect real experience** (`num_steps` steps across `num_envs`):
   - WM predicts next state/reward; **surprise** = `CE(next-state) + MSE(reward)`
     ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L408-L425)).
   - **intrinsic reward** = `clip(surprise · intrinsic_coef, 0, intrinsic_reward_clip)`;
     stored reward = `extrinsic + intrinsic` (in `"passive"` mode surprise is zeroed).
   - Detects **regime switches** and snapshots the `anchor_model`
     ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L390-L395)).
   - Logs `charts/success_rate`, `charts/regime_id`, `charts/episodic_return`, etc.
3. **GAE** on the collected rollout.
4. **Phase B — PPO on real data**: `update_epochs × minibatches`, with the anchoring penalty
   if enabled.
5. **Phase B2 — replay** (only if `replay_ratio > 0.01` and memory is warm): build a mixed
   batch of fresh + episodic-memory transitions (prioritized toward older regimes), compute
   1-step returns, and run one PPO update
   ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L525-L575)).
6. **Phase C — train the world model**: CE(state) + MSE(reward) for `update_epochs`.
7. **Archive** the rollout into episodic memory (extrinsic rewards only).
8. **Phase D — dream**: sample seed states from the buffer, roll the WM `imagined_horizon`
   steps, GAE, and run **one** PPO epoch on the imagined data (gradients do not flow into the
   WM) ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L609-L656)).
9. **Log + checkpoint** (`save_every_updates`) and return the stats dict.

> The paper describes this as a three-phase Dyna loop (real / WM-train / dream). The code adds
> two more steps that the Brain controls: **replay (B2)** and **anchoring** (folded into B/B2/D).

## What the Brain controls

[`InnerTrainState`](../../src/lifelong_learning/agents/ppo/train.py#L76) exposes the mutable
levers the Brain writes each decision (via
[`MetaEnv._apply_action`](../../src/lifelong_learning/agents/brain/meta_env.py#L325)):

| Lever | Field written | Used in |
| --- | --- | --- |
| learning rate | `optimizer.param_groups[0]["lr"]` | all PPO updates |
| entropy coefficient | `cfg.ent_coef` | PPO loss |
| intrinsic curiosity coef | `intrinsic_coef` | Phase A reward |
| imagined horizon | `imagined_horizon` | Phase D dreaming |
| replay ratio | `replay_ratio` | Phase B2 |
| replay prioritization | `replay_prioritization` | Phase B2 sampling |
| anchoring weight | `cfg.anchoring_weight` | anchoring penalty |
| neuromodulation context code | `model.set_context_code(...)` | feature gating (docs 05) |

## Stats returned to the Brain

`run_inner_update` returns the dict that
[`SignalExtractor`](../../src/lifelong_learning/agents/brain/signals.py#L90) turns into the
19-dim observation: `mean_episodic_return`, `success_rate`, `failure_rate`, `mean_surprise`,
`wm_loss_state`, `wm_loss_reward`, `policy_entropy`, `policy_loss`, `value_loss`, the current
values of every controllable HP, `episodic_memory_fullness`, `current_mode_regime`, and `done`
([train.py](../../src/lifelong_learning/agents/ppo/train.py#L754-L793)). See
[04-outer-brain-and-metaenv.md](./04-outer-brain-and-metaenv.md) for how these become the
observation and reward.
