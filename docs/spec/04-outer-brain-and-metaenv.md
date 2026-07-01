# 04 — The Outer Brain & MetaEnv

The **Brain** is a PPO agent whose environment is *the inner Dyna-PPO learner's entire
training run*. It observes a 19-dim summary of inner training and emits a 15-dim action that
tunes the inner learner's plasticity. This is the "double-RL" / meta-learning core of the
repo (paper §3.7).

Source: [`src/lifelong_learning/agents/brain/`](../../src/lifelong_learning/agents/brain/).

## MetaEnv — the inner run as a Gym environment

[`MetaEnv`](../../src/lifelong_learning/agents/brain/meta_env.py#L30) wraps the inner loop:

- **Observation space**: `Box(-10, 10, shape=(19,))`
  ([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py#L115)).
- **Action space**: `Box(-1, 1, shape=(15,))`
  ([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py#L118)).
- **One outer step** = `decision_interval` inner `run_inner_update` calls
  ([`_run_n_updates`](../../src/lifelong_learning/agents/brain/meta_env.py#L271)). `reset`
  also runs `decision_interval` updates to produce a meaningful first observation.
- **One outer episode** = one full inner training run; it `terminated`s when the inner loop
  reaches `total_timesteps` (the inner `done` flag)
  ([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py#L265)).
- Multiple `MetaEnv`s run in parallel (`brain_num_envs`, `sync`/`async`); async uses
  subprocesses so inner runs are genuinely parallel.

Each interval of inner updates is collapsed into one Brain-facing summary by
[`_summarize_interval_stats`](../../src/lifelong_learning/agents/brain/meta_env.py#L281)
(means for loss/surprise signals; aggregated success/failure/return over completed episodes).

## The observation — 19 signals

[`SignalExtractor.extract`](../../src/lifelong_learning/agents/brain/signals.py#L111) builds
the vector from the inner stats. The fixed order is
[`SIGNAL_NAMES`](../../src/lifelong_learning/agents/brain/signals.py#L14):

| # | Signal | # | Signal |
| --- | --- | --- | --- |
| 0 | `mean_episodic_return` | 10 | `current_lr` |
| 1 | `success_rate` | 11 | `current_ent_coef` |
| 2 | `failure_rate` | 12 | `current_intrinsic_coef` |
| 3 | `mean_surprise` | 13 | `current_imagined_horizon` |
| 4 | `surprise_delta` | 14 | `steps_since_surprise_spike` |
| 5 | `wm_loss_state` | 15 | `current_replay_ratio` |
| 6 | `wm_loss_reward` | 16 | `current_replay_prioritization` |
| 7 | `policy_entropy` | 17 | `current_anchoring_weight` |
| 8 | `policy_loss` | 18 | `episodic_memory_fullness` |
| 9 | `value_loss` | | |

Normalization ([signals.py](../../src/lifelong_learning/agents/brain/signals.py#L155-L204)):

- A **Welford running normalizer** ([`RunningNormalizer`](../../src/lifelong_learning/agents/brain/signals.py#L39))
  standardizes all signals and clips to `±10`.
- **Bounded signals are then overwritten** with a static mapping so the Brain keeps absolute
  magnitude awareness: success/failure rates → `[-1,1]`; `lr`, `ent_coef`, `intrinsic_coef` →
  log-scaled to `[-1,1]` within their bounds; `imagined_horizon` linear over `[1,30]`;
  `replay_ratio` over `[0,0.5]`; `replay_prioritization` over `[0,1]`; `anchoring_weight` over
  `[0,0.5]`; `episodic_memory_fullness` over `[0,1]`.
- **Surprise-spike detection** ([`_detect_spike`](../../src/lifelong_learning/agents/brain/signals.py#L208)):
  a z-score over recent surprise history above `surprise_spike_threshold` (default `2.0`) resets
  `steps_since_surprise_spike` to 0 — a soft, learned proxy for "a regime probably just switched."
- The normalizer state persists across Brain episodes for stability; per-episode spike state is
  reset ([signals.py](../../src/lifelong_learning/agents/brain/signals.py#L225-L230)).

## The action — 7 levers + an 8-dim context code

Layout ([neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py#L11-L17)):
indices `0:7` are scalar levers; indices `7:15` are the neuromodulation **context code**.

[`_apply_action`](../../src/lifelong_learning/agents/brain/meta_env.py#L325) maps each action
component **linearly to an absolute hyperparameter value** via `map_to_range(a, (lo, hi))`
(i.e. `a ∈ [-1,1] → lo + (a+1)/2·(hi-lo)`):

| idx | Inner HP set | Bounds (defaults) |
| --- | --- | --- |
| 0 | learning rate | `[min_inner_lr, max_inner_lr]` = `[1e-4, 3e-3]` |
| 1 | entropy coefficient | `[min_ent_coef, max_ent_coef]` = `[1e-3, 0.1]` |
| 2 | intrinsic curiosity coef | `[min_intrinsic_coef, max_intrinsic_coef]` = `[1e-3, 0.5]` |
| 3 | imagined horizon (rounded int) | `[1, 30]` |
| 4 | replay ratio | `[0.0, 0.5]` |
| 5 | replay prioritization | `[0.0, 1.0]` |
| 6 | anchoring weight | `[0.0, 0.5]` |
| 7:15 | neuromodulation context code → `model.set_context_code(...)` | (raw `[-1,1]^8`) |

> **Implementation vs. paper.** The paper (and `docs/plans/PLAN.md`) describe the action as
> "multiplicative scaling." The current code applies an **absolute linear mapping to fixed
> bounds**, not a multiplier on the previous value. Trust the code.

If `disable_neuromodulation=True`, the Brain still emits all 15 dims but the context code is
ignored ([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py#L357)). The
context-code path and its diagnostics are documented in
[05-neuromodulation.md](./05-neuromodulation.md).

## The meta-reward — three modes

Selected by `reward_mode` ([meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py#L235-L259)):

- **`auc`** (default): `success_rate + α·mean_return − β·failure_rate`
  (`α = reward_alpha = 0.1`, `β = reward_beta = 0.5`). Rewards overall area-under-the-curve
  performance.
- **`recovery`**: a shaping reward that explicitly pays for *recovering*:
  `5·max(0, Δsuccess) + 0.3·success − 3·max(0, 0.8−success) − 0.5·failure`. Used by the frozen
  benchmark.
- **`curriculum`**: the `auc` base reward, multiplied **10×** whenever the inner run is on a
  *revisited* regime (tracked via `_regime_exposures`), to up-weight retained competence.

## The Brain agent

[`meta_agent.py`](../../src/lifelong_learning/agents/brain/meta_agent.py) implements the Brain
as a PPO actor-critic: a 2-layer MLP with a **tanh-squashed Gaussian** policy over the 15-dim
continuous action (so actions stay in `[-1,1]`), trained with PPO at `brain_lr` and
`brain_ent_coef`. It is the agent that `scripts/train_brain.py` optimizes.

## Imitation pretraining (warm start)

Before outer-loop PPO, `scripts/train_brain.py` can run `pretrain_episodes` of
**imitation pretraining** ([train_brain.py](../../scripts/train_brain.py#L451-L525)): a
hand-coded heuristic produces target actions (increase exploration-oriented levers after low
success / surprise spikes, taper toward exploitation as the learner recovers), and the Brain is
trained by **MSE** to imitate them. Two heuristics exist:

- `pretrain_mode="basic"` — binary explore/exploit.
- `pretrain_mode="recovery"` — multi-tier, surprise-reactive (the default; also what the
  benchmark uses, though the benchmark sets `pretrain_episodes=0`).

After pretraining, the prefix/counters are restored and normal RL episodes begin
([train_brain.py](../../scripts/train_brain.py#L542-L549)).

## Where training output goes

`scripts/train_brain.py` writes everything under **`runs/<run_name>_<timestamp>/`** —
`config.txt`, the trained `brain_model.pt`, `brain_trends/`, and per-episode inner logs. That
is a **run**. Evaluating a trained Brain (`scripts/eval_brain.py`) writes under **`evals/`**.
The full artifact contract is [06-runs-and-evals.md](./06-runs-and-evals.md).
