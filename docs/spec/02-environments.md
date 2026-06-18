# 02 — Environments

The environment is a non-stationary MiniGrid task: the agent must reach a goal whose
"correct" identity **flips periodically**, with no observable cue that the flip happened.
This is what makes catastrophic forgetting measurable.

Source: [`src/lifelong_learning/envs/`](../../src/lifelong_learning/envs/).

## The task

Two base environments are registered in
[envs/`__init__.py`](../../src/lifelong_learning/envs/__init__.py#L7-L29):

| Env id | Class | Notes |
| --- | --- | --- |
| `MiniGrid-MultiGoal-8x8-v0` | [`MultiGoalEnv`](../../src/lifelong_learning/envs/multi_goal.py#L9) | up to 6 colored goals; goal count = `num_regimes` |
| `MiniGrid-MultiGoal-5x5-v0` | `MultiGoalEnv` | smaller grid used for fast benchmarks |
| `MiniGrid-DualGoal-8x8-v0` | [`DualGoalEnv`](../../src/lifelong_learning/envs/dual_goal.py) | fixed two-goal variant |
| `MiniGrid-DualGoal-5x5-v0` | `DualGoalEnv` | |

[`MultiGoalEnv`](../../src/lifelong_learning/envs/multi_goal.py) places one `Goal` per color
(`green, blue, purple, red, yellow, grey`, in that order) and the agent at a random free
cell. On `step`, it checks **all** goal positions (MiniGrid's base env only tracks one) and,
on a goal hit, terminates and reports `info["hit_goal_index"]`
([multi_goal.py](../../src/lifelong_learning/envs/multi_goal.py#L65-L80)). The base reward
emitted here is just a positive trigger; the *real* reward contingency is imposed by the
regime wrapper below.

## The wrapper stack

[`make_env`](../../src/lifelong_learning/envs/make_env.py#L10) builds the full stack
(inner → outer). For `MultiGoal` envs it sets `num_goals = num_regimes`.

```text
gym.make(env_id, max_episode_steps=256, [num_goals=num_regimes])
  └─ FullyObsWrapper            full grid view (not the 7×7 partial view)
       └─ ActionReduceWrapper   Discrete(7) → Discrete(3): [left, right, forward]
            └─ OneHotPartialObsWrapper   (H,W,3) symbolic → (21, H, W) one-hot float
                 └─ RegimeGoalSwapWrapper   imposes the non-stationary reward
```

- **Observation**: `(21, H, W)` one-hot float tensor —
  [one_hot.py](../../src/lifelong_learning/envs/wrappers/one_hot.py#L5). The 21 channels are
  `11 object types + 6 colors + 4 states/direction`. For 8×8 this is `(21, 8, 8)`.
- **Actions**: `Discrete(3)` — turn left, turn right, move forward
  ([action_reduce.py](../../src/lifelong_learning/envs/wrappers/action_reduce.py)).
- **Episode cap**: `max_episode_steps=256`
  ([make_env.py](../../src/lifelong_learning/envs/make_env.py#L26)).

## Regime switching (the non-stationarity)

[`RegimeGoalSwapWrapper`](../../src/lifelong_learning/envs/regime_wrapper.py#L7) is where
forgetting comes from. It defines the reward contingency by the current `regime_id`:

- **Regime `r`** ⇒ goal index `r` is **correct** (`+5`), every other goal is **wrong** (`-1`).
- Every step also incurs a **`-0.01`** time penalty
  ([regime_wrapper.py](../../src/lifelong_learning/envs/regime_wrapper.py#L69-L97)).

```text
reached correct goal (hit_goal_index == regime_id):  reward = -0.01 + 5.0   → reached_good_goal=1
reached wrong goal  (hit_goal_index != regime_id):   reward = -0.01 - 1.0   → reached_bad_goal=1
timed out (truncated):                               reward = -0.01         → timed_out=1
otherwise (mid-episode step):                        reward = -0.01
```

The `regime_id` advances **deterministically** by a global step (or episode) counter
([`_update_regime_deterministic`](../../src/lifelong_learning/envs/regime_wrapper.py#L38)):

```text
regime_id = (start_regime + (step_count // steps_per_regime)) % num_regimes
# or, if episodes_per_regime is set instead:
regime_id = (start_regime + (cumulative_episodes // episodes_per_regime)) % num_regimes
```

Key properties:

- **Shared step counter.** All parallel vector-env workers share one `shared_step_counter`
  list so they switch regime *simultaneously*
  ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L186-L199),
  [regime_wrapper.py](../../src/lifelong_learning/envs/regime_wrapper.py#L40)).
- **The regime is hidden.** Nothing in the observation changes when the regime flips — only
  the reward function does. The agent can only *infer* a switch from unexpected rewards
  (this is the deliberate "no task signal" design in the paper, §1).
- **`regime_id` is exposed for logging only**, via `info["regime_id"]`. The inner loop logs
  it to `charts/regime_id` ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L388)),
  and the scorer uses that trace to locate switches (see
  [07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md)).

## Info dict contract

Each `step` populates `info` with the keys the inner loop and scorer rely on:

| Key | Meaning |
| --- | --- |
| `regime_id` | active regime this step (for logging / switch detection) |
| `reached_good_goal` | `1.0` if the correct goal was reached, else `0.0` |
| `reached_bad_goal` | `1.0` if a wrong goal was reached, else `0.0` |
| `timed_out` | `1.0` if the episode truncated at the step cap |
| `hit_goal_index` | (from `MultiGoalEnv`) which goal color index was reached |

These drive the inner loop's `success_rate` / `failure_rate` / `timeout_rate` outcome
windows ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L465-L492)).

## What is configurable

`num_regimes`, `start_regime`, `randomize_start_regime`, and `steps_per_regime`
(or `episodes_per_regime`) are surfaced all the way up through `MetaEnv` and the training
CLIs. The benchmark **freezes** these (e.g. `num_regimes=2`,
`inner_steps_per_regime=100_000`) — see
[07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md).
