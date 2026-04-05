import json

from lifelong_learning.research.benchmarking import (
    detect_regime_switch_steps,
    find_first_sustained_threshold_step,
    get_frozen_benchmark,
    parse_run_config,
    score_brain_run,
    summarize_threshold_recovery,
)


def test_detect_regime_switch_steps_finds_value_changes():
    regime_points = [
        [0, 0.0],
        [10, 0.0],
        [20, 1.0],
        [30, 1.0],
        [40, 0.0],
    ]

    assert detect_regime_switch_steps(regime_points) == [20, 40]


def test_get_frozen_benchmark_includes_pilot_spec():
    spec = get_frozen_benchmark("fast_switch_pilot_v1")

    assert spec.fixed_train_args["inner_total_timesteps"] == 12_000
    assert spec.fixed_train_args["brain_episodes"] == 1
    assert spec.sustained_points_required == 2


def test_get_frozen_benchmark_uses_async_outer_vectorization_for_full_runs():
    scout = get_frozen_benchmark("fast_switch_scout_v1")
    holdout = get_frozen_benchmark("fast_switch_holdout_v1")

    assert scout.fixed_train_args["brain_num_envs"] == 4
    assert scout.fixed_train_args["brain_vectorization"] == "async"
    assert holdout.fixed_train_args["brain_num_envs"] == 4
    assert holdout.fixed_train_args["brain_vectorization"] == "async"


def test_find_first_sustained_threshold_step_ignores_single_spike():
    success_points = [
        [105, 0.81],
        [110, 0.72],
        [115, 0.83],
        [120, 0.84],
        [125, 0.85],
    ]

    sustained_step = find_first_sustained_threshold_step(
        success_points,
        threshold=0.80,
        start_step=100,
        end_step=200,
        sustained_points_required=3,
    )

    assert sustained_step == 115


def test_summarize_threshold_recovery_returns_median_hit_rate():
    regime_points = [
        [0, 0.0],
        [10, 0.0],
        [20, 1.0],
        [30, 1.0],
        [40, 0.0],
        [50, 0.0],
    ]
    success_points = [
        [22, 0.81],
        [24, 0.82],
        [26, 0.84],
        [42, 0.91],
        [44, 0.92],
        [46, 0.93],
    ]

    summary = summarize_threshold_recovery(
        success_points,
        regime_points,
        threshold=0.80,
        sustained_points_required=3,
    )

    assert summary.switch_count == 2
    assert summary.hit_count == 2
    assert summary.hit_rate == 1.0
    assert summary.per_switch_steps == [2, 2]
    assert summary.median_steps == 2.0


def test_parse_run_config_reads_scalars(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                "Brain Training Configuration:",
                "----------------------------------------",
                "brain_episodes: 4",
                "disable_neuromodulation: False",
                "reward_mode: recovery",
                "inner_steps_per_regime: 100000",
            ]
        ),
        encoding="utf-8",
    )

    config = parse_run_config(tmp_path)

    assert config["brain_episodes"] == 4
    assert config["disable_neuromodulation"] is False
    assert config["reward_mode"] == "recovery"
    assert config["inner_steps_per_regime"] == 100000


def test_score_brain_run_aggregates_recovery_and_neuromodulation(tmp_path):
    run_dir = tmp_path / "demo_run"
    brain_trends = run_dir / "brain_trends"
    inner_dir = run_dir / "episode_1" / "ep1_env0_0001"
    brain_trends.mkdir(parents=True)
    inner_dir.mkdir(parents=True)

    (run_dir / "config.txt").write_text(
        "\n".join(
            [
                "Brain Training Configuration:",
                "----------------------------------------",
                "brain_episodes: 4",
                "inner_steps_per_regime: 100000",
                "reward_mode: recovery",
            ]
        ),
        encoding="utf-8",
    )

    with open(brain_trends / "demo_data.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "brain/episode_avg_success_rate": [[1, 0.45], [2, 0.55], [3, 0.65], [4, 0.75]]
            },
            handle,
        )

    with open(inner_dir / "ep1_env0_data.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "charts/regime_id": [
                    [0, 0.0],
                    [100000, 1.0],
                    [200000, 0.0],
                ],
                "charts/success_rate": [
                    [100020, 0.81],
                    [100040, 0.82],
                    [100060, 0.83],
                    [100100, 0.95],
                    [100120, 0.96],
                    [100140, 0.97],
                    [200020, 0.80],
                    [200040, 0.81],
                    [200060, 0.82],
                    [200100, 0.95],
                    [200120, 0.96],
                    [200140, 0.97],
                ],
                "brain_neuromod/policy_kl_vs_unmasked": [
                    [100010, 0.01],
                    [150000, 0.02],
                    [200010, 0.03],
                ],
                "brain_neuromod/value_delta_abs_vs_unmasked": [
                    [100010, 0.10],
                    [150000, 0.20],
                    [200010, 0.30],
                ],
            },
            handle,
        )

    score = score_brain_run(run_dir)

    assert score.mean_episode_avg_success_rate == 0.6
    assert score.median_steps_to_80 == 20.0
    assert score.median_steps_to_95 == 100.0
    assert score.hit_rate_80 == 1.0
    assert score.hit_rate_95 == 1.0
    assert score.mean_post_switch_policy_kl is not None
    assert score.mean_post_switch_value_delta_abs is not None
    assert score.mean_post_switch_neuromod_activity is not None
    assert score.composite_score is not None
