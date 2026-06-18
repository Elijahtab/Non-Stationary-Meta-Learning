import json
from pathlib import Path

import scripts.run_frozen_benchmark as run_frozen_benchmark


def test_generate_benchmark_plots_writes_inner_env_high_scale_artifacts(tmp_path):
    inner_dir = tmp_path / "episode_1" / "ep1_env0_0001"
    inner_dir.mkdir(parents=True)

    data = {
        "charts/regime_id": [[0, 0], [100, 0], [200, 1], [300, 1]],
        "charts/success_rate": [[100, 0.3], [200, 0.82], [300, 0.9]],
        "charts/episodic_return": [[100, 0.0], [200, 1.0], [300, 2.0]],
        "charts/reward_step_mean": [[100, 0.1], [200, 0.2], [300, 0.3]],
    }
    for dim in range(8):
        data[f"brain_context/context_{dim}"] = [[100, 0.1 * dim], [200, 0.1 * dim]]
    for channel in range(64):
        data[f"brain_neuromod/channel_mean_{channel}"] = [[100, 0.0], [200, 0.2]]
    data["brain_neuromod/policy_kl_vs_unmasked"] = [[100, 0.01], [200, 0.02]]
    data["brain_neuromod/value_delta_abs_vs_unmasked"] = [[100, 0.03], [200, 0.04]]
    data["brain_neuromod/entropy_delta_vs_unmasked"] = [[100, -0.01], [200, -0.02]]

    with open(inner_dir / "demo_data.json", "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    summary = run_frozen_benchmark.generate_benchmark_plots(
        tmp_path,
        interval=100,
        smoothing=0.1,
    )

    assert summary["folder_count"] == 1
    assert summary["errors"] == []
    assert summary["folders"][0]["png_count"] >= 2
    assert any(path.endswith("_neuromodulation_dashboard.png") for path in summary["folders"][0]["png_paths"])


def test_derive_high_scale_interval_tracks_regime_length():
    spec = run_frozen_benchmark.get_frozen_benchmark("fast_switch_scout_v1")

    assert run_frozen_benchmark.derive_high_scale_interval(spec) == 50_000
