import sys
from pathlib import Path

import scripts.train_ppo as train_ppo_script
from lifelong_learning.agents.ppo.train import resolve_inner_save_dir


def test_train_ppo_forwards_replay_flags(monkeypatch):
    captured = {}

    def fake_train_ppo(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train_ppo_script, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--replay_ratio",
            "0.25",
            "--replay_prioritization",
            "0.75",
            "--run_name",
            "cli_test",
        ],
    )

    train_ppo_script.main()

    assert captured["replay_ratio"] == 0.25
    assert captured["replay_prioritization"] == 0.75


def test_resolve_inner_save_dir_rehomes_episode_level_checkpoint_dirs():
    save_dir = "runs/demo_run/episode_3/inner_checkpoints"
    log_dir = "runs/demo_run/episode_3"
    logger_full_dir = "runs/demo_run/episode_3/ep3_env1_20260405-120000"

    resolved = resolve_inner_save_dir(
        save_dir=save_dir,
        log_dir=log_dir,
        logger_full_dir=logger_full_dir,
    )

    assert Path(resolved) == Path(
        "runs/demo_run/episode_3/ep3_env1_20260405-120000/inner_checkpoints"
    )
