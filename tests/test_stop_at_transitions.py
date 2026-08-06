import json

import pytest

from splendor_rl.config import PPOConfig
from splendor_rl.progress import ProgressConfig, ProgressMode
from splendor_rl.schedules import entropy_coefficient, linear_learning_rate
from splendor_rl.train import train


def tiny_config():
    return PPOConfig(
        seed=9,
        num_players=2,
        num_envs=1,
        hidden_sizes=[8],
        transitions_per_update=8,
        update_epochs=1,
        minibatch_size=8,
        total_transitions=32,
        checkpoint_interval=32,
        evaluation_interval=32,
        evaluate_initial_policy=False,
        evaluation_games_per_matchup=1,
        max_turns=20,
    )


def test_stop_target_preserves_schedule_and_finishes_orchestration(tmp_path):
    config = tiny_config()
    train(
        config,
        tmp_path,
        stop_at_transitions=8,
        progress_config=ProgressConfig(ProgressMode.NEVER),
    )
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["total_transitions"] == 8
    assert summary["final_learning_rate"] == pytest.approx(
        linear_learning_rate(3e-4, 0, 8, 32, True)
    )
    assert summary["final_entropy_coefficient"] == pytest.approx(
        entropy_coefficient(0.01, 0.001, 0.8, 8, 32)
    )
    assert (tmp_path / "checkpoints" / "step_000000008.pt").exists()
    assert (tmp_path / "evaluations" / "step_000000008" / "summary.json").exists()


def test_resume_at_same_or_later_stop_target_is_rejected(tmp_path):
    config = tiny_config()
    train(
        config,
        tmp_path,
        stop_at_transitions=8,
        progress_config=ProgressConfig(ProgressMode.NEVER),
    )
    checkpoint = tmp_path / "checkpoints" / "step_000000008.pt"
    with pytest.raises(ValueError, match="not below stop target"):
        train(config, tmp_path, checkpoint, stop_at_transitions=8)
