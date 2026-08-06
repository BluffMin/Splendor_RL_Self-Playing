from splendor_rl.config import PPOConfig
from splendor_rl.train import train


def test_two_update_smoke_creates_finite_checkpoint(tmp_path):
    config = PPOConfig(
        num_envs=2,
        hidden_sizes=[16],
        transitions_per_update=16,
        minibatch_size=8,
        update_epochs=1,
        total_transitions=32,
        checkpoint_interval=16,
        evaluation_interval=16,
        evaluate_initial_policy=True,
        evaluation_games_per_matchup=1,
        max_turns=10,
    )
    _, _, collector = train(config, tmp_path)
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / "step_000000000.pt").exists()
    assert (tmp_path / "checkpoints" / "best_average_rank.pt").exists()
    assert (tmp_path / "evaluations" / "step_000000000" / "summary.json").exists()
    assert (tmp_path / "evaluations" / "step_000000016" / "summary.json").exists()
    assert collector.illegal_actions == 0 and collector.invariant_violations == 0
