import json

from splendor_rl.evaluation import evaluate_ladder
from splendor_rl.models import SharedActor


def test_evaluation_seed_schema_and_training_mode_restore(tmp_path):
    actor = SharedActor(475, 373, [8])
    actor.train()
    summary = evaluate_ladder(
        actor, output_dir=tmp_path, games_per_matchup=1,
        evaluation_seed_base=123, device="cpu", num_players=4, save_replays=False
    )
    stored = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert (
        actor.training and summary == stored and summary["evaluation_seed_base"] == 123
    )
    assert set(summary["matchups"]) >= {
        "policy_vs_random",
        "policy_vs_greedy",
        "mixed_ladder",
    }
