import json

from splendor_rl.evaluation import evaluate_ladder
from splendor_rl.models import SharedActor


def test_two_player_evaluation_has_five_matchups_and_two_seats(tmp_path):
    summary = evaluate_ladder(
        SharedActor(475, 373, [8]),
        output_dir=tmp_path,
        games_per_matchup=2,
        evaluation_seed_base=200000,
        device="cpu",
        num_players=2,
        save_replays=False,
    )
    assert (
        summary["num_players"] == 2
        and len(summary["matchups"]) == 5
        and "mixed_ladder" not in summary["matchups"]
    )
    assert 1 <= summary["aggregate"]["average_rank"] <= 2
    assert set(summary["aggregate"]["seat_metrics"]) == {"0", "1"}
    stored = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert stored == summary
