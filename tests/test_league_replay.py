from splendor_env.replay import load_recording, verify_recording
from splendor_rl.evaluation import evaluate_ladder
from splendor_rl.models import SharedActor


def test_league_metadata_preserves_native_replay(tmp_path):
    evaluate_ladder(
        SharedActor(475, 373, [8]),
        output_dir=tmp_path,
        games_per_matchup=1,
        evaluation_seed_base=4,
        device="cpu",
        num_players=2,
    )
    document = load_recording(tmp_path / "policy_vs_random" / "game_0000.json")
    document["game_metadata"].update(
        {
            "training_mode": "league_2p",
            "candidate_seat": 0,
            "opponent_id": "champion_0000",
            "acting_policy_id": "candidate",
        }
    )
    verify_recording(document)
