from splendor_env.replay import load_recording, verify_recording
from splendor_rl.evaluation import evaluate_ladder
from splendor_rl.models import SharedActor


def test_two_player_replay_contains_only_real_players(tmp_path):
    evaluate_ladder(
        SharedActor(475, 373, [8]),
        output_dir=tmp_path,
        games_per_matchup=1,
        evaluation_seed_base=210000,
        device="cpu",
        num_players=2,
        save_replays=True,
    )
    path = tmp_path / "policy_vs_random" / "game_0000.json"
    doc = load_recording(path)
    game = verify_recording(doc)
    assert game.num_players == 2 and len(doc["final_summary"]["players"]) == 2
    turns = doc["turns"][:4]
    assert [t["acting_player"] for t in turns] == [0, 1, 0, 1]
