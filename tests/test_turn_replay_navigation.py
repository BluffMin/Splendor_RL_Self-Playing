from splendor_env.core import SplendorGame
from splendor_env.event_schema import build_turn_records, summarize_turn
from splendor_env.recording import EpisodeRecorder


def test_turn_title_is_primary_action(tmp_path):
    game = SplendorGame(2, seed=3)
    rec = EpisodeRecorder(tmp_path / "x.json")
    rec.attach(game)
    while game.turns_completed < 8:
        game.step(game.legal_actions()[0])
    doc = rec.finalize()
    turns = build_turn_records(doc["events"])
    assert all(
        "payment plan" not in summarize_turn(t).splitlines()[1].lower() for t in turns
    )
