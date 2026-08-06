from __future__ import annotations

import json

from splendor_env.actions import action_id
from splendor_env.core import SplendorGame
from splendor_env.recording import EpisodeRecorder
from splendor_env.visualization.html_export import export_game_view, export_replay


def recorded_private_reservation(tmp_path):
    source = tmp_path / "game.json"
    game = SplendorGame(2, seed=10)
    recorder = EpisodeRecorder(source, "full")
    recorder.attach(game)
    game.step(action_id("reserve_deck", 1))
    hidden_id = game.players[0].reserved[0].card.card_id
    recorder.finalize()
    return source, game, hidden_id


def test_self_contained_html_and_controls(tmp_path) -> None:
    source, game, _ = recorded_private_reservation(tmp_path)
    output = export_replay(source, tmp_path / "viewer.html")
    text = output.read_text(encoding="utf-8")
    assert "id=\"perspective\"" in text
    assert "id=\"eventSlider\"" in text
    assert "Previous turn" in text and "Next decision" in text
    assert "http://" not in text and "https://" not in text
    live = export_game_view(game, tmp_path / "live.html", perspective=1)
    assert live.exists()


def test_perspective_sanitized_export_contains_no_private_id(tmp_path) -> None:
    source, _, hidden_id = recorded_private_reservation(tmp_path)
    output = export_replay(
        source,
        tmp_path / "safe.html",
        data_mode="perspective-sanitized-data",
        perspective=1,
    )
    text = output.read_text(encoding="utf-8")
    assert hidden_id not in text
    assert "Hidden reservation" in text  # Bundled renderer supports the masked card face.


def test_script_breakout_text_is_escaped(tmp_path) -> None:
    source, _, _ = recorded_private_reservation(tmp_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    document["events"][0]["action_text"] = "</script><script>alert(1)</script>"
    source.write_text(json.dumps(document), encoding="utf-8")
    text = export_replay(source, tmp_path / "escaped.html").read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in text
    assert "\\u003c/script\\u003e" in text
