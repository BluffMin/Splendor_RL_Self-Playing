import json

from league_helpers import models

from splendor_rl.league.matrix import build_matchup_matrix
from splendor_rl.league.train import _frozen_actor


def test_matrix_has_half_diagonal_and_outputs(tmp_path):
    actor, _ = models()
    policies = {
        "a": _frozen_actor(actor, "a", 0, "cpu"),
        "b": _frozen_actor(actor, "b", 0, "cpu"),
    }
    result = build_matchup_matrix(
        policies, games_per_pair=2, seed_base=1, output_dir=tmp_path
    )
    assert result["matrix"][0][0] == result["matrix"][1][1] == 0.5
    assert result["games_per_pair"] == 2
    assert (tmp_path / "matchup_matrix.csv").exists()
    assert json.loads((tmp_path / "matchup_matrix.json").read_text())["policy_ids"] == [
        "a",
        "b",
    ]
