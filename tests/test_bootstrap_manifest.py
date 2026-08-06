import json

from splendor_rl.league.config import LeagueConfig
from splendor_rl.league.train import train_league


def test_empty_bootstrap_manifest_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"num_players": 2, "policies": []}))
    config = LeagueConfig(
        num_players=2, total_transitions=8, transitions_per_update=8, num_envs=1
    )
    try:
        train_league(config, tmp_path / "run", bootstrap_manifest=path)
    except ValueError as error:
        assert "policies" in str(error)
    else:
        raise AssertionError("empty manifest accepted")
