import pytest
import torch

from splendor_rl.league.bootstrap import BootstrapConfig, discover_checkpoints
from splendor_rl.models import SharedActor


def checkpoint(path, *, players=2, state=None):
    actor = SharedActor(475, 373, [8])
    torch.save(
        {
            "actor_state_dict": state or actor.state_dict(),
            "num_players": players,
            "global_transition_count": 7,
            "observation_sizes": {"actor": 475, "critic": 475, "action": 373},
            "config": {"hidden_sizes": [8], "num_players": players},
        },
        path,
    )
    return actor.state_dict()


def test_bootstrap_config_loads_nested_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "bootstrap:\n  fixed_bot_games_per_matchup: 2\n  pair_count: 2\n  pool_size: 2\n"
    )
    config = BootstrapConfig.load(path)
    assert config.fixed_bot_games_per_matchup == 2 and config.pair_count == 2


def test_discovery_finds_named_and_numbered_and_deduplicates(tmp_path):
    root = tmp_path / "run" / "checkpoints"
    root.mkdir(parents=True)
    state = checkpoint(root / "best_vs_random.pt")
    checkpoint(root / "latest.pt", state=state)
    checkpoint(root / "step_000000007.pt")
    candidates = discover_checkpoints(root.parent, BootstrapConfig())
    assert {item["source_selection"] for item in candidates} == {
        "best_vs_random",
        "step_000000007",
    }


def test_discovery_skips_missing_optional_and_rejects_player_count(tmp_path):
    root = tmp_path / "run" / "checkpoints"
    root.mkdir(parents=True)
    checkpoint(root / "latest.pt", players=4)
    with pytest.raises(ValueError, match="two-player"):
        discover_checkpoints(root.parent, BootstrapConfig())
