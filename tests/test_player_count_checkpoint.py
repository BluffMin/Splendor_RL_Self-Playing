import pytest
import torch

from splendor_rl.checkpoint import load_checkpoint, save_checkpoint
from splendor_rl.config import PPOConfig
from splendor_rl.evaluate import resolve_checkpoint_num_players
from splendor_rl.models import PrivilegedCritic, SharedActor


def test_checkpoint_player_count_metadata_and_corruption(tmp_path):
    actor = SharedActor(4, 3, [8])
    critic = PrivilegedCritic(5, [8])
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    sizes = {"actor": 4, "critic": 5, "action": 3}
    path = tmp_path / "two.pt"
    save_checkpoint(
        path,
        actor,
        critic,
        opt,
        PPOConfig(num_players=2, hidden_sizes=[8]),
        0,
        0,
        sizes,
    )
    data = torch.load(path, weights_only=False)
    assert data["num_players"] == 2 and data["max_players_in_observation"] == 4
    data["num_players"] = 4
    torch.save(data, path)
    with pytest.raises(ValueError, match="inconsistent"):
        load_checkpoint(path, actor, expected_sizes=sizes, restore_rng=False)
    with pytest.raises(ValueError, match="trained with num_players=2"):
        resolve_checkpoint_num_players(
            {"num_players": 2, "config": {"num_players": 2}}, 4
        )
