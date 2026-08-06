import torch

from splendor_rl.checkpoint import load_checkpoint, save_checkpoint
from splendor_rl.config import PPOConfig
from splendor_rl.models import PrivilegedCritic, SharedActor


def test_v041_recovers_players_from_config(tmp_path):
    actor = SharedActor(4, 3, [8])
    critic = PrivilegedCritic(5, [8])
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    sizes = {"actor": 4, "critic": 5, "action": 3}
    path = tmp_path / "old.pt"
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
    data["schema_version"] = "0.4.1"
    data.pop("num_players")
    torch.save(data, path)
    assert (
        load_checkpoint(path, actor, expected_sizes=sizes, restore_rng=False)[
            "num_players"
        ]
        == 2
    )
