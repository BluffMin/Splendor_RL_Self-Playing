import warnings

import torch

from splendor_rl.checkpoint import load_checkpoint, save_checkpoint
from splendor_rl.config import PPOConfig
from splendor_rl.models import PrivilegedCritic, SharedActor


def test_v040_defaults_and_outputs(tmp_path):
    actor = SharedActor(4, 3, [8])
    critic = PrivilegedCritic(5, [8])
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    sizes = {"actor": 4, "critic": 5, "action": 3}
    path = tmp_path / "old.pt"
    save_checkpoint(path, actor, critic, opt, PPOConfig(hidden_sizes=[8]), 5, 1, sizes)
    data = torch.load(path, weights_only=False)
    data["schema_version"] = "0.4.0"
    data.pop("schedule_state")
    data.pop("best_metrics")
    torch.save(data, path)
    restored = SharedActor(4, 3, [8])
    with warnings.catch_warnings(record=True) as caught:
        loaded = load_checkpoint(
            path, restored, expected_sizes=sizes, restore_rng=False
        )
    assert (
        caught
        and loaded["best_metrics"] == {}
        and loaded["critic_information_scope"]["full_deck_order"] is False
    )
