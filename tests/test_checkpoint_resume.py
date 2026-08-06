import torch

from splendor_rl.checkpoint import load_checkpoint, save_checkpoint
from splendor_rl.config import PPOConfig
from splendor_rl.models import PrivilegedCritic, SharedActor


def test_checkpoint_roundtrip(tmp_path):
    actor = SharedActor(4, 3, [8])
    critic = PrivilegedCritic(5, [8])
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
    sizes = {"actor": 4, "critic": 5, "action": 3}
    x = torch.randn(2, 4)
    expected = actor(x).detach()
    save_checkpoint(
        tmp_path / "x.pt", actor, critic, opt, PPOConfig(hidden_sizes=[8]), 10, 2, sizes
    )
    restored = SharedActor(4, 3, [8])
    data = load_checkpoint(
        tmp_path / "x.pt", restored, expected_sizes=sizes, restore_rng=False
    )
    assert torch.equal(expected, restored(x)) and data["global_transition_count"] == 10
