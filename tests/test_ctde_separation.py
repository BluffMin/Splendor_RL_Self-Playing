import torch

from splendor_env.core import SplendorGame
from splendor_rl.models import PrivilegedCritic, SharedActor


def test_actor_public_critic_privileged_separation():
    game = SplendorGame(2, seed=2)
    actor = SharedActor(475, 373, [16])
    critic = PrivilegedCritic(475, [16])
    public = game.observation(1)
    privileged = game.observation(1, omniscient=True)
    with torch.no_grad():
        logits = actor(torch.tensor(public).unsqueeze(0))
        value = critic(torch.tensor(privileged).unsqueeze(0))
    assert logits.shape == (1, 373) and value.shape == (1,)
