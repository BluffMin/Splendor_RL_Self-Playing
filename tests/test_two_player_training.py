import numpy as np

from splendor_env.core import GLOBAL_OBSERVATION_SIZE, PLAYER_BLOCK_SIZE, SplendorGame
from splendor_env.wrappers import rank_rewards


class Ranking:
    num_players = 2

    def __init__(self, tie=False):
        self.tie = tie

    def final_ranking(self):
        return (
            [{"rank": 1, "players": [0, 1]}]
            if self.tie
            else [{"rank": 1, "players": [0]}, {"rank": 2, "players": [1]}]
        )


def test_two_player_rewards_and_padded_observations():
    assert rank_rewards(Ranking()) == {0: 1.0, 1: -1.0}
    assert rank_rewards(Ranking(True)) == {0: 0.0, 1: 0.0}
    game = SplendorGame(2, seed=1)
    for perspective in (0, 1):
        actor = game.observation(perspective)
        critic = game.observation(perspective, omniscient=True)
        assert actor.shape == critic.shape == (
            475,
        ) and game.legal_action_mask().shape == (373,)
        padding = slice(
            GLOBAL_OBSERVATION_SIZE + 2 * PLAYER_BLOCK_SIZE,
            GLOBAL_OBSERVATION_SIZE + 4 * PLAYER_BLOCK_SIZE,
        )
        assert np.all(actor[padding] == 0) and np.all(critic[padding] == 0)
