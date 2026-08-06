import pytest

from splendor_env.core import SplendorGame
from splendor_env.wrappers import rank_rewards, rank_utility


def test_rank_utilities_and_tie_average():
    assert [rank_utility(i, 4) for i in range(4)] == pytest.approx(
        [1, 1 / 3, -1 / 3, -1]
    )
    game = SplendorGame(4, seed=1)
    rewards = rank_rewards(game)
    assert list(rewards.values()) == pytest.approx([0, 0, 0, 0]) and sum(
        rewards.values()
    ) == pytest.approx(0)
