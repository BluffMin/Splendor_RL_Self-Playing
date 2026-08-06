"""RL-only observation, reward, and canonical-payment facade."""

from __future__ import annotations

import numpy as np

from ..actions import N_ACTIONS
from ..core import OBSERVATION_SIZE, SplendorGame
from .canonical_payment import CanonicalPaymentWrapper


def rank_utility(rank_index: float, num_players: int) -> float:
    return 1.0 - 2.0 * rank_index / (num_players - 1)


def rank_rewards(game: SplendorGame) -> dict[int, float]:
    rewards: dict[int, float] = {}
    occupied = 0
    for group in game.final_ranking():
        players = list(group["players"])
        utility = sum(
            rank_utility(i, game.num_players)
            for i in range(occupied, occupied + len(players))
        ) / len(players)
        rewards.update({int(player): utility for player in players})
        occupied += len(players)
    if abs(sum(rewards.values())) >= 1e-6:
        raise AssertionError("rank rewards are not constant-sum")
    return rewards


class SelfPlayWrapper:
    actor_observation_size = OBSERVATION_SIZE
    critic_state_size = OBSERVATION_SIZE
    action_size = N_ACTIONS

    def __init__(
        self,
        num_players: int = 4,
        *,
        seed: int | None = None,
        payment_mode: str = "canonical",
        max_turns: int | None = None,
    ) -> None:
        self.game = SplendorGame(num_players, seed=seed)
        self.payment = CanonicalPaymentWrapper(self.game, payment_mode)  # type: ignore[arg-type]
        self.payment_mode = payment_mode
        self.max_turns = max_turns
        self.truncated = False

    def actor_observation(self, player_id: int) -> np.ndarray:
        return self.game.observation(player_id, omniscient=False)

    def critic_state(self, player_id: int) -> np.ndarray:
        """Return a privileged egocentric observation.

        Private reserved-card payloads are visible, but the complete hidden
        deck order is not encoded.
        """
        return self.game.observation(player_id, omniscient=True)

    def action_mask(self) -> np.ndarray:
        return self.game.legal_action_mask().astype(bool)

    def step(self, action: int):
        result = self.payment.policy_step(action)
        if (
            self.max_turns
            and not self.game.done
            and self.game.turns_completed >= self.max_turns
        ):
            self.game.truncate()
            self.truncated = True
        return result

    def rewards(self) -> dict[int, float]:
        return (
            rank_rewards(self.game)
            if self.game.terminated
            else {i: 0.0 for i in range(self.game.num_players)}
        )
