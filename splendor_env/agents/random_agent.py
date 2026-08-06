from __future__ import annotations

import numpy as np

from ..actions import ACTIONS
from ..core import SplendorGame


class RandomLegalAgent:
    """Uniform seeded sampler over the current legal action mask."""

    def __init__(self, seed: int | None = None, *, avoid_deadlock: bool = False) -> None:
        self.rng = np.random.default_rng(seed)
        self.avoid_deadlock = bool(avoid_deadlock)

    def act(self, game: SplendorGame) -> int:
        legal = game.legal_actions()
        if self.avoid_deadlock:
            for kinds in (
                {"buy_visible", "buy_reserved"},
                {"take_distinct", "take_double"},
            ):
                preferred = [action for action in legal if ACTIONS[action].kind in kinds]
                if preferred:
                    legal = preferred
                    break
        return int(self.rng.choice(legal))
