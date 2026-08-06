from __future__ import annotations

from ..actions import ACTIONS, DISCARD_OFFSET, NOBLE_OFFSET, PAYMENT_OFFSET
from ..core import Phase, SplendorGame


class GreedyAgent:
    """Deterministic validation agent favoring points and cheap cards."""

    def act(self, game: SplendorGame) -> int:
        legal = game.legal_actions()
        if game.phase == Phase.PAYMENT:
            return PAYMENT_OFFSET  # Plans are sorted by least gold first.
        if game.phase == Phase.DISCARD:
            return DISCARD_OFFSET
        if game.phase == Phase.NOBLE:
            return min(a for a in legal if a >= NOBLE_OFFSET)
        buys = [a for a in legal if ACTIONS[a].kind in {"buy_visible", "buy_reserved"}]
        if buys:

            def value(action: int) -> tuple[int, int, int]:
                spec = ACTIONS[action]
                card = (
                    game._visible_card(spec.payload)
                    if spec.kind == "buy_visible"
                    else game.players[game.current_player].reserved[spec.payload].card
                )
                assert card is not None
                return (-card.points, sum(card.cost), action)

            return min(buys, key=value)
        takes = [
            a for a in legal if ACTIONS[a].kind in {"take_distinct", "take_double"}
        ]
        if takes:
            return min(takes)
        return min(legal)
