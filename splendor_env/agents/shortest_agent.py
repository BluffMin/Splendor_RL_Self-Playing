from __future__ import annotations

from ..actions import ACTIONS, DISCARD_OFFSET, NOBLE_OFFSET, PAYMENT_OFFSET
from ..core import Phase, SplendorGame


class ShortestAgent:
    """Deterministic demo agent preferring the cheapest available purchase."""

    def act(self, game: SplendorGame) -> int:
        legal = game.legal_actions()
        if game.phase == Phase.PAYMENT:
            return PAYMENT_OFFSET
        if game.phase == Phase.DISCARD:
            return DISCARD_OFFSET
        if game.phase == Phase.NOBLE:
            return min(a for a in legal if a >= NOBLE_OFFSET)
        buys = [a for a in legal if ACTIONS[a].kind in {"buy_visible", "buy_reserved"}]
        if buys:
            def key(action: int) -> tuple[int, int, int]:
                spec = ACTIONS[action]
                card = game._visible_card(spec.payload) if spec.kind == "buy_visible" else game.players[game.current_player].reserved[spec.payload].card
                assert card is not None
                return (sum(card.cost), -card.points, action)
            return min(buys, key=key)
        takes = [a for a in legal if ACTIONS[a].kind in {"take_distinct", "take_double"}]
        return min(takes or legal)
