"""Pure perspective-aware view models, independent of HTML rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..actions import GEM_COLORS, TOKEN_COLORS
from ..core import Phase, SplendorGame
from ..data import CARDS, NOBLES, Card, Noble

Perspective = int | Literal["current", "omniscient"]
LayoutMode = Literal["table", "egocentric"]

CARD_BY_ID = {card.card_id: card for card in CARDS}
NOBLE_BY_ID = {noble.noble_id: noble for noble in NOBLES}


@dataclass(frozen=True, slots=True)
class CardView:
    card_id: str | None
    tier: int
    visible: bool
    bonus_color: str | None
    points: int | None
    cost: dict[str, int] | None
    reservation_origin: str | None = None
    market_slot: int | None = None


@dataclass(frozen=True, slots=True)
class NobleView:
    noble_id: str
    points: int
    requirements: dict[str, int]


@dataclass(frozen=True, slots=True)
class PlayerView:
    player_id: int
    display_name: str
    is_current: bool
    score: int
    tokens: dict[str, int]
    bonuses: dict[str, int]
    purchased_card_count: int
    purchased_cards_by_color: dict[str, int]
    purchased_cards_by_tier: dict[int, int]
    purchased_cards: tuple[CardView, ...]
    reserved_cards: tuple[CardView, ...]
    nobles: tuple[NobleView, ...]
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class BoardView:
    game_id: str | None
    num_players: int
    perspective: Perspective
    layout: LayoutMode
    round_id: int
    turn_id: int
    decision_id: int
    phase: str
    current_player: int
    bank_tokens: dict[str, int]
    market_cards: tuple[CardView, ...]
    deck_counts: dict[int, int]
    nobles: tuple[NobleView, ...]
    players: tuple[PlayerView, ...]
    last_action_text: str | None
    last_action_player: int | None
    pending_choice_text: str | None
    is_terminal: bool
    winner_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StateDelta:
    changed_market_slots: tuple[int, ...]
    token_changes: dict[str, int]
    player_token_changes: dict[int, dict[str, int]]
    player_score_changes: dict[int, int]
    purchased_card_ids: tuple[str, ...]
    reserved_card_ids: tuple[str, ...]
    acquired_noble_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_perspective(
    perspective: Perspective, current_player: int
) -> tuple[int, bool]:
    if perspective == "omniscient":
        return current_player, True
    if perspective == "current":
        return current_player, False
    return int(perspective), False


def _card_view(
    card: Card,
    *,
    visible: bool = True,
    origin: str | None = None,
    market_slot: int | None = None,
) -> CardView:
    return CardView(
        card_id=card.card_id if visible else None,
        tier=card.tier + 1,
        visible=visible,
        bonus_color=card.bonus_color if visible else None,
        points=card.points if visible else None,
        cost=card.cost_by_color() if visible else None,
        reservation_origin=origin,
        market_slot=market_slot,
    )


def _noble_view(noble: Noble) -> NobleView:
    return NobleView(noble.noble_id, noble.points, noble.requirements_by_color())


def _pending_text(
    phase: str, payment_count: int, excess: int, player: int
) -> str | None:
    if phase == Phase.PAYMENT.value:
        return f"P{player} is choosing a payment plan · {payment_count} legal plans"
    if phase == Phase.DISCARD.value:
        return f"P{player} must return {excess} tokens"
    if phase == Phase.NOBLE.value:
        return f"P{player} is choosing a noble"
    return None


def board_view_from_game(
    game: SplendorGame,
    *,
    perspective: Perspective = "current",
    layout: LayoutMode = "table",
    game_id: str | None = None,
) -> BoardView:
    """Convert a live engine state without mutating or querying legal actions."""
    viewer, omniscient = _resolve_perspective(perspective, game.current_player)
    if not 0 <= viewer < game.num_players:
        raise ValueError("invalid visualization perspective")
    market = tuple(
        _card_view(card, market_slot=tier * 4 + slot)
        if card is not None
        else CardView(
            None, tier + 1, False, None, None, None, market_slot=tier * 4 + slot
        )
        for tier in range(3)
        for slot, card in enumerate(game.visible[tier])
    )
    rank_by_player = (
        {
            player: int(group["rank"])
            for group in game.final_ranking()
            for player in group["players"]
        }
        if game.done
        else {}
    )
    order = list(range(game.num_players))
    if layout == "egocentric":
        order = [
            (viewer + offset) % game.num_players for offset in range(game.num_players)
        ]
    players = []
    for player_id in order:
        player = game.players[player_id]
        reserved = tuple(
            _card_view(
                reservation.card,
                visible=game._reservation_visible(
                    player_id, viewer, reservation, omniscient
                ),
                origin=reservation.origin,
            )
            for reservation in player.reserved
        )
        players.append(
            PlayerView(
                player_id=player_id,
                display_name=f"Player {player_id}",
                is_current=player_id == game.current_player and not game.done,
                score=player.score,
                tokens=dict(zip(TOKEN_COLORS, map(int, player.tokens), strict=True)),
                bonuses=dict(zip(GEM_COLORS, map(int, player.bonuses), strict=True)),
                purchased_card_count=len(player.purchased),
                purchased_cards_by_color={
                    color: sum(c.bonus_color == color for c in player.purchased)
                    for color in GEM_COLORS
                },
                purchased_cards_by_tier={
                    tier: sum(c.tier == tier - 1 for c in player.purchased)
                    for tier in (1, 2, 3)
                },
                purchased_cards=tuple(_card_view(card) for card in player.purchased),
                reserved_cards=reserved,
                nobles=tuple(_noble_view(noble) for noble in player.nobles),
                rank=rank_by_player.get(player_id),
            )
        )
    excess = max(game.players[game.current_player].token_count - 10, 0)
    return BoardView(
        game_id=game_id,
        num_players=game.num_players,
        perspective=perspective,
        layout=layout,
        round_id=game.round_id,
        turn_id=game.turns_completed,
        decision_id=game.decision_id,
        phase=game.phase.value,
        current_player=game.current_player,
        bank_tokens=dict(zip(TOKEN_COLORS, map(int, game.bank), strict=True)),
        market_cards=market,
        deck_counts={tier + 1: len(game.decks[tier]) for tier in range(3)},
        nobles=tuple(_noble_view(noble) for noble in game.nobles),
        players=tuple(players),
        last_action_text=game.last_action_text,
        last_action_player=game.last_actor,
        pending_choice_text=_pending_text(
            game.phase.value,
            len(game.pending_payment_plans),
            excess,
            game.current_player,
        ),
        is_terminal=game.done,
        winner_ids=tuple(game.winner_ids()) if game.done else (),
    )


def board_view_from_snapshot(
    snapshot: dict[str, Any],
    *,
    perspective: Perspective = "current",
    layout: LayoutMode = "table",
) -> BoardView:
    """Convert an omniscient state dictionary using the same visibility rules."""
    current = int(snapshot["current_player"])
    viewer, omniscient = _resolve_perspective(perspective, current)
    num_players = int(snapshot["num_players"])
    if not 0 <= viewer < num_players:
        raise ValueError("invalid visualization perspective")
    market = tuple(
        _card_view(CARD_BY_ID[card_id], market_slot=tier * 4 + slot)
        if card_id is not None
        else CardView(
            None, tier + 1, False, None, None, None, market_slot=tier * 4 + slot
        )
        for tier, row in enumerate(snapshot["visible"])
        for slot, card_id in enumerate(row)
    )
    order = list(range(num_players))
    if layout == "egocentric":
        order = [(viewer + offset) % num_players for offset in range(num_players)]
    players = []
    for player_id in order:
        player = snapshot["players"][player_id]
        purchased = [CARD_BY_ID[card_id] for card_id in player["purchased"]]
        reserved = []
        for item in player["reserved"]:
            card = CARD_BY_ID[item["card_id"]]
            origin = item["origin"]
            can_see = omniscient or player_id == viewer or origin == "visible"
            reserved.append(_card_view(card, visible=can_see, origin=origin))
        players.append(
            PlayerView(
                player_id=player_id,
                display_name=f"Player {player_id}",
                is_current=player_id == current
                and not snapshot["terminated"]
                and not snapshot["truncated"],
                score=int(player["score"]),
                tokens=dict(zip(TOKEN_COLORS, player["tokens"], strict=True)),
                bonuses=dict(zip(GEM_COLORS, player["bonuses"], strict=True)),
                purchased_card_count=len(purchased),
                purchased_cards_by_color={
                    color: sum(c.bonus_color == color for c in purchased)
                    for color in GEM_COLORS
                },
                purchased_cards_by_tier={
                    tier: sum(c.tier == tier - 1 for c in purchased)
                    for tier in (1, 2, 3)
                },
                purchased_cards=tuple(_card_view(card) for card in purchased),
                reserved_cards=tuple(reserved),
                nobles=tuple(
                    _noble_view(NOBLE_BY_ID[noble_id]) for noble_id in player["nobles"]
                ),
            )
        )
    phase = str(snapshot["phase"])
    excess = max(sum(snapshot["players"][current]["tokens"]) - 10, 0)
    return BoardView(
        game_id=None,
        num_players=num_players,
        perspective=perspective,
        layout=layout,
        round_id=int(snapshot["round_id"]),
        turn_id=int(snapshot["turns_completed"]),
        decision_id=int(snapshot["decision_id"]),
        phase=phase,
        current_player=current,
        bank_tokens=dict(zip(TOKEN_COLORS, snapshot["bank"], strict=True)),
        market_cards=market,
        deck_counts={tier + 1: len(snapshot["decks"][tier]) for tier in range(3)},
        nobles=tuple(
            _noble_view(NOBLE_BY_ID[noble_id]) for noble_id in snapshot["nobles"]
        ),
        players=tuple(players),
        last_action_text=None,
        last_action_player=None,
        pending_choice_text=_pending_text(
            phase, len(snapshot["pending_payment_plans"]), excess, current
        ),
        is_terminal=bool(snapshot["terminated"] or snapshot["truncated"]),
        winner_ids=(),
    )


def compute_state_delta(
    pre_snapshot: dict[str, Any], post_snapshot: dict[str, Any]
) -> StateDelta:
    """Compute resource and holding changes between omniscient state dictionaries."""
    changed_slots = tuple(
        tier * 4 + slot
        for tier in range(3)
        for slot in range(4)
        if pre_snapshot["visible"][tier][slot] != post_snapshot["visible"][tier][slot]
    )
    token_changes = {
        color: int(post_snapshot["bank"][i]) - int(pre_snapshot["bank"][i])
        for i, color in enumerate(TOKEN_COLORS)
        if post_snapshot["bank"][i] != pre_snapshot["bank"][i]
    }
    player_token_changes: dict[int, dict[str, int]] = {}
    score_changes: dict[int, int] = {}
    purchased: list[str] = []
    reserved: list[str] = []
    nobles: list[str] = []
    for player_id, (before, after) in enumerate(
        zip(pre_snapshot["players"], post_snapshot["players"], strict=True)
    ):
        changes = {
            color: int(after["tokens"][i]) - int(before["tokens"][i])
            for i, color in enumerate(TOKEN_COLORS)
            if after["tokens"][i] != before["tokens"][i]
        }
        if changes:
            player_token_changes[player_id] = changes
        if after["score"] != before["score"]:
            score_changes[player_id] = int(after["score"]) - int(before["score"])
        purchased += [x for x in after["purchased"] if x not in before["purchased"]]
        before_reserved = {x["card_id"] for x in before["reserved"]}
        reserved += [
            x["card_id"]
            for x in after["reserved"]
            if x["card_id"] not in before_reserved
        ]
        nobles += [x for x in after["nobles"] if x not in before["nobles"]]
    return StateDelta(
        changed_slots,
        token_changes,
        player_token_changes,
        score_changes,
        tuple(purchased),
        tuple(reserved),
        tuple(nobles),
    )
