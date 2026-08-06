"""Deterministic rules engine for the original Splendor base game."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any

import numpy as np

from .actions import (
    ACTIONS,
    DISCARD_OFFSET,
    DISCARD_SIZE,
    GEM_COLORS,
    N_ACTIONS,
    NOBLE_OFFSET,
    PAYMENT_OFFSET,
    PAYMENT_SIZE,
    TOKEN_COLORS,
    describe_action,
)
from .data import CARDS, NOBLES, Card, Noble

MAX_PLAYERS = 4
MAX_RESERVED = 3
MAX_NOBLES = 5
WINNING_SCORE = 15
MAX_TOKENS = 10


class Phase(str, Enum):
    NORMAL = "normal"
    PAYMENT = "payment"
    DISCARD = "discard"
    NOBLE = "noble"
    TERMINAL = "terminal"


PHASES = tuple(Phase)
GLOBAL_OBSERVATION_SIZE = 215
PLAYER_PUBLIC_SIZE = 17
RESERVED_SLOT_SIZE = 16
PLAYER_BLOCK_SIZE = PLAYER_PUBLIC_SIZE + MAX_RESERVED * RESERVED_SLOT_SIZE
OBSERVATION_SIZE = GLOBAL_OBSERVATION_SIZE + MAX_PLAYERS * PLAYER_BLOCK_SIZE
OBS_LAYOUT = {
    "phase": slice(0, 5),
    "global": slice(0, GLOBAL_OBSERVATION_SIZE),
    "players": slice(GLOBAL_OBSERVATION_SIZE, OBSERVATION_SIZE),
}
assert OBSERVATION_SIZE == 475


class InvalidActionError(ValueError):
    """Raised when an action is outside the current legal-action mask."""


class NoLegalActionError(RuntimeError):
    """Raised when an unfinished normal state has no official action."""


@dataclass(frozen=True, slots=True)
class PaymentPlan:
    colored: tuple[int, int, int, int, int]
    gold_by_color: tuple[int, int, int, int, int]

    @property
    def total_gold(self) -> int:
        return sum(self.gold_by_color)

    def to_dict(self) -> dict[str, object]:
        return {
            "colored": dict(zip(GEM_COLORS, self.colored, strict=True)),
            "gold_by_color": dict(zip(GEM_COLORS, self.gold_by_color, strict=True)),
            "total_gold": self.total_gold,
        }


@dataclass(frozen=True, slots=True)
class DiscardPlan:
    tokens: tuple[int, int, int, int, int, int]

    def to_dict(self) -> dict[str, int]:
        return dict(zip(TOKEN_COLORS, self.tokens, strict=True))


@dataclass(slots=True)
class Reservation:
    card: Card
    hidden_to_opponents: bool

    @property
    def origin(self) -> str:
        return "deck" if self.hidden_to_opponents else "visible"


@dataclass(slots=True)
class PlayerState:
    tokens: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int16))
    bonuses: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int16))
    score: int = 0
    purchased: list[Card] = field(default_factory=list)
    nobles: list[Noble] = field(default_factory=list)
    reserved: list[Reservation] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return int(self.tokens.sum())


@dataclass(frozen=True, slots=True)
class EngineStep:
    actor: int
    action: int
    score_delta: int
    turn_ended: bool
    terminated: bool
    truncated: bool
    phase: Phase
    decision_id: int
    turn_id: int
    round_id: int
    automatic_resolution: tuple[str, ...] = ()


def enumerate_legal_payment_plans(
    player: PlayerState, card: Card
) -> tuple[PaymentPlan, ...]:
    """Enumerate every exact legal colored/gold payment in stable order."""
    required = np.maximum(np.asarray(card.cost, dtype=np.int16) - player.bonuses, 0)
    plans: list[PaymentPlan] = []
    ranges = [range(int(required[i]) + 1) for i in range(5)]
    for gold in product(*ranges):
        colored = tuple(int(required[i]) - gold[i] for i in range(5))
        if all(colored[i] <= int(player.tokens[i]) for i in range(5)) and sum(gold) <= int(player.tokens[5]):
            plans.append(PaymentPlan(colored, tuple(int(x) for x in gold)))
    plans.sort(key=lambda plan: (plan.total_gold, plan.colored, plan.gold_by_color))
    return tuple(plans)


def enumerate_legal_discard_plans(
    player: PlayerState, excess: int
) -> tuple[DiscardPlan, ...]:
    """Enumerate exact token-return combinations in lexicographic order."""
    if excess < 0:
        raise ValueError("excess must be non-negative")
    plans = [
        DiscardPlan(tuple(int(x) for x in values))
        for values in product(*(range(min(int(n), excess) + 1) for n in player.tokens))
        if sum(values) == excess
    ]
    return tuple(sorted(set(plans), key=lambda plan: plan.tokens))


class SplendorGame:
    """Seeded base-game engine with no non-official termination rule."""

    def __init__(self, num_players: int = 2, *, seed: int | None = None) -> None:
        if not 2 <= num_players <= 4:
            raise ValueError("num_players must be between 2 and 4")
        self.num_players = int(num_players)
        self._seed = seed
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.reset(seed=seed)

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def initial_colored_token_count(self) -> int:
        return {2: 4, 3: 5, 4: 7}[self.num_players]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def round_id(self) -> int:
        return self.turns_completed // self.num_players

    def add_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Attach a passive decision-event listener."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.players = [PlayerState() for _ in range(self.num_players)]
        self.bank = np.asarray([self.initial_colored_token_count] * 5 + [5], dtype=np.int16)
        self.decks = []
        for tier in range(3):
            deck = [card for card in CARDS if card.tier == tier]
            self.rng.shuffle(deck)
            self.decks.append(deck)
        self.visible: list[list[Card | None]] = [[None] * 4 for _ in range(3)]
        for tier in range(3):
            for slot in range(4):
                self.visible[tier][slot] = self._draw(tier)
        noble_indices = self.rng.choice(len(NOBLES), size=self.num_players + 1, replace=False)
        selected = {int(i) for i in noble_indices}
        self.nobles = [NOBLES[i] for i in sorted(selected)]
        self.unused_nobles = [n for i, n in enumerate(NOBLES) if i not in selected]
        self.current_player = 0
        self.phase = Phase.NORMAL
        self.turns_completed = 0
        self.decision_id = 0
        self.final_round_triggered = False
        self.trigger_player: int | None = None
        self.terminated = False
        self.truncated = False
        self.end_reason: str | None = None
        self.last_action: int | None = None
        self.last_actor: int | None = None
        self.last_action_text: str | None = None
        self.pending_purchase: tuple[str, int, Card] | None = None
        self.pending_payment_plans: tuple[PaymentPlan, ...] = ()
        self.pending_discard_plans: tuple[DiscardPlan, ...] = ()
        self._automatic: list[str] = []
        self.validate_invariants()

    def _draw(self, tier: int) -> Card | None:
        return self.decks[tier].pop() if self.decks[tier] else None

    def _visible_card(self, flat_slot: int) -> Card | None:
        tier, slot = divmod(flat_slot, 4)
        return self.visible[tier][slot]

    def _remove_visible(self, flat_slot: int) -> Card:
        tier, slot = divmod(flat_slot, 4)
        card = self.visible[tier][slot]
        if card is None:
            raise RuntimeError("attempted to remove an empty visible slot")
        self.visible[tier][slot] = self._draw(tier)
        return card

    def can_afford(self, player_index: int, card: Card) -> bool:
        return bool(enumerate_legal_payment_plans(self.players[player_index], card))

    def eligible_noble_slots(self, player_index: int | None = None) -> list[int]:
        player_index = self.current_player if player_index is None else player_index
        bonuses = self.players[player_index].bonuses
        return [i for i, n in enumerate(self.nobles) if np.all(bonuses >= np.asarray(n.requirements))]

    def legal_action_mask(self, player_index: int | None = None) -> np.ndarray:
        """Return a 373-entry mask with only the current phase region enabled."""
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if self.done or (player_index is not None and player_index != self.current_player):
            return mask
        if self.phase == Phase.PAYMENT:
            if not self.pending_payment_plans:
                raise NoLegalActionError("payment phase has no legal plans")
            mask[PAYMENT_OFFSET : PAYMENT_OFFSET + len(self.pending_payment_plans)] = 1
            return mask
        if self.phase == Phase.DISCARD:
            if not self.pending_discard_plans:
                raise NoLegalActionError("discard phase has no legal plans")
            mask[DISCARD_OFFSET : DISCARD_OFFSET + len(self.pending_discard_plans)] = 1
            return mask
        if self.phase == Phase.NOBLE:
            for slot in self.eligible_noble_slots():
                mask[NOBLE_OFFSET + slot] = 1
            return mask
        if self.phase != Phase.NORMAL:
            return mask
        available = [i for i in range(5) if self.bank[i] > 0]
        if available:
            required_size = min(3, len(available))
            for action, spec in enumerate(ACTIONS[:25]):
                if len(spec.payload) == required_size and all(self.bank[c] > 0 for c in spec.payload):
                    mask[action] = 1
        for color in range(5):
            if self.bank[color] >= 4:
                mask[25 + color] = 1
        for slot in range(12):
            card = self._visible_card(slot)
            if card is not None and self.can_afford(self.current_player, card):
                mask[30 + slot] = 1
        player = self.players[self.current_player]
        if len(player.reserved) < MAX_RESERVED:
            for slot in range(12):
                if self._visible_card(slot) is not None:
                    mask[42 + slot] = 1
            for tier in range(3):
                if self.decks[tier]:
                    mask[54 + tier] = 1
        for slot, reservation in enumerate(player.reserved):
            if self.can_afford(self.current_player, reservation.card):
                mask[57 + slot] = 1
        if not mask.any():
            raise NoLegalActionError(
                f"no official action: player={self.current_player}, turn={self.turns_completed}, "
                f"bank={self.bank.tolist()}, reserves={len(player.reserved)}"
            )
        return mask

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.legal_action_mask()).astype(int).tolist()

    def step(self, action: int) -> EngineStep:
        """Apply one decision, validate conservation, and emit a replay event."""
        if self.done:
            raise RuntimeError("cannot step a finished game")
        action = int(action)
        if not 0 <= action < N_ACTIONS or not self.legal_action_mask()[action]:
            raise InvalidActionError(f"illegal action {action}: {describe_action(action)}")
        pre_hash = self.state_hash()
        actor = self.current_player
        score_before = self.players[actor].score
        turn_before = self.turns_completed
        phase_before = self.phase
        turn_id = self.turns_completed
        round_id = self.round_id
        self._automatic = []
        spec = ACTIONS[action]
        if self.phase == Phase.PAYMENT:
            selected_payment = self.pending_payment_plans[action - PAYMENT_OFFSET]
            action_params: Any = selected_payment.to_dict()
            self._complete_purchase(selected_payment)
        elif self.phase == Phase.DISCARD:
            selected_discard = self.pending_discard_plans[action - DISCARD_OFFSET]
            action_params = selected_discard.to_dict()
            self._apply_discard(selected_discard)
        elif self.phase == Phase.NOBLE:
            action_params = {
                "slot": action - NOBLE_OFFSET,
                "noble_id": self.nobles[action - NOBLE_OFFSET].noble_id,
            }
            self._choose_noble(action - NOBLE_OFFSET)
        elif spec.kind == "take_distinct":
            action_params = {"colors": [GEM_COLORS[i] for i in spec.payload]}
            self._take_distinct(spec.payload)
        elif spec.kind == "take_double":
            action_params = {"color": GEM_COLORS[int(spec.payload)]}
            self._take_double(int(spec.payload))
        elif spec.kind == "buy_visible":
            action_params = {"slot": int(spec.payload)}
            card = self._visible_card(int(spec.payload))
            if card is None:
                raise RuntimeError("visible card disappeared")
            self._begin_purchase("visible", int(spec.payload), card)
        elif spec.kind == "reserve_visible":
            action_params = {"slot": int(spec.payload)}
            self._reserve(self._remove_visible(int(spec.payload)), hidden=False)
            self._after_normal_action()
        elif spec.kind == "reserve_deck":
            action_params = {"tier": int(spec.payload) + 1}
            self._reserve(self.decks[int(spec.payload)].pop(), hidden=True)
            self._after_normal_action()
        elif spec.kind == "buy_reserved":
            action_params = {"slot": int(spec.payload)}
            slot = int(spec.payload)
            self._begin_purchase("reserved", slot, self.players[actor].reserved[slot].card)
        else:
            raise RuntimeError(f"unexpected action {spec}")
        self.last_action = action
        self.last_actor = actor
        self.last_action_text = describe_action(action)
        self.decision_id += 1
        self.validate_invariants()
        result = EngineStep(
            actor=actor,
            action=action,
            score_delta=self.players[actor].score - score_before,
            turn_ended=self.turns_completed > turn_before,
            terminated=self.terminated,
            truncated=self.truncated,
            phase=self.phase,
            decision_id=self.decision_id - 1,
            turn_id=turn_id,
            round_id=round_id,
            automatic_resolution=tuple(self._automatic),
        )
        event = {
            "decision_id": result.decision_id,
            "turn_id": turn_id,
            "round_id": round_id,
            "phase": phase_before.value,
            "player": actor,
            "action_id": action,
            "action_type": spec.kind,
            "action_params": action_params,
            "action_text": describe_action(action),
            "automatic": False,
            "automatic_resolution": list(self._automatic),
            "turn_completed": result.turn_ended,
            "pre_state_hash": pre_hash,
            "post_state_hash": self.state_hash(),
        }
        for listener in tuple(self._listeners):
            listener(event)
        return result

    def _take_distinct(self, colors: Iterable[int]) -> None:
        for color in colors:
            self.bank[color] -= 1
            self.players[self.current_player].tokens[color] += 1
        self._after_normal_action()

    def _take_double(self, color: int) -> None:
        self.bank[color] -= 2
        self.players[self.current_player].tokens[color] += 2
        self._after_normal_action()

    def _reserve(self, card: Card, *, hidden: bool) -> None:
        player = self.players[self.current_player]
        player.reserved.append(Reservation(card, hidden))
        if self.bank[5] > 0:
            self.bank[5] -= 1
            player.tokens[5] += 1

    def _begin_purchase(self, source: str, slot: int, card: Card) -> None:
        plans = enumerate_legal_payment_plans(self.players[self.current_player], card)
        if not plans:
            raise RuntimeError(f"no legal payment for {card.card_id}")
        if len(plans) > PAYMENT_SIZE:
            raise RuntimeError("payment plan region overflow")
        self.pending_purchase = (source, slot, card)
        self.pending_payment_plans = plans
        self.phase = Phase.PAYMENT

    def _complete_purchase(self, plan: PaymentPlan) -> None:
        if self.pending_purchase is None:
            raise RuntimeError("no pending purchase")
        source, slot, card = self.pending_purchase
        player = self.players[self.current_player]
        colored = np.asarray(plan.colored, dtype=np.int16)
        player.tokens[:5] -= colored
        self.bank[:5] += colored
        player.tokens[5] -= plan.total_gold
        self.bank[5] += plan.total_gold
        removed = self._remove_visible(slot) if source == "visible" else player.reserved.pop(slot).card
        if removed.card_id != card.card_id:
            raise RuntimeError("pending purchase card changed")
        player.purchased.append(card)
        player.bonuses[card.bonus] += 1
        player.score += card.points
        self.pending_purchase = None
        self.pending_payment_plans = ()
        self._after_normal_action()

    def _after_normal_action(self) -> None:
        excess = self.players[self.current_player].token_count - MAX_TOKENS
        if excess > 0:
            plans = enumerate_legal_discard_plans(self.players[self.current_player], excess)
            if len(plans) > DISCARD_SIZE:
                raise RuntimeError(f"discard plan region overflow: {len(plans)}")
            self.pending_discard_plans = plans
            self.phase = Phase.DISCARD
        else:
            self._resolve_noble_or_end_turn()

    def _apply_discard(self, plan: DiscardPlan) -> None:
        returned = np.asarray(plan.tokens, dtype=np.int16)
        self.players[self.current_player].tokens -= returned
        self.bank += returned
        if self.players[self.current_player].token_count != MAX_TOKENS:
            raise RuntimeError("discard did not leave exactly ten tokens")
        self.pending_discard_plans = ()
        self._resolve_noble_or_end_turn()

    def _resolve_noble_or_end_turn(self) -> None:
        eligible = self.eligible_noble_slots()
        if len(eligible) == 1:
            noble_id = self.nobles[eligible[0]].noble_id
            self._award_noble(eligible[0])
            self._automatic.append(f"award_noble:{noble_id}")
            self._end_turn()
        elif len(eligible) > 1:
            self.phase = Phase.NOBLE
        else:
            self._end_turn()

    def _choose_noble(self, slot: int) -> None:
        if slot not in self.eligible_noble_slots():
            raise InvalidActionError(f"noble slot {slot} is not eligible")
        self._award_noble(slot)
        self._end_turn()

    def _award_noble(self, slot: int) -> None:
        noble = self.nobles.pop(slot)
        player = self.players[self.current_player]
        player.nobles.append(noble)
        player.score += noble.points

    def _end_turn(self) -> None:
        actor = self.current_player
        self.turns_completed += 1
        if not self.final_round_triggered and self.players[actor].score >= WINNING_SCORE:
            self.final_round_triggered = True
            self.trigger_player = actor
        if self.final_round_triggered and actor == self.num_players - 1:
            self.terminated = True
            self.phase = Phase.TERMINAL
            self.end_reason = "official_game_end"
            return
        self.current_player = (actor + 1) % self.num_players
        self.phase = Phase.NORMAL

    def truncate(self, reason: str = "max_turns_truncation") -> None:
        """Mark an environment-owned safety truncation without changing rules."""
        self.truncated = True
        self.phase = Phase.TERMINAL
        self.end_reason = reason

    def final_ranking(self) -> list[dict[str, object]]:
        """Return competition rankings, preserving exact ties."""
        groups: dict[tuple[int, int], list[int]] = {}
        for i, player in enumerate(self.players):
            groups.setdefault((-player.score, len(player.purchased)), []).append(i)
        ranking: list[dict[str, object]] = []
        rank = 1
        for key in sorted(groups):
            players = groups[key]
            ranking.append({"rank": rank, "players": players})
            rank += len(players)
        return ranking

    def winner_ids(self) -> list[int]:
        return list(self.final_ranking()[0]["players"])

    def winners(self) -> list[int]:
        return self.winner_ids()

    def is_tied(self) -> bool:
        return len(self.winner_ids()) > 1

    def terminal_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.num_players, dtype=np.float32)
        if not self.done:
            return rewards
        winners = self.winner_ids()
        if len(winners) == self.num_players:
            return rewards
        rewards[winners] = 1.0
        penalty = -len(winners) / (self.num_players - len(winners))
        for i in range(self.num_players):
            if i not in winners:
                rewards[i] = penalty
        return rewards

    def _reservation_visible(self, owner: int, perspective: int, reservation: Reservation, omniscient: bool) -> bool:
        return omniscient or owner == perspective or not reservation.hidden_to_opponents

    def observation(self, perspective: int, *, omniscient: bool = False) -> np.ndarray:
        """Encode an egocentric partial observation with explicit phase context."""
        if not 0 <= perspective < self.num_players:
            raise ValueError("invalid perspective")
        values: list[float] = [float(self.phase == phase) for phase in PHASES]
        values += [float(self.done), float(self.final_round_triggered), min(self.round_id / 50.0, 1.0)]
        values += (self.bank.astype(np.float32) / np.asarray([7, 7, 7, 7, 7, 5])).tolist()
        values += [len(self.decks[i]) / (40, 30, 20)[i] for i in range(3)]
        for tier in range(3):
            for slot in range(4):
                values += self._encode_card(self.visible[tier][slot])
        for slot in range(MAX_NOBLES):
            values += self._encode_noble(self.nobles[slot] if slot < len(self.nobles) else None)
        # Pending purchase: present, visible-origin, private-origin, tier one-hot, payload.
        pending = [0.0] * 17
        if self.pending_purchase is not None:
            source, _, card = self.pending_purchase
            reservation = next((r for r in self.players[self.current_player].reserved if r.card.card_id == card.card_id), None)
            is_private = reservation is not None and reservation.hidden_to_opponents
            pending[:3] = [1.0, float(not is_private), float(is_private)]
            pending[3 + card.tier] = 1.0
            can_see = source == "visible" or reservation is None or self._reservation_visible(self.current_player, perspective, reservation, omniscient)
            if can_see:
                pending[6:] = self._encode_card(card)[1:]
        values += pending
        values += [min(len(self.pending_payment_plans) / PAYMENT_SIZE, 1.0), min(max(self.players[self.current_player].token_count - 10, 0) / 3.0, 1.0)]
        assert len(values) == GLOBAL_OBSERVATION_SIZE
        order = [(perspective + offset) % self.num_players for offset in range(self.num_players)]
        for player_slot in range(MAX_PLAYERS):
            if player_slot >= self.num_players:
                values += [0.0] * PLAYER_BLOCK_SIZE
                continue
            player_index = order[player_slot]
            player = self.players[player_index]
            values += [1.0, float(player_index == self.current_player and not self.done)]
            values += np.clip(player.tokens.astype(np.float32) / 13.0, 0, 1).tolist()
            values += np.clip(player.bonuses.astype(np.float32) / 20.0, 0, 1).tolist()
            values += [min(player.score / 30.0, 1.0), min(len(player.purchased) / 30.0, 1.0), min(len(player.nobles) / 5.0, 1.0), len(player.reserved) / 3.0]
            for reserved_slot in range(MAX_RESERVED):
                if reserved_slot >= len(player.reserved):
                    values += [0.0] * RESERVED_SLOT_SIZE
                    continue
                reservation = player.reserved[reserved_slot]
                tier = [0.0] * 3
                tier[reservation.card.tier] = 1.0
                values += [1.0, float(reservation.hidden_to_opponents)] + tier
                if self._reservation_visible(player_index, perspective, reservation, omniscient):
                    values += self._encode_card(reservation.card)[1:]
                else:
                    values += [0.0] * 11
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(f"encoder produced {result.shape}, expected {(OBSERVATION_SIZE,)}")
        return np.clip(result, 0, 1)

    @staticmethod
    def _encode_card(card: Card | None) -> list[float]:
        if card is None:
            return [0.0] * 12
        bonus = [0.0] * 5
        bonus[card.bonus] = 1.0
        return [1.0, card.points / 5.0] + bonus + [cost / 7.0 for cost in card.cost]

    @staticmethod
    def _encode_noble(noble: Noble | None) -> list[float]:
        return [0.0] * 7 if noble is None else [1.0, noble.points / 3.0] + [cost / 4.0 for cost in noble.requirements]

    def state(self) -> np.ndarray:
        return self.observation(0, omniscient=True)

    def to_state_dict(self, omniscient: bool = True) -> dict[str, Any]:
        """Return a deterministic JSON-compatible state representation."""
        del omniscient
        return {
            "seed": self._seed,
            "num_players": self.num_players,
            "current_player": self.current_player,
            "phase": self.phase.value,
            "turns_completed": self.turns_completed,
            "round_id": self.round_id,
            "decision_id": self.decision_id,
            "bank": self.bank.tolist(),
            "decks": [[c.card_id for c in deck] for deck in self.decks],
            "visible": [[c.card_id if c else None for c in row] for row in self.visible],
            "nobles": [n.noble_id for n in self.nobles],
            "unused_nobles": [n.noble_id for n in self.unused_nobles],
            "players": [
                {
                    "tokens": p.tokens.tolist(),
                    "bonuses": p.bonuses.tolist(),
                    "score": p.score,
                    "purchased": [c.card_id for c in p.purchased],
                    "reserved": [{"card_id": r.card.card_id, "origin": r.origin} for r in p.reserved],
                    "nobles": [n.noble_id for n in p.nobles],
                }
                for p in self.players
            ],
            "pending_purchase": None if self.pending_purchase is None else {"source": self.pending_purchase[0], "slot": self.pending_purchase[1], "card_id": self.pending_purchase[2].card_id},
            "pending_payment_plans": [p.to_dict() for p in self.pending_payment_plans],
            "pending_discard_plans": [p.to_dict() for p in self.pending_discard_plans],
            "final_round_triggered": self.final_round_triggered,
            "trigger_player": self.trigger_player,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "end_reason": self.end_reason,
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.to_state_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def final_summary(self) -> dict[str, Any]:
        """Return complete final holdings and competition ranks."""
        rank_by_player = {p: group["rank"] for group in self.final_ranking() for p in group["players"]}
        players = []
        for i, player in enumerate(self.players):
            players.append({
                "player_id": i,
                "rank": rank_by_player[i],
                "score": player.score,
                "tokens": dict(zip(TOKEN_COLORS, map(int, player.tokens), strict=True)),
                "bonuses": dict(zip(GEM_COLORS, map(int, player.bonuses), strict=True)),
                "purchased_card_count": len(player.purchased),
                "purchased_cards": [card.to_dict() for card in player.purchased],
                "reserved_cards": [{**r.card.to_dict(), "origin": r.origin, "hidden_to_opponents": r.hidden_to_opponents} for r in player.reserved],
                "nobles": [n.to_dict() for n in player.nobles],
            })
        return {"ranking": self.final_ranking(), "winner_ids": self.winner_ids(), "players": players, "state_hash": self.state_hash()}

    def render_final_summary(self, omniscient: bool = True) -> str:
        del omniscient
        lines = ["Final Result", "============", ""]
        for group in self.final_ranking():
            for player_id in group["players"]:
                p = self.players[player_id]
                lines.append(f"Rank {group['rank']}: P{player_id} — {p.score} points, {len(p.purchased)} purchased cards")
        for i, player in enumerate(self.players):
            lines += ["", f"P{i} purchased cards:"] + [f"  {c.short_text()}" for c in player.purchased]
            lines += [f"P{i} reserved cards:"] + [f"  {r.origin}: {r.card.short_text()}" for r in player.reserved]
            lines += [f"P{i} nobles:"] + [f"  {n.noble_id}: {n.points} pt, requires {n.requirements_by_color()}" for n in player.nobles]
        return "\n".join(lines)

    def render(self, perspective: int | None = None, *, omniscient: bool = False, use_color: bool = False) -> str:
        """Render a deterministic plain-text, perspective-safe game view."""
        del use_color
        perspective = self.current_player if perspective is None else perspective
        if not 0 <= perspective < self.num_players:
            raise ValueError("invalid perspective")
        def card_text(card: Card | None) -> str:
            return "[empty]" if card is None else card.short_text()
        lines = [
            f"Splendor | view=P{perspective} | round={self.round_id} | turn={self.turns_completed} | phase={self.phase.value} | player={self.current_player}",
            "Bank  " + " ".join(f"{name}:{int(self.bank[i])}" for i, name in enumerate(TOKEN_COLORS)),
        ]
        for tier in reversed(range(3)):
            lines.append(f"Tier {tier + 1} ({len(self.decks[tier])} hidden): " + " | ".join(card_text(c) for c in self.visible[tier]))
        lines.append("Nobles: " + " | ".join(f"{n.noble_id}:{n.requirements_by_color()}" for n in self.nobles))
        if self.last_action_text:
            lines.append(f"Last action: P{self.last_actor} {self.last_action_text}")
        for i, player in enumerate(self.players):
            current = " <- current" if i == self.current_player else ""
            lines.append(f"P{i}{current}: score={player.score}, cards={len(player.purchased)}, bonuses={player.bonuses.tolist()}, tokens={player.tokens.tolist()}, reserved={len(player.reserved)}, nobles={len(player.nobles)}")
            if not player.reserved:
                lines.append("  reserved: (none)")
            for reservation in player.reserved:
                if not self._reservation_visible(i, perspective, reservation, omniscient):
                    lines.append(f"  reserved: [Tier {reservation.card.tier + 1} hidden card]")
                else:
                    label = "deck-private" if reservation.hidden_to_opponents else "visible-public"
                    lines.append(f"  reserved: [{label}] {reservation.card.short_text()}")
        if self.done:
            lines.append(f"END reason={self.end_reason}, winners={self.winner_ids()}, rewards={self.terminal_rewards().tolist()}")
        return "\n".join(lines)

    def validate_invariants(self) -> None:
        """Validate card, noble, token, score, bonus, and phase conservation."""
        if np.any(self.bank < 0):
            raise AssertionError(f"negative bank tokens: {self.bank}")
        expected = np.asarray([self.initial_colored_token_count] * 5 + [5])
        actual = self.bank.copy()
        card_ids: list[str] = []
        for deck in self.decks:
            card_ids += [c.card_id for c in deck]
        for row in self.visible:
            card_ids += [c.card_id for c in row if c is not None]
        for tier, row in enumerate(self.visible):
            if self.decks[tier] and any(card is None for card in row):
                raise AssertionError("empty market slot while tier deck remains")
        for player in self.players:
            if np.any(player.tokens < 0):
                raise AssertionError("negative player tokens")
            if len(player.reserved) > MAX_RESERVED:
                raise AssertionError("too many reserved cards")
            if self.phase not in {Phase.DISCARD, Phase.TERMINAL} and player.token_count > MAX_TOKENS:
                raise AssertionError("player above token limit outside discard")
            actual += player.tokens
            card_ids += [c.card_id for c in player.purchased]
            card_ids += [r.card.card_id for r in player.reserved]
            score = sum(c.points for c in player.purchased) + sum(n.points for n in player.nobles)
            if player.score != score:
                raise AssertionError("score does not match holdings")
            bonuses = np.zeros(5, dtype=np.int16)
            for card in player.purchased:
                bonuses[card.bonus] += 1
            if not np.array_equal(player.bonuses, bonuses):
                raise AssertionError("bonuses do not match purchased cards")
        if not np.array_equal(actual, expected):
            raise AssertionError(f"token conservation failed: {actual} != {expected}")
        if len(card_ids) != 90 or len(set(card_ids)) != 90:
            raise AssertionError("development-card conservation failed")
        noble_ids = [n.noble_id for n in self.nobles + self.unused_nobles]
        for player in self.players:
            noble_ids += [n.noble_id for n in player.nobles]
        if len(noble_ids) != 10 or len(set(noble_ids)) != 10:
            raise AssertionError("noble conservation failed")

    _assert_invariants = validate_invariants
