"""Rules engine and vector encoder for the original Splendor base game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .actions import (
    ACTIONS,
    COLORS,
    N_ACTIONS,
    PAYMENT_OFFSET,
    TOKEN_COLORS,
    describe_action,
)
from .data import CARDS, NOBLES, Card, Noble

Phase = Literal["normal", "payment", "discard", "noble"]

MAX_PLAYERS = 4
MAX_RESERVED = 3
MAX_NOBLES = 5
WINNING_SCORE = 15
MAX_TOKENS = 10
OBSERVATION_SIZE = 419


class InvalidActionError(ValueError):
    """Raised when an action is outside the current legal-action mask."""


@dataclass(slots=True)
class Reservation:
    card: Card
    hidden_to_opponents: bool


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


class SplendorGame:
    """Deterministic-by-seed Splendor game engine.

    The engine is independent of Gymnasium/PettingZoo so it can be unit-tested
    and used for high-throughput simulation directly. The PettingZoo adapter is
    defined in :mod:`splendor_env.pettingzoo_env`.
    """

    def __init__(
        self,
        num_players: int = 2,
        *,
        max_turns: int | None = None,
        allow_deadlock_pass: bool = False,
        seed: int | None = None,
    ) -> None:
        if not 2 <= num_players <= 4:
            raise ValueError("num_players must be between 2 and 4")
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.num_players = int(num_players)
        self.max_turns = int(max_turns) if max_turns is not None else None
        self.allow_deadlock_pass = bool(allow_deadlock_pass)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @property
    def initial_colored_token_count(self) -> int:
        return {2: 4, 3: 5, 4: 7}[self.num_players]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)

        self.players = [PlayerState() for _ in range(self.num_players)]
        self.bank = np.array(
            [self.initial_colored_token_count] * 5 + [5], dtype=np.int16
        )

        self.decks: list[list[Card]] = []
        for tier in range(3):
            deck = [card for card in CARDS if card.tier == tier]
            self.rng.shuffle(deck)
            self.decks.append(deck)

        self.visible: list[list[Card | None]] = [[None] * 4 for _ in range(3)]
        for tier in range(3):
            for slot in range(4):
                self.visible[tier][slot] = self._draw(tier)

        noble_indices = self.rng.choice(
            len(NOBLES), size=self.num_players + 1, replace=False
        )
        self.nobles = [NOBLES[int(i)] for i in noble_indices]

        self.current_player = 0
        self.phase: Phase = "normal"
        self.turns_completed = 0
        self.final_round_triggered = False
        self.trigger_player: int | None = None
        self.terminated = False
        self.truncated = False
        self.end_reason: str | None = None
        self.last_action: int | None = None
        self.last_actor: int | None = None
        self.consecutive_passes = 0
        self.pending_purchase: tuple[str, int, Card] | None = None
        self._assert_invariants()

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

    def _payment(self, player: PlayerState, card: Card) -> tuple[np.ndarray, int]:
        remaining = np.maximum(np.asarray(card.cost, dtype=np.int16) - player.bonuses, 0)
        colored = np.minimum(player.tokens[:5], remaining)
        gold = int((remaining - colored).sum())
        return colored, gold

    def can_afford(self, player_index: int, card: Card) -> bool:
        colored, gold = self._payment(self.players[player_index], card)
        del colored
        return int(self.players[player_index].tokens[5]) >= gold

    def _legal_payment(self, player: PlayerState, card: Card, allocation: tuple[int, ...]) -> bool:
        remaining = np.maximum(np.asarray(card.cost, dtype=np.int16) - player.bonuses, 0)
        gold_by_color = np.asarray(allocation, dtype=np.int16)
        colored = remaining - gold_by_color
        return bool(
            np.all(gold_by_color <= remaining)
            and np.all(colored >= 0)
            and np.all(colored <= player.tokens[:5])
            and int(gold_by_color.sum()) <= int(player.tokens[5])
        )

    def eligible_noble_slots(self, player_index: int | None = None) -> list[int]:
        if player_index is None:
            player_index = self.current_player
        bonuses = self.players[player_index].bonuses
        return [
            i
            for i, noble in enumerate(self.nobles)
            if np.all(bonuses >= np.asarray(noble.cost, dtype=np.int16))
        ]

    def legal_action_mask(self, player_index: int | None = None) -> np.ndarray:
        """Return the legal-action mask for the current decision state."""
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if self.done:
            return mask
        if player_index is not None and player_index != self.current_player:
            return mask

        player = self.players[self.current_player]

        if self.phase == "payment":
            if self.pending_purchase is None:
                raise RuntimeError("payment phase has no pending purchase")
            card = self.pending_purchase[2]
            for action in range(PAYMENT_OFFSET, N_ACTIONS):
                if self._legal_payment(player, card, ACTIONS[action].payload):
                    mask[action] = 1
            return mask

        if self.phase == "discard":
            for color in range(6):
                if player.tokens[color] > 0:
                    mask[60 + color] = 1
            return mask

        if self.phase == "noble":
            for slot in self.eligible_noble_slots():
                mask[66 + slot] = 1
            return mask

        available = [i for i in range(5) if self.bank[i] > 0]
        if available:
            required_size = min(3, len(available))
            for action_id, spec in enumerate(ACTIONS[:25]):
                if len(spec.payload) == required_size and all(
                    self.bank[color] > 0 for color in spec.payload
                ):
                    mask[action_id] = 1

        for color in range(5):
            if self.bank[color] >= 4:
                mask[25 + color] = 1

        for flat_slot in range(12):
            card = self._visible_card(flat_slot)
            if card is not None and self.can_afford(self.current_player, card):
                mask[30 + flat_slot] = 1

        can_reserve = len(player.reserved) < MAX_RESERVED
        if can_reserve:
            for flat_slot in range(12):
                if self._visible_card(flat_slot) is not None:
                    mask[42 + flat_slot] = 1
            for tier in range(3):
                if self.decks[tier]:
                    mask[54 + tier] = 1

        for slot, reservation in enumerate(player.reserved):
            if self.can_afford(self.current_player, reservation.card):
                mask[57 + slot] = 1

        if not mask.any() and self.allow_deadlock_pass:
            mask[71] = 1
        return mask

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.legal_action_mask()).astype(int).tolist()

    def step(self, action: int) -> EngineStep:
        if self.done:
            raise RuntimeError("cannot step a finished game")
        try:
            action = int(action)
        except (TypeError, ValueError) as exc:
            raise InvalidActionError(f"action must be an integer, got {action!r}") from exc
        if not 0 <= action < N_ACTIONS:
            raise InvalidActionError(f"action {action} is outside [0, {N_ACTIONS})")

        mask = self.legal_action_mask()
        if not mask[action]:
            legal = ", ".join(
                f"{a}: {describe_action(a)}" for a in np.flatnonzero(mask).tolist()
            )
            raise InvalidActionError(
                f"illegal action {action} ({describe_action(action)}); legal actions: {legal}"
            )

        actor = self.current_player
        score_before = self.players[actor].score
        turns_before = self.turns_completed
        spec = ACTIONS[action]
        if self.phase == "normal" and spec.kind != "pass":
            self.consecutive_passes = 0

        if self.phase == "payment":
            self._complete_purchase(tuple(spec.payload))
        elif self.phase == "discard":
            self._discard_one(int(spec.payload))
        elif self.phase == "noble":
            self._choose_noble(int(spec.payload))
        elif spec.kind == "pass":
            self._pass_turn()
        elif spec.kind == "take_distinct":
            self._take_distinct(spec.payload)
        elif spec.kind == "take_double":
            self._take_double(int(spec.payload))
        elif spec.kind == "buy_visible":
            self._begin_purchase("visible", int(spec.payload), self._visible_card(int(spec.payload)))
        elif spec.kind == "reserve_visible":
            card = self._remove_visible(int(spec.payload))
            self._reserve(card, hidden=False)
            self._after_normal_action()
        elif spec.kind == "reserve_deck":
            tier = int(spec.payload)
            card = self.decks[tier].pop()
            self._reserve(card, hidden=True)
            self._after_normal_action()
        elif spec.kind == "buy_reserved":
            slot = int(spec.payload)
            self._begin_purchase("reserved", slot, self.players[actor].reserved[slot].card)
        else:
            raise RuntimeError(f"unexpected action for phase {self.phase}: {spec}")

        self.last_action = action
        self.last_actor = actor
        self._assert_invariants()
        return EngineStep(
            actor=actor,
            action=action,
            score_delta=self.players[actor].score - score_before,
            turn_ended=self.turns_completed > turns_before,
            terminated=self.terminated,
            truncated=self.truncated,
            phase=self.phase,
        )


    def _pass_turn(self) -> None:
        """Advance only when the official action set is empty.

        If every player passes consecutively, the position is considered a
        deadlock and the episode is truncated using the current standings.
        """
        actor = self.current_player
        self.consecutive_passes += 1
        self.turns_completed += 1
        self.phase = "normal"
        if self.consecutive_passes >= self.num_players:
            self.truncated = True
            self.end_reason = "deadlock"
            return
        if self.max_turns is not None and self.turns_completed >= self.max_turns:
            self.truncated = True
            self.end_reason = "max_turns"
            return
        self.current_player = (actor + 1) % self.num_players

    def _take_distinct(self, colors: Iterable[int]) -> None:
        player = self.players[self.current_player]
        for color in colors:
            self.bank[color] -= 1
            player.tokens[color] += 1
        self._after_normal_action()

    def _take_double(self, color: int) -> None:
        player = self.players[self.current_player]
        self.bank[color] -= 2
        player.tokens[color] += 2
        self._after_normal_action()

    def _reserve(self, card: Card, *, hidden: bool) -> None:
        player = self.players[self.current_player]
        player.reserved.append(
            Reservation(card=card, hidden_to_opponents=bool(hidden))
        )
        # Reserving remains legal even when the bank has no gold; the player
        # simply receives no gold in that case.
        if self.bank[5] > 0:
            self.bank[5] -= 1
            player.tokens[5] += 1

    def _begin_purchase(self, source: str, slot: int, card: Card | None) -> None:
        if card is None:
            raise RuntimeError("attempted to buy an empty card slot")
        self.pending_purchase = (source, slot, card)
        self.phase = "payment"

    def _complete_purchase(self, allocation: tuple[int, ...]) -> None:
        if self.pending_purchase is None:
            raise RuntimeError("no purchase is pending")
        source, slot, card = self.pending_purchase
        player = self.players[self.current_player]
        if not self._legal_payment(player, card, allocation):
            raise RuntimeError("selected payment is no longer legal")
        remaining = np.maximum(np.asarray(card.cost, dtype=np.int16) - player.bonuses, 0)
        gold = int(sum(allocation))
        colored = remaining - np.asarray(allocation, dtype=np.int16)
        if source == "visible":
            removed = self._remove_visible(slot)
        else:
            removed = player.reserved.pop(slot).card
        if removed.card_id != card.card_id:
            raise RuntimeError("pending purchase card changed")
        player.tokens[:5] -= colored
        self.bank[:5] += colored
        player.tokens[5] -= gold
        self.bank[5] += gold
        player.bonuses[card.bonus] += 1
        player.purchased.append(card)
        player.score += card.points
        self.pending_purchase = None
        self._after_normal_action()

    def _discard_one(self, color: int) -> None:
        player = self.players[self.current_player]
        player.tokens[color] -= 1
        self.bank[color] += 1
        if player.token_count <= MAX_TOKENS:
            self._resolve_noble_or_end_turn()

    def _choose_noble(self, slot: int) -> None:
        if slot not in self.eligible_noble_slots():
            raise RuntimeError("noble eligibility changed unexpectedly")
        self._award_noble(slot)
        self._end_turn()

    def _after_normal_action(self) -> None:
        if self.players[self.current_player].token_count > MAX_TOKENS:
            self.phase = "discard"
        else:
            self._resolve_noble_or_end_turn()

    def _resolve_noble_or_end_turn(self) -> None:
        eligible = self.eligible_noble_slots()
        if not eligible:
            self._end_turn()
        elif len(eligible) == 1:
            self._award_noble(eligible[0])
            self._end_turn()
        else:
            self.phase = "noble"

    def _award_noble(self, slot: int) -> None:
        noble = self.nobles.pop(slot)
        player = self.players[self.current_player]
        player.nobles.append(noble)
        player.score += noble.points

    def _end_turn(self) -> None:
        actor = self.current_player
        self.turns_completed += 1
        self.phase = "normal"

        if not self.final_round_triggered and self.players[actor].score >= WINNING_SCORE:
            self.final_round_triggered = True
            self.trigger_player = actor

        if self.final_round_triggered and actor == self.num_players - 1:
            self.terminated = True
            self.end_reason = "score_threshold"
            return

        if self.max_turns is not None and self.turns_completed >= self.max_turns:
            self.truncated = True
            self.end_reason = "max_turns"
            return

        self.current_player = (actor + 1) % self.num_players

    def winners(self) -> list[int]:
        """Winner indices using score, then fewest purchased development cards."""
        best_score = max(player.score for player in self.players)
        candidates = [
            i for i, player in enumerate(self.players) if player.score == best_score
        ]
        fewest_cards = min(len(self.players[i].purchased) for i in candidates)
        return [i for i in candidates if len(self.players[i].purchased) == fewest_cards]

    def terminal_rewards(self) -> np.ndarray:
        """Return zero-sum terminal rewards; all-zero before the game ends."""
        rewards = np.zeros(self.num_players, dtype=np.float32)
        if not self.done:
            return rewards
        winners = self.winners()
        if len(winners) == self.num_players:
            return rewards
        rewards[winners] = 1.0
        loser_penalty = -len(winners) / (self.num_players - len(winners))
        for i in range(self.num_players):
            if i not in winners:
                rewards[i] = loser_penalty
        return rewards

    def observation(self, perspective: int, *, omniscient: bool = False) -> np.ndarray:
        if not 0 <= perspective < self.num_players:
            raise ValueError("invalid perspective")
        values: list[float] = []

        values.extend(float(self.phase == phase) for phase in ("normal", "payment", "discard", "noble"))
        values.append(float(self.done))
        values.append(float(self.final_round_triggered))
        values.append(
            min(self.turns_completed / self.max_turns, 1.0)
            if self.max_turns is not None
            else 0.0
        )

        bank_den = np.asarray([7, 7, 7, 7, 7, 5], dtype=np.float32)
        values.extend((self.bank.astype(np.float32) / bank_den).tolist())
        values.extend(
            [
                len(self.decks[0]) / 40.0,
                len(self.decks[1]) / 30.0,
                len(self.decks[2]) / 20.0,
            ]
        )

        for tier in range(3):
            for slot in range(4):
                values.extend(self._encode_card(self.visible[tier][slot]))

        for slot in range(MAX_NOBLES):
            noble = self.nobles[slot] if slot < len(self.nobles) else None
            values.extend(self._encode_noble(noble))

        order = [
            (perspective + offset) % self.num_players
            for offset in range(self.num_players)
        ]
        for player_slot in range(MAX_PLAYERS):
            if player_slot >= self.num_players:
                values.extend([0.0] * 56)
                continue
            player_index = order[player_slot]
            player = self.players[player_index]
            values.extend(
                [
                    1.0,
                    float(player_index == self.current_player and not self.done),
                ]
            )
            values.extend(np.clip(player.tokens.astype(np.float32) / 13.0, 0, 1).tolist())
            values.extend(np.clip(player.bonuses.astype(np.float32) / 20.0, 0, 1).tolist())
            values.append(min(player.score / 30.0, 1.0))
            values.append(min(len(player.purchased) / 30.0, 1.0))
            values.append(min(len(player.nobles) / 5.0, 1.0))
            values.append(len(player.reserved) / 3.0)

            for reserved_slot in range(MAX_RESERVED):
                if reserved_slot >= len(player.reserved):
                    values.extend([0.0] * 13)
                    continue
                reservation = player.reserved[reserved_slot]
                hidden = reservation.hidden_to_opponents
                can_see = omniscient or player_index == perspective or not hidden
                if can_see:
                    values.extend(
                        [1.0, float(hidden)]
                        + self._encode_card(reservation.card)[1:]
                    )
                else:
                    values.extend([1.0, 1.0] + [0.0] * 11)

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                f"encoder produced {observation.shape}, expected {(OBSERVATION_SIZE,)}"
            )
        return np.clip(observation, 0.0, 1.0)

    @staticmethod
    def _encode_card(card: Card | None) -> list[float]:
        # present, points, 5-way bonus, 5 costs = 12 values
        if card is None:
            return [0.0] * 12
        bonus = [0.0] * 5
        bonus[card.bonus] = 1.0
        return (
            [1.0, card.points / 5.0]
            + bonus
            + [cost / 7.0 for cost in card.cost]
        )

    @staticmethod
    def _encode_noble(noble: Noble | None) -> list[float]:
        # present, points, 5 requirements = 7 values
        if noble is None:
            return [0.0] * 7
        return [1.0, noble.points / 3.0] + [cost / 4.0 for cost in noble.cost]

    def state(self) -> np.ndarray:
        """Omniscient global state for centralized critics."""
        return self.observation(0, omniscient=True)

    def render(self) -> str:
        def card_text(card: Card | None) -> str:
            if card is None:
                return "[empty]"
            bonus = COLORS[card.bonus][0].upper()
            cost = "/".join(str(x) for x in card.cost)
            return f"{bonus}+ {card.points}pt ({cost})"

        lines = [
            f"Splendor | phase={self.phase} | player={self.current_player} | turns={self.turns_completed}",
            "Bank  " + " ".join(f"{name}:{int(self.bank[i])}" for i, name in enumerate(TOKEN_COLORS)),
        ]
        for tier in reversed(range(3)):
            lines.append(
                f"Tier {tier + 1} ({len(self.decks[tier])} hidden): "
                + " | ".join(card_text(card) for card in self.visible[tier])
            )
        lines.append(
            "Nobles: "
            + " | ".join("/".join(str(x) for x in noble.cost) for noble in self.nobles)
        )
        for i, player in enumerate(self.players):
            lines.append(
                f"P{i}: score={player.score}, cards={len(player.purchased)}, "
                f"bonuses={player.bonuses.tolist()}, tokens={player.tokens.tolist()}, "
                f"reserved={len(player.reserved)}, nobles={len(player.nobles)}"
            )
        if self.done:
            lines.append(
                f"END reason={self.end_reason}, winners={self.winners()}, rewards={self.terminal_rewards().tolist()}"
            )
        return "\n".join(lines)

    def _assert_invariants(self) -> None:
        if np.any(self.bank < 0):
            raise AssertionError(f"negative bank tokens: {self.bank}")
        for player in self.players:
            if np.any(player.tokens < 0):
                raise AssertionError(f"negative player tokens: {player.tokens}")
            if len(player.reserved) > MAX_RESERVED:
                raise AssertionError("too many reserved cards")
            if self.phase == "normal" and not self.done and player.token_count > MAX_TOKENS:
                # Only the current player may exceed the limit, and only while discarding.
                raise AssertionError("player above token limit outside discard phase")

        expected = np.asarray(
            [self.initial_colored_token_count] * 5 + [5], dtype=np.int16
        )
        actual = self.bank.copy()
        for player in self.players:
            actual += player.tokens
        if not np.array_equal(actual, expected):
            raise AssertionError(f"token conservation failed: {actual} != {expected}")

        all_card_ids: list[int] = []
        for deck in self.decks:
            all_card_ids.extend(card.card_id for card in deck)
        for tier in self.visible:
            all_card_ids.extend(card.card_id for card in tier if card is not None)
        for player in self.players:
            all_card_ids.extend(card.card_id for card in player.purchased)
            all_card_ids.extend(res.card.card_id for res in player.reserved)
        if len(all_card_ids) != 90 or len(set(all_card_ids)) != 90:
            raise AssertionError("development-card conservation failed")

        noble_ids = [noble.noble_id for noble in self.nobles]
        for player in self.players:
            noble_ids.extend(noble.noble_id for noble in player.nobles)
        if len(noble_ids) != self.num_players + 1 or len(set(noble_ids)) != len(noble_ids):
            raise AssertionError("noble conservation failed")
