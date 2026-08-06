from __future__ import annotations

import numpy as np
import pytest

from splendor_env.actions import PAYMENT_OFFSET, action_id
from splendor_env.core import MAX_TOKENS, OBSERVATION_SIZE, SplendorGame
from splendor_env.data import CARDS, NOBLES


def test_data_sizes() -> None:
    assert len(CARDS) == 90
    assert [sum(card.tier == tier for card in CARDS) for tier in range(3)] == [40, 30, 20]
    assert len(NOBLES) == 10
    assert any(card.tier == 1 and card.bonus == 1 and card.points == 2 and card.cost == (1, 0, 0, 2, 4) for card in CARDS)
    assert any(card.tier == 1 and card.bonus == 2 and card.points == 2 and card.cost == (0, 2, 0, 4, 1) for card in CARDS)


@pytest.mark.parametrize("num_players, colored", [(2, 4), (3, 5), (4, 7)])
def test_setup(num_players: int, colored: int) -> None:
    game = SplendorGame(num_players=num_players, seed=0)
    assert game.bank.tolist() == [colored] * 5 + [5]
    assert [len(deck) for deck in game.decks] == [36, 26, 16]
    assert len(game.nobles) == num_players + 1
    assert game.observation(0).shape == (OBSERVATION_SIZE,)
    assert game.state().shape == (OBSERVATION_SIZE,)


def test_forced_discard_is_agent_decision() -> None:
    game = SplendorGame(num_players=2, seed=1)
    game.players[0].tokens[:5] = 2
    game.bank[:5] -= 2
    assert game.players[0].token_count == MAX_TOKENS

    take_three = action_id("take_distinct", (0, 1, 2))
    assert game.legal_action_mask()[take_three]
    game.step(take_three)
    assert game.phase == "discard"
    assert game.current_player == 0
    assert game.players[0].token_count == 13

    for color in (0, 1, 2):
        discard = action_id("discard_one", color)
        assert game.legal_action_mask()[discard]
        game.step(discard)

    assert game.players[0].token_count == 10
    assert game.phase == "normal"
    assert game.current_player == 1


def test_gold_payment_computation() -> None:
    game = SplendorGame(num_players=2, seed=2)
    player = game.players[0]
    card = CARDS[0]  # cost total = 3
    player.tokens[5] = 3
    game.bank[5] = 2
    colored, gold = game._payment(player, card)
    assert colored.sum() == 0
    assert gold == 3
    assert game.can_afford(0, card)


def test_player_can_choose_gold_instead_of_owned_color() -> None:
    game = SplendorGame(num_players=2, seed=2)
    player = game.players[0]
    card = game.visible[0][0]
    assert card is not None
    required = np.asarray(card.cost, dtype=np.int16)
    player.tokens[:5] = required
    player.tokens[5] = 1
    game.bank[:5] -= required
    game.bank[5] -= 1
    game._begin_purchase("visible", 0, card)

    color = int(np.flatnonzero(required)[0])
    allocation = [0] * 5
    allocation[color] = 1
    payment_action = action_id("choose_payment", tuple(allocation))
    assert payment_action >= PAYMENT_OFFSET
    assert game.legal_action_mask()[payment_action]
    gold_before = int(player.tokens[5])
    color_before = int(player.tokens[color])
    game.step(payment_action)
    assert player.tokens[5] == gold_before - 1
    assert player.tokens[color] == color_before - required[color] + 1


def test_random_rollouts_preserve_invariants() -> None:
    for num_players in (2, 3, 4):
        for seed in range(8):
            game = SplendorGame(
                num_players=num_players,
                seed=seed,
                max_turns=250,
                allow_deadlock_pass=True,
            )
            rng = np.random.default_rng(seed + 1000)
            substeps = 0
            while not game.done:
                legal = game.legal_actions()
                assert legal
                game.step(int(rng.choice(legal)))
                substeps += 1
                assert substeps < 5000
            assert game.winners()
            rewards = game.terminal_rewards()
            assert rewards.shape == (num_players,)
            assert abs(float(rewards.sum())) < 1e-6


def test_hidden_reservation_is_masked_from_opponent() -> None:
    game = SplendorGame(num_players=2, seed=3)
    action = action_id("reserve_deck", 0)
    game.step(action)
    assert len(game.players[0].reserved) == 1
    assert game.players[0].reserved[0].hidden_to_opponents

    own = game.observation(0)
    opponent = game.observation(1)
    # Payment phase adds one global feature; player blocks begin at index 195.
    own_reserved_start = 195 + 17
    opponent_view_of_p0_start = 195 + 56 + 17
    assert own[own_reserved_start] == 1.0
    assert own[own_reserved_start + 1] == 1.0
    assert own[own_reserved_start + 2 : own_reserved_start + 13].sum() > 0
    assert opponent[opponent_view_of_p0_start] == 1.0
    assert opponent[opponent_view_of_p0_start + 1] == 1.0
    assert opponent[opponent_view_of_p0_start + 2 : opponent_view_of_p0_start + 13].sum() == 0
