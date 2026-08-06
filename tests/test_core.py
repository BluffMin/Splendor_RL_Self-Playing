from __future__ import annotations

import json

import numpy as np
import pytest

from splendor_env.actions import (
    DISCARD_OFFSET,
    GEM_COLORS,
    N_ACTIONS,
    PAYMENT_OFFSET,
    action_id,
)
from splendor_env.agents import GreedyAgent
from splendor_env.core import (
    GLOBAL_OBSERVATION_SIZE,
    MAX_TOKENS,
    OBSERVATION_SIZE,
    PLAYER_BLOCK_SIZE,
    PLAYER_PUBLIC_SIZE,
    RESERVED_SLOT_SIZE,
    Phase,
    PlayerState,
    SplendorGame,
    enumerate_legal_discard_plans,
    enumerate_legal_payment_plans,
)
from splendor_env.data import CARDS, NOBLES, Card
from splendor_env.recording import EpisodeRecorder, append_games_summary_csv
from splendor_env.replay import (
    ReplayVerificationError,
    load_recording,
    verify_recording,
)


def test_card_and_noble_data_are_structurally_complete() -> None:
    assert GEM_COLORS == ("white", "blue", "green", "red", "black")
    assert len(CARDS) == 90
    assert [sum(c.tier == tier for c in CARDS) for tier in range(3)] == [40, 30, 20]
    assert len({c.card_id for c in CARDS}) == 90
    for tier, expected in enumerate((8, 6, 4)):
        assert [sum(c.tier == tier and c.bonus_color == color for c in CARDS) for color in GEM_COLORS] == [expected] * 5
    white = [c for c in CARDS if c.tier == 1 and c.bonus_color == "white" and c.points == 2 and c.cost == (0, 0, 1, 4, 2)]
    blue = [c for c in CARDS if c.tier == 1 and c.bonus_color == "blue" and c.points == 2 and c.cost == (2, 0, 0, 1, 4)]
    assert len(white) == len(blue) == 1
    assert len(NOBLES) == 10
    assert all(n.points == 3 for n in NOBLES)
    assert len({n.noble_id for n in NOBLES}) == 10
    assert len({n.requirements for n in NOBLES}) == 10
    assert sorted(sum(x > 0 for x in n.requirements) for n in NOBLES) == [2] * 5 + [3] * 5


@pytest.mark.parametrize("num_players,colored", [(2, 4), (3, 5), (4, 7)])
def test_setup_and_observation(num_players: int, colored: int) -> None:
    game = SplendorGame(num_players, seed=0)
    assert game.bank.tolist() == [colored] * 5 + [5]
    assert [len(deck) for deck in game.decks] == [36, 26, 16]
    assert len(game.nobles) == num_players + 1
    assert N_ACTIONS == 373
    assert OBSERVATION_SIZE == 475
    for perspective in range(num_players):
        assert game.observation(perspective).shape == (475,)
    assert game.state().shape == (475,)
    unused = GLOBAL_OBSERVATION_SIZE + num_players * PLAYER_BLOCK_SIZE
    assert np.count_nonzero(game.observation(0)[unused:]) == 0
    game.validate_invariants()


def make_card(cost: tuple[int, int, int, int, int]) -> Card:
    return Card("TEST", 0, 0, "white", cost)


def test_payment_plan_enumeration_is_complete_and_deterministic() -> None:
    player = PlayerState()
    player.tokens[:] = [2, 1, 0, 0, 0, 2]
    card = make_card((2, 1, 0, 0, 0))
    plans = enumerate_legal_payment_plans(player, card)
    assert plans == enumerate_legal_payment_plans(player, card)
    assert len(plans) == len(set(plans))
    assert plans[0].total_gold == 0
    assert any(p.colored[0] == 1 and p.gold_by_color[0] == 1 for p in plans)
    assert any(p.gold_by_color[0] == 1 and p.gold_by_color[1] == 1 for p in plans)
    for plan in plans:
        assert tuple(plan.colored[i] + plan.gold_by_color[i] for i in range(5)) == card.cost
    poor = PlayerState()
    assert enumerate_legal_payment_plans(poor, make_card((1, 0, 0, 0, 0))) == ()


@pytest.mark.parametrize("total,excess", [(11, 1), (12, 2), (13, 3)])
def test_discard_plan_enumeration(total: int, excess: int) -> None:
    player = PlayerState()
    player.tokens[:] = [2, 2, 2, 2, 2, total - 10]
    plans = enumerate_legal_discard_plans(player, excess)
    assert plans and len(plans) == len(set(plans))
    assert any(plan.tokens[5] > 0 for plan in plans)
    for plan in plans:
        assert sum(plan.tokens) == excess
        assert all(plan.tokens[i] <= player.tokens[i] for i in range(6))


def test_forced_discard_is_one_combination_decision() -> None:
    game = SplendorGame(2, seed=1)
    game.players[0].tokens[:5] = 2
    game.bank[:5] -= 2
    game.step(action_id("take_distinct", (0, 1, 2)))
    assert game.phase == Phase.DISCARD
    assert game.players[0].token_count == 13
    assert np.flatnonzero(game.legal_action_mask()).min() == DISCARD_OFFSET
    result = game.step(DISCARD_OFFSET)
    assert result.turn_ended
    assert game.players[0].token_count == MAX_TOKENS
    game.validate_invariants()


def test_action_mask_uses_only_current_phase_region() -> None:
    game = SplendorGame(2, seed=11)
    assert np.flatnonzero(game.legal_action_mask()).max() < PAYMENT_OFFSET
    card = game.visible[0][0]
    assert card is not None
    required = np.asarray(card.cost, dtype=np.int16)
    game.players[0].tokens[:5] = required
    game.bank[:5] -= required
    game.step(action_id("buy_visible", 0))
    legal = np.flatnonzero(game.legal_action_mask())
    assert legal.min() >= PAYMENT_OFFSET and legal.max() < DISCARD_OFFSET


def test_purchase_uses_selected_payment_and_conserves_tokens() -> None:
    game = SplendorGame(2, seed=2)
    card = game.visible[0][0]
    assert card is not None
    required = np.asarray(card.cost, dtype=np.int16)
    game.players[0].tokens[:5] = required
    game.players[0].tokens[5] = 1
    game.bank[:5] -= required
    game.bank[5] -= 1
    game.step(action_id("buy_visible", 0))
    assert game.phase == Phase.PAYMENT
    plan_index = next(i for i, p in enumerate(game.pending_payment_plans) if p.total_gold == 1)
    game.step(PAYMENT_OFFSET + plan_index)
    assert card in game.players[0].purchased
    game.validate_invariants()


def test_reservation_visibility_and_state() -> None:
    game = SplendorGame(2, seed=3)
    game.step(action_id("reserve_deck", 1))
    own = game.observation(0)
    opponent = game.observation(1)
    own_start = GLOBAL_OBSERVATION_SIZE + PLAYER_PUBLIC_SIZE
    other_start = GLOBAL_OBSERVATION_SIZE + PLAYER_BLOCK_SIZE + PLAYER_PUBLIC_SIZE
    assert own[own_start : own_start + 5].tolist() == [1, 1, 0, 1, 0]
    assert own[own_start + 5 : own_start + RESERVED_SLOT_SIZE].sum() > 0
    assert opponent[other_start : other_start + 5].tolist() == [1, 1, 0, 1, 0]
    assert opponent[other_start + 5 : other_start + RESERVED_SLOT_SIZE].sum() == 0
    assert game.state()[own_start + 5 : own_start + RESERVED_SLOT_SIZE].sum() > 0
    assert "[Tier 2 hidden card]" in game.render(perspective=1)


def test_visible_reservation_stays_public() -> None:
    game = SplendorGame(2, seed=4)
    card = game.visible[0][0]
    game.step(action_id("reserve_visible", 0))
    other_start = GLOBAL_OBSERVATION_SIZE + PLAYER_BLOCK_SIZE + PLAYER_PUBLIC_SIZE
    opponent = game.observation(1)
    assert opponent[other_start] == 1 and opponent[other_start + 1] == 0
    assert opponent[other_start + 5 : other_start + 16].sum() > 0
    assert card is game.players[0].reserved[0].card
    assert "[visible-public]" in game.render(perspective=1)


def test_ranking_supports_all_tie_breaks() -> None:
    game = SplendorGame(4, seed=5)
    game.players[0].score = game.players[2].score = 16
    game.players[1].score = 16
    game.players[3].score = 12
    game.players[1].purchased.append(CARDS[0])
    assert game.final_ranking() == [
        {"rank": 1, "players": [0, 2]},
        {"rank": 3, "players": [1]},
        {"rank": 4, "players": [3]},
    ]
    assert game.winner_ids() == [0, 2]
    assert game.is_tied()


def test_noble_auto_award_and_multiple_choice() -> None:
    single = SplendorGame(2, seed=12)
    noble = NOBLES[0]
    single.nobles = [noble]
    single.players[0].bonuses[:] = noble.requirements
    single._resolve_noble_or_end_turn()
    assert single.players[0].nobles == [noble]
    assert single.players[0].score == 3

    multiple = SplendorGame(2, seed=13)
    multiple.nobles = [NOBLES[0], NOBLES[1]]
    multiple.players[0].bonuses[:] = np.maximum(NOBLES[0].requirements, NOBLES[1].requirements)
    multiple._resolve_noble_or_end_turn()
    assert multiple.phase == Phase.NOBLE
    multiple._choose_noble(0)
    assert len(multiple.players[0].nobles) == 1
    assert len(multiple.nobles) == 1


def test_final_round_finishes_equal_turn_count() -> None:
    game = SplendorGame(2, seed=14)
    game.players[0].score = 15
    game._end_turn()
    assert not game.done and game.current_player == 1
    game._end_turn()
    assert game.terminated and game.end_reason == "official_game_end"
    assert game.turns_completed == 2


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_random_rollouts_preserve_invariants(num_players: int) -> None:
    for seed in range(5):
        game = SplendorGame(num_players, seed=seed)
        agent = GreedyAgent()
        while not game.done:
            game.step(agent.act(game))
            game.validate_invariants()
            assert game.decision_id < 5000
        assert game.end_reason == "official_game_end"
        assert game.turns_completed % num_players == 0


def test_recording_round_trip_and_tamper_detection(tmp_path) -> None:
    path = tmp_path / "game.json"
    game = SplendorGame(2, seed=8)
    recorder = EpisodeRecorder(path, "full")
    before = game.observation(1).copy()
    mask_before = game.legal_action_mask().copy()
    recorder.attach(game)
    np.testing.assert_array_equal(before, game.observation(1))
    np.testing.assert_array_equal(mask_before, game.legal_action_mask())
    rng = np.random.default_rng(8)
    while not game.done:
        game.step(int(rng.choice(game.legal_actions())))
    document = recorder.finalize()
    assert path.exists() and document["events"] and document["snapshots"]
    replayed = verify_recording(load_recording(path))
    assert replayed.state_hash() == game.state_hash()
    csv_path = tmp_path / "games.csv"
    append_games_summary_csv(csv_path, "game_0000", 8, game)
    assert csv_path.exists()
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["events"][0]["action_id"] = 999
    with pytest.raises((ReplayVerificationError, ValueError)):
        verify_recording(tampered)
