from __future__ import annotations

from copy import deepcopy

from splendor_env.core import SplendorGame
from splendor_env.visualization.view_model import compute_state_delta


def test_delta_detects_resources_and_holdings() -> None:
    game = SplendorGame(2, seed=8)
    before = game.to_state_dict()
    after = deepcopy(before)
    after["bank"][0] -= 1
    after["players"][0]["tokens"][0] += 1
    after["players"][0]["score"] += 3
    after["players"][0]["purchased"].append("T1-WHITE-01")
    after["players"][0]["reserved"].append(
        {"card_id": "T2-BLUE-01", "origin": "visible"}
    )
    after["players"][0]["nobles"].append("N-01")
    after["visible"][0][0] = "CHANGED-CARD"
    delta = compute_state_delta(before, after)
    assert delta.token_changes["white"] == -1
    assert delta.player_token_changes[0]["white"] == 1
    assert delta.player_score_changes[0] == 3
    assert delta.purchased_card_ids == ("T1-WHITE-01",)
    assert delta.reserved_card_ids == ("T2-BLUE-01",)
    assert delta.acquired_noble_ids == ("N-01",)
    assert delta.changed_market_slots == (0,)


def test_no_resource_delta_for_identical_states() -> None:
    state = SplendorGame(2, seed=9).to_state_dict()
    delta = compute_state_delta(state, deepcopy(state))
    assert not delta.changed_market_slots
    assert not delta.token_changes
    assert not delta.player_token_changes
