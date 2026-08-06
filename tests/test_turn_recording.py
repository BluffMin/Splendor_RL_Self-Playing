from splendor_env.core import SplendorGame
from splendor_env.event_schema import build_turn_records


def test_decisions_group_into_one_turn():
    game = SplendorGame(2, seed=4)
    events = []
    game.add_event_listener(events.append)
    while not any(e["phase_before"] == "payment" for e in events):
        game.step(game.legal_actions()[0])
    records = build_turn_records(events)
    assert records[-1].completed
    assert records[-1].primary_action_type not in {"payment", "choose_payment_plan"}
    assert len(records[-1].decision_ids) >= 2
