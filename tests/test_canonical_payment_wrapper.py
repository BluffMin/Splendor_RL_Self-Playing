from splendor_env.actions import ACTIONS
from splendor_env.core import Phase, SplendorGame
from splendor_env.wrappers import CanonicalPaymentWrapper


def test_canonical_uses_first_exact_plan_and_hides_payment_decision():
    game = SplendorGame(2, seed=4)
    wrapper = CanonicalPaymentWrapper(game)
    events = []
    game.add_event_listener(events.append)
    while wrapper.canonical_payments == 0:
        legal = game.legal_actions()
        buys = [a for a in legal if ACTIONS[a].kind in {"buy_visible", "buy_reserved"}]
        wrapper.policy_step((buys or legal)[0])
    payments = [e for e in events if e["action_type"] == "canonical_payment"]
    assert (
        payments
        and payments[0]["automatic"]
        and payments[0]["selected_plan_index"] == 0
    )
    expected = game.initial_colored_token_count * 5 + 5
    assert (
        game.phase != Phase.PAYMENT
        and sum(game.bank) + sum(p.token_count for p in game.players) == expected
    )
