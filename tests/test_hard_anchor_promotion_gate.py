from splendor_rl.league.promotion import anchor_group_regression_decision

BASE = {"greedy": 0.5, "noble": 0.5, "blocking": 0.5, "random": 0.9, "shortest": 0.8}


def test_hard_aggregate_and_single_regressions_fail():
    aggregate = anchor_group_regression_decision(
        {key: value - 0.03 for key, value in BASE.items()}, BASE
    )
    assert not aggregate["passed"]
    candidate = dict(BASE)
    candidate["blocking"] -= 0.06
    assert (
        "single_hard_anchor_regression"
        in anchor_group_regression_decision(candidate, BASE)["reasons"]
    )


def test_small_saturated_regression_passes_but_large_fails():
    candidate = dict(BASE)
    candidate["random"] -= 0.03
    assert anchor_group_regression_decision(candidate, BASE)["passed"]
    candidate["random"] -= 0.02
    assert not anchor_group_regression_decision(candidate, BASE)["passed"]
