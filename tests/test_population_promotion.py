from splendor_rl.population.gates import population_promotion_decision


def test_all_population_promotion_gates():
    base = {
        "head_passed": True,
        "anchors_passed": True,
        "meta_delta": 0.01,
        "exploiter_regression": 0.01,
    }
    assert population_promotion_decision(**base)["passed"]
    for change, reason in [
        ({"head_passed": False}, "head_to_head_gate"),
        ({"anchors_passed": False}, "anchor_gate"),
        ({"meta_delta": -0.01}, "meta_strategy_regression"),
        ({"exploiter_regression": 0.04}, "exploiter_robustness_regression"),
    ]:
        result = population_promotion_decision(**(base | change))
        assert reason in result["reasons"]
