from splendor_rl.league.promotion import bootstrap_confidence_interval


def test_bootstrap_is_deterministic_and_contains_mean():
    result = bootstrap_confidence_interval(
        [0, 0.5, 1, 1], samples=500, confidence=0.95, seed=7
    )
    assert result == bootstrap_confidence_interval(
        [0, 0.5, 1, 1], samples=500, confidence=0.95, seed=7
    )
    assert (
        result["lower_confidence_bound"]
        <= result["mean_score"]
        <= result["upper_confidence_bound"]
    )
