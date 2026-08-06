import pytest

from splendor_rl.league.promotion import regression_decision


def test_regression_gate_passes_and_fails():
    assert regression_decision({"a": 0.5}, {"a": 0.52})["passed"]
    failed = regression_decision({"a": 0.4, "b": 0.5}, {"a": 0.6, "b": 0.5})
    assert not failed["passed"] and failed["max_single_regression"] == pytest.approx(
        0.2
    )
