import numpy as np

from splendor_rl.population.meta import solve_symmetric_meta_strategy


def test_rps_is_uniform_and_finite():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    result = solve_symmetric_meta_strategy(rps, iterations=10000, seed=1)
    assert (
        np.allclose(result.probabilities, 1 / 3, atol=0.03)
        and np.isfinite(result.probabilities).all()
    )
    assert np.isclose(result.probabilities.sum(), 1)


def test_dominated_policy_gets_low_probability():
    matrix = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], float)
    result = solve_symmetric_meta_strategy(matrix, iterations=10000, seed=1)
    assert result.probabilities[0] > 0.9 and result.probabilities[2] < 0.05
