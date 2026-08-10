import numpy as np

from splendor_rl.population.meta import antisymmetrize_score_matrix, detect_cycles


def test_antisymmetrization_preserves_raw_and_is_zero_sum():
    raw = np.array([[0.5, 0.7], [0.4, 0.5]])
    copy = raw.copy()
    solved = antisymmetrize_score_matrix(raw)
    assert np.array_equal(raw, copy)
    assert np.allclose(solved, -solved.T)
    assert np.allclose(np.diag(solved), 0)


def test_cycle_diagnostics():
    matrix = np.array([[0.5, 0.6, 0.4], [0.4, 0.5, 0.6], [0.6, 0.4, 0.5]])
    assert detect_cycles(matrix, ["a", "b", "c"], 0.55)[0]["policies"] == [
        "a",
        "b",
        "c",
    ]
