import numpy as np

from splendor_rl.population.meta import farthest_point_selection


def test_required_hash_unique_and_diverse_policy_selected():
    ids = ["champ", "same", "near", "far"]
    matrix = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.6],
            [0, 1, 0, 0.5],
        ]
    )
    selected = farthest_point_selection(
        ids, matrix, ["champ"], 3, {"champ": "x", "same": "x", "near": "n", "far": "f"}
    )
    assert selected == ["champ", "far", "near"]
