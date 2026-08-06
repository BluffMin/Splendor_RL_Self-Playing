import numpy as np

from splendor_rl.league.bootstrap import select_diverse_pool


def test_pool_contains_champion_and_uses_matchup_diversity():
    ids = ["champ", "hard", "pop", "diverse"]
    candidates = [{"candidate_id": value} for value in ids]
    enriched = [
        {
            "candidate_id": value,
            "hard_anchor_score": 1 - index / 10,
            "population_paired_score": 1 - index / 10,
        }
        for index, value in enumerate(ids)
    ]
    matrix = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0, 1, 0, 0.5],
        ]
    )
    selected, _ = select_diverse_pool(candidates, enriched, matrix, "champ", 4)
    assert selected[0] == "champ" and "diverse" in selected and len(set(selected)) == 4
