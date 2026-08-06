import numpy as np

from splendor_rl.league.bootstrap import select_bootstrap_champion


def item(name, transition=1):
    return {"candidate_id": name, "transition_count": transition}


def metric(hard, worst=None, aggregate=None, rounds=60):
    return {
        "hard_anchor_score": hard,
        "saturated_anchor_score": 0.8,
        "worst_hard_anchor_score": worst or hard,
        "aggregate_anchor_score": aggregate or hard,
        "average_rounds_on_wins": rounds,
    }


def test_hard_anchor_precedes_filename_and_population_breaks_tie():
    candidates = [item("best_vs_random"), item("hard")]
    selected, _, _ = select_bootstrap_champion(
        candidates,
        {"best_vs_random": metric(0.4), "hard": metric(0.6)},
        np.array([[0.5, 0.9], [0.1, 0.5]]),
    )
    assert selected["candidate_id"] == "hard"
    candidates = [item("a"), item("b")]
    selected, _, _ = select_bootstrap_champion(
        candidates,
        {"a": metric(0.5), "b": metric(0.5)},
        np.array([[0.5, 0.4], [0.6, 0.5]]),
    )
    assert selected["candidate_id"] == "b"


def test_worst_aggregate_and_round_efficiency_are_later_tiebreaks():
    candidates = [item("a"), item("b")]
    matrix = np.full((2, 2), 0.5)
    selected, _, _ = select_bootstrap_champion(
        candidates,
        {"a": metric(0.5, 0.40, 0.55, 50), "b": metric(0.5, 0.45, 0.50, 40)},
        matrix,
    )
    assert selected["candidate_id"] == "b"
