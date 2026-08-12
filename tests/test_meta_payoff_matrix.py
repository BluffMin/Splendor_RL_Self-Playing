import numpy as np

from splendor_rl.population.meta import antisymmetrize_score_matrix, detect_cycles


def test_mixture_evaluation_uses_fixed_total_pair_budget(monkeypatch):
    from splendor_rl.population.train import _mixture_score

    calls = []

    class Pool:
        def load(self, policy_id):
            return policy_id

    def evaluate(actor, opponent, **kwargs):
        calls.append((opponent, kwargs["pair_count"]))
        return {"pair_scores": [0.5]}

    monkeypatch.setattr(
        "splendor_rl.population.train.paired_actor_evaluation", evaluate
    )
    meta = {"policy_ids": ["a", "b", "c"], "probabilities": [0.2, 0.3, 0.5]}
    assert _mixture_score(object(), Pool(), meta, 7, 42) == 0.5
    assert len(calls) == 7 and all(pair_count == 1 for _, pair_count in calls)


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
