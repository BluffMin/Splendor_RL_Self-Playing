import numpy as np
import pytest

from splendor_rl.league.sampling import pfsp_probabilities, pfsp_weight, sample_pfsp


def test_pfsp_is_positive_normalized_and_peaks_at_half():
    assert pfsp_weight(0.5) > pfsp_weight(0) > 0
    probabilities = pfsp_probabilities([0, 0.5, 1])
    assert np.isfinite(probabilities).all() and probabilities.sum() == pytest.approx(1)
    assert probabilities[1] == probabilities.max()
    assert pfsp_probabilities([0.5]).tolist() == [1.0]


def test_pfsp_sampling_is_deterministic_and_unseen_is_selectable():
    first = sample_pfsp(["a", "unseen"], [0.9, 0.5], np.random.default_rng(5))
    second = sample_pfsp(["a", "unseen"], [0.9, 0.5], np.random.default_rng(5))
    assert first[0:2] == second[0:2]
    assert all(first[2] > 0)
