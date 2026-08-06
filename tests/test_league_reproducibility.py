import numpy as np

from splendor_rl.league.sampling import sample_pfsp


def test_assignment_rng_state_roundtrip():
    rng = np.random.default_rng(123)
    state = rng.bit_generator.state
    expected = [sample_pfsp(["a", "b"], [0.2, 0.5], rng)[0] for _ in range(10)]
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    actual = [sample_pfsp(["a", "b"], [0.2, 0.5], restored)[0] for _ in range(10)]
    assert actual == expected
