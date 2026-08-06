import numpy as np

from splendor_rl.gae import normalize_advantages, variable_discount_gae


def test_hand_computed_variable_discount():
    adv, ret = variable_discount_gae([0, 1], [0.2, 0.3], [0.3, 0], [1, 0], 0.5)
    assert np.allclose(adv, [0.45, 0.7]) and np.allclose(ret, [0.65, 1])
    assert np.isfinite(normalize_advantages(adv)).all()
