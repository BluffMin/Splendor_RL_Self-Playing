from splendor_rl.league.promotion import promotion_decision


def test_promotion_thresholds_and_identical_hash():
    assert promotion_decision(0.55, 0.51)[0]
    assert not promotion_decision(0.55, 0.50)[0]
    passed, reasons = promotion_decision(0.9, 0.8, identical_hash=True)
    assert not passed and "candidate_and_champion_actor_hash_identical" in reasons
