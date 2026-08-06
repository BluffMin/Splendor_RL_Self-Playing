from splendor_rl.schedules import next_interval_threshold


def test_resume_threshold_skips_completed_interval():
    assert next_interval_threshold(0, 1000) == 1000
    assert next_interval_threshold(1000, 1000) == 2000
    assert next_interval_threshold(114688, 100000) == 200000
