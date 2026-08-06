from splendor_rl.schedules import next_interval_threshold


def test_resume_does_not_repeat_checkpoint_or_evaluation_threshold():
    assert next_interval_threshold(500000, 100000) == 600000
    assert next_interval_threshold(500001, 100000) == 600000
