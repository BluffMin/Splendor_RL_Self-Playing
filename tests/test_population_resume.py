from splendor_rl.population.scheduler import DeficitScheduler


def test_scheduler_resume_sequence_matches_uninterrupted():
    original = DeficitScheduler({"main": 0.6, "e0": 0.2, "e1": 0.2})
    [original.next() for _ in range(17)]
    resumed = DeficitScheduler.from_state_dict(original.state_dict())
    assert [original.next() for _ in range(100)] == [resumed.next() for _ in range(100)]
