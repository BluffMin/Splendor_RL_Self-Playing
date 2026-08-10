from splendor_rl.population.scheduler import DeficitScheduler


def test_scheduler_rng_is_not_required_for_reproducibility():
    a = DeficitScheduler({"a": 0.5, "b": 0.5})
    b = DeficitScheduler({"a": 0.5, "b": 0.5})
    assert [a.next() for _ in range(50)] == [b.next() for _ in range(50)]
