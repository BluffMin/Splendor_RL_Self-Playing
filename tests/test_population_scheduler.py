from splendor_rl.population.scheduler import DeficitScheduler


def test_scheduler_is_deterministic_balanced_and_resumable():
    weights = {"main": 0.6, "a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}
    first = DeficitScheduler(weights)
    sequence = [first.next() for _ in range(100)]
    second = DeficitScheduler(weights)
    assert sequence == [second.next() for _ in range(100)]
    assert first.counts == {"main": 60, "a": 10, "b": 10, "c": 10, "d": 10}
    restored = DeficitScheduler.from_state_dict(first.state_dict())
    assert [first.next() for _ in range(20)] == [restored.next() for _ in range(20)]


def test_scheduler_supports_disabled_roles():
    scheduler = DeficitScheduler({"main": 1.0, "disabled": 0.0})
    assert {scheduler.next() for _ in range(10)} == {"main"}
