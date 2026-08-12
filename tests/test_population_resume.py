from splendor_rl.population.scheduler import DeficitScheduler


def test_snapshot_write_is_idempotent_after_state_commit_gap(tmp_path):
    from league_helpers import models

    from splendor_rl.league.pool import OpponentPool
    from splendor_rl.population.config import PopulationConfig
    from splendor_rl.population.train import _add_snapshot_idempotent

    actor, _ = models()
    pool = OpponentPool(tmp_path, [8])
    config = PopulationConfig(hidden_sizes=[8])
    sizes = {"actor": 475, "critic": 475, "action": 373}
    first = _add_snapshot_idempotent(
        pool, actor, "recent_main_000000512", "recent", 512, config, sizes
    )
    repeated = _add_snapshot_idempotent(
        pool, actor, "recent_main_000000512", "recent", 512, config, sizes
    )
    assert repeated.sha256 == first.sha256


def test_recovery_removes_only_uncommitted_population_snapshots(tmp_path):
    from league_helpers import models

    from splendor_rl.league.pool import OpponentPool
    from splendor_rl.population.config import ROLES, PopulationConfig
    from splendor_rl.population.learner import make_learner
    from splendor_rl.population.train import _reconcile_pool_to_committed_state

    actor, critic = models()
    config = PopulationConfig(hidden_sizes=[8])
    sizes = {"actor": 475, "critic": 475, "action": 373}
    pool = OpponentPool(tmp_path, [8])
    for opponent_id, transition in (
        ("recent_step_019000000", 19_000_000),
        ("recent_main_003014656", 3_014_656),
    ):
        pool.add_snapshot(
            actor,
            opponent_id=opponent_id,
            source_type="recent",
            created_transition=transition,
            champion_version=None,
            training_seed=42,
            actor_obs_size=475,
            action_size=373,
        )
    learners = {
        role: make_learner(
            role, actor.state_dict(), critic.state_dict(), config, sizes, "cpu"
        )
        for role in ROLES
    }
    removed = _reconcile_pool_to_committed_state(
        pool, learners, 2_998_272, config, sizes
    )
    assert removed == ["recent_main_003014656"]
    assert "recent_step_019000000" in pool.metadata


def test_scheduler_resume_sequence_matches_uninterrupted():
    original = DeficitScheduler({"main": 0.6, "e0": 0.2, "e1": 0.2})
    [original.next() for _ in range(17)]
    resumed = DeficitScheduler.from_state_dict(original.state_dict())
    assert [original.next() for _ in range(100)] == [resumed.next() for _ in range(100)]
