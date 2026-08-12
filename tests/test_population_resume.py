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


def test_meta_cache_reuses_immutable_actor_hash_pairs(tmp_path):
    import json

    from splendor_rl.population.train import _load_meta_pair_cache

    output = tmp_path / "meta/step_000000001"
    output.mkdir(parents=True)
    payload = {
        "policy_ids": ["a", "b"],
        "policy_hashes": {"a": "ha", "b": "hb"},
        "raw_score_matrix": [[0.5, 0.7], [0.3, 0.5]],
        "games": [[0, 100], [100, 0]],
        "standard_errors": [[0, 0.04], [0.04, 0]],
    }
    (output / "meta_strategy.json").write_text(json.dumps(payload))
    cached = _load_meta_pair_cache(tmp_path, object())
    assert cached[("ha", "hb")] == {
        "score": 0.7,
        "games": 100,
        "standard_error": 0.04,
    }


def test_meta_cache_migrates_active_ids_only_at_exact_replayed_boundary(tmp_path):
    import json
    from types import SimpleNamespace

    from splendor_rl.population.train import _load_meta_pair_cache

    output = tmp_path / "meta/step_000000100"
    output.mkdir(parents=True)
    payload = {
        "population_transition": 100,
        "policy_ids": ["active_main", "active_main_exploiter_0"],
        "raw_score_matrix": [[0.5, 0.6], [0.4, 0.5]],
        "games": [[0, 20], [20, 0]],
        "standard_errors": [[0, 0.1], [0.1, 0]],
    }
    (output / "meta_strategy.json").write_text(json.dumps(payload))
    pool = SimpleNamespace(
        metadata={
            "active_main": SimpleNamespace(sha256="main-hash"),
            "active_main_exploiter_0": SimpleNamespace(sha256="exploiter-hash"),
        }
    )

    assert _load_meta_pair_cache(tmp_path, pool) == {}
    cached = _load_meta_pair_cache(tmp_path, pool, replay_transition=100)
    assert cached[("exploiter-hash", "main-hash")]["score"] == 0.4
    assert _load_meta_pair_cache(tmp_path, pool, replay_transition=101) == {}


def test_scheduler_resume_sequence_matches_uninterrupted():
    original = DeficitScheduler({"main": 0.6, "e0": 0.2, "e1": 0.2})
    [original.next() for _ in range(17)]
    resumed = DeficitScheduler.from_state_dict(original.state_dict())
    assert [original.next() for _ in range(100)] == [resumed.next() for _ in range(100)]
