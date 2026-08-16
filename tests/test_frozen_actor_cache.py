def test_frozen_actor_cache_is_bounded_and_lru(tmp_path):
    from league_helpers import models

    from splendor_rl.league.pool import OpponentPool

    actor, _ = models()
    pool = OpponentPool(tmp_path, [8], max_cached_actors=2)
    for index in range(3):
        pool.add_snapshot(
            actor, opponent_id=f"p{index}", source_type="recent",
            created_transition=index, champion_version=None, training_seed=1,
            actor_obs_size=475, action_size=373,
        )
    pool.load("p0")
    pool.load("p1")
    pool.load("p0")
    pool.load("p2")
    assert list(pool.loaded) == ["p0", "p2"]

