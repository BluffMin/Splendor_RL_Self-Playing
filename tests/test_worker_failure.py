import pytest


def test_worker_failure_is_reported_with_worker_and_environment(tmp_path):
    from league_helpers import models

    from splendor_rl.league.pool import OpponentPool
    from splendor_rl.league.records import MatchRecords
    from splendor_rl.population.config import PopulationConfig
    from splendor_rl.population.multiprocess_collector import (
        MultiprocessBatchedPopulationCollector,
    )

    actor, critic = models()
    pool = OpponentPool(tmp_path / "pool", [8])
    pool.add_snapshot(
        actor, opponent_id="champion_0000", source_type="champion",
        created_transition=0, champion_version=0, training_seed=1,
        actor_obs_size=475, action_size=373,
    )
    config = PopulationConfig(
        hidden_sizes=[8], num_rollout_workers=1, envs_per_worker=1
    )
    collector = MultiprocessBatchedPopulationCollector(
        actor, critic, role="main", pool=pool,
        records=MatchRecords(tmp_path / "records.json", 10), config=config,
        selector=lambda rng: ("champion", "champion_0000", 1.0),
        update_index=0, device="cpu",
    )
    try:
        collector._connections[0].send(("step", [(0, 9999)]))
        with pytest.raises(RuntimeError, match="worker=0, env=0"):
            collector._receive(collector._connections[0], "steps")
    finally:
        collector.close()

