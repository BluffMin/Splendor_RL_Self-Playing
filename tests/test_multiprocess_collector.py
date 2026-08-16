from types import SimpleNamespace

import numpy as np


def test_multiprocess_collector_batches_and_shuts_down(tmp_path):
    from league_helpers import models

    from splendor_rl.league.pool import OpponentPool
    from splendor_rl.league.records import MatchRecords
    from splendor_rl.population.multiprocess_collector import (
        MultiprocessBatchedPopulationCollector,
    )

    actor, critic = models()
    pool = OpponentPool(tmp_path / "pool", [8])
    pool.add_snapshot(
        actor,
        opponent_id="champion_0000",
        source_type="champion",
        created_transition=0,
        champion_version=0,
        training_seed=7,
        actor_obs_size=475,
        action_size=373,
    )
    config = SimpleNamespace(
        gamma=0.997,
        seed=7,
        num_rollout_workers=2,
        envs_per_worker=2,
        payment_mode="canonical",
        max_turns=300,
    )
    records = MatchRecords(tmp_path / "records.json", 20)
    selector = lambda rng: ("champion", "champion_0000", 1.0)
    collector = MultiprocessBatchedPopulationCollector(
        actor,
        critic,
        role="main",
        pool=pool,
        records=records,
        config=config,
        selector=selector,
        update_index=0,
        device="cpu",
    )
    try:
        batch, advantages, returns, metrics = collector.collect(64)
    finally:
        processes = list(collector._processes)
        collector.close()
    assert len(batch) == len(advantages) == len(returns) == 64
    assert np.isfinite(advantages).all() and np.isfinite(returns).all()
    assert metrics["illegal_actions"] == 0
    assert metrics["invariant_violations"] == 0
    assert metrics["mean_actor_batch_size"] > 1
    assert all(not process.is_alive() for process in processes)

