from types import SimpleNamespace


def test_v060_threshold_migration_moves_to_next_boundary():
    from splendor_rl.population.config import ROLES, PopulationConfig
    from splendor_rl.population.train import _migrate_thresholds

    config = PopulationConfig(
        meta_update_interval=2_000_000,
        main_promotion_interval=2_000_000,
        exploiter_evaluation_interval_learner_transitions=1_000_000,
    )
    learners = {
        role: SimpleNamespace(transitions=12_189_696 if role == "main" else 2_031_616)
        for role in ROLES
    }
    migrated = _migrate_thresholds(
        {"promotion_attempts": 3, "promotion_successes": 1},
        learners,
        20_316_160,
        config,
    )
    assert migrated["next_meta"] == 22_000_000
    assert migrated["next_promotion"] == 22_000_000
    assert migrated["next_exploiter_eval_by_role"]["main_exploiter_0"] == 3_000_000
    assert migrated["promotion_attempts"] == 3

