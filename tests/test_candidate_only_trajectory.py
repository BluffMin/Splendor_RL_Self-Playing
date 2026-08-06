from league_helpers import models, tiny_config

from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.records import MatchRecords
from splendor_rl.league.rollout import LeagueRolloutCollector


def collector(tmp_path, **fractions):
    config = tiny_config(**fractions)
    actor, critic = models()
    pool = OpponentPool(tmp_path / "pool", [8])
    pool.add_snapshot(
        actor,
        opponent_id="champion_0000",
        source_type="champion",
        created_transition=0,
        champion_version=0,
        training_seed=1,
        actor_obs_size=475,
        action_size=373,
    )
    return LeagueRolloutCollector(
        actor,
        critic,
        pool=pool,
        records=MatchRecords(tmp_path / "records.json"),
        config=config,
    )


def test_frozen_opponent_transitions_are_excluded(tmp_path):
    batch, *_ = collector(
        tmp_path,
        current_selfplay_fraction=0,
        champion_fraction=1,
        historical_pfsp_fraction=0,
    ).collect(16)
    assert {item.learning_role for item in batch} == {"candidate"}
    assert all(item.discount in {0, 0.997, 1} for item in batch)


def test_current_selfplay_uses_both_seats(tmp_path):
    batch, *_ = collector(
        tmp_path,
        current_selfplay_fraction=1,
        champion_fraction=0,
        historical_pfsp_fraction=0,
    ).collect(32)
    assert {item.learning_role for item in batch} == {"selfplay_candidate"}
    assert {item.player_id for item in batch} == {0, 1}
