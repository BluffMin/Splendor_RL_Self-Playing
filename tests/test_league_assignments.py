from league_helpers import models, tiny_config


def test_league_rejects_non_two_player_config():
    with pytest.raises(ValueError, match="exactly two players"):
        LeagueConfig(num_players=4).validate()


def test_league_config_loads_real_yaml(tmp_path):
    path = tmp_path / "league.yaml"
    path.write_text(
        "training_mode: league_2p\nnum_players: 2\ntotal_transitions: 20000000\n",
        encoding="utf-8",
    )
    config = LeagueConfig.load(path)
    assert config.total_transitions == 20_000_000


from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.records import MatchRecords
from splendor_rl.league.rollout import LeagueRolloutCollector


def test_frozen_candidate_seats_are_balanced(tmp_path):
    config = tiny_config(
        current_selfplay_fraction=0, champion_fraction=1, historical_pfsp_fraction=0
    )
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
    collector = LeagueRolloutCollector(
        actor,
        critic,
        pool=pool,
        records=MatchRecords(tmp_path / "records.json"),
        config=config,
    )
    collector.collect(32)
    assert (
        abs(collector.candidate_seat_counts[0] - collector.candidate_seat_counts[1])
        <= 1
    )
    assert all(item.mode == "candidate_vs_champion" for item in collector.assignments)


import pytest

from splendor_rl.league.config import LeagueConfig
