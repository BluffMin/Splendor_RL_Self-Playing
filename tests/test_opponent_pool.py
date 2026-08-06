import json

import pytest
from league_helpers import models

from splendor_rl.league.pool import OpponentPool


def add(pool, actor, opponent_id, source="recent", transition=0):
    return pool.add_snapshot(
        actor,
        opponent_id=opponent_id,
        source_type=source,
        created_transition=transition,
        champion_version=0 if source == "champion" else None,
        training_seed=1,
        actor_obs_size=475,
        action_size=373,
    )


def test_pool_archive_cap_atomic_index_and_validation(tmp_path):
    actor, _ = models()
    pool = OpponentPool(tmp_path, [8])
    add(pool, actor, "champion_0000", "champion")
    add(pool, actor, "recent_1", transition=1)
    add(pool, actor, "recent_2", transition=2)
    pool.trim_recent(1)
    assert pool.hall_of_fame_ids == ["champion_0000"] and pool.recent_ids == [
        "recent_2"
    ]
    assert (
        json.loads((tmp_path / "index.json").read_text())["schema_version"] == "0.5.0"
    )
    with pytest.raises(ValueError, match="duplicate"):
        add(pool, actor, "champion_0000", "champion")
    assert pool.load("champion_0000").metadata.sha256
    (tmp_path / pool.metadata["recent_2"].file_name).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        pool.load("recent_2")
