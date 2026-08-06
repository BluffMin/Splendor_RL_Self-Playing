import json

from league_helpers import tiny_config

from splendor_rl.league.state import atomic_json_write, load_league_state
from splendor_rl.league.train import train_league
from splendor_rl.progress import ProgressConfig, ProgressMode


def test_league_state_atomic_roundtrip(tmp_path):
    path = tmp_path / "league_state.json"
    value = {"schema_version": "0.5.0", "candidate": {"global_transition_count": 8}}
    atomic_json_write(path, value)
    assert load_league_state(path) == value
    assert not (tmp_path / "league_state.json.tmp").exists()


def test_candidate_and_league_state_resume_without_duplicate_threshold(tmp_path):
    config = tiny_config(promotion_interval=32, matchup_matrix_interval=32)
    disabled = ProgressConfig(ProgressMode.NEVER)
    train_league(config, tmp_path, stop_at_transitions=8, progress_config=disabled)
    assert (tmp_path / "champion" / "current_actor.pt").exists()
    assert (tmp_path / "champion" / "current_metadata.json").exists()
    checkpoint = tmp_path / "candidate" / "checkpoints" / "step_000000008.pt"
    train_league(
        config,
        tmp_path,
        resume=checkpoint,
        stop_at_transitions=16,
        progress_config=disabled,
    )
    state = load_league_state(tmp_path / "league_state.json")
    assert state["candidate"]["global_transition_count"] == 16
    assert state["thresholds"]["next_snapshot"] == 24
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "league_training.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["learning_rate"] == 0.00015
