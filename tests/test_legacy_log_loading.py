import json

from splendor_env.recording import load_episode_log


def test_unversioned_detection(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        json.dumps({"config": {"num_players": 2}, "seed": 1, "events": []}),
        encoding="utf-8",
    )
    log = load_episode_log(p)
    assert log.is_legacy and log.source_schema_version == "legacy-unversioned"
