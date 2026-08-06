import hashlib

from splendor_env.migrations.migrate_logs_v032 import migrate


def test_migration_does_not_modify_source(tmp_path):
    source = tmp_path / "x.txt"
    source.write_text("turn counter=9 actor=P0 choose payment plan 0", encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = migrate(source, tmp_path / "out", include_text_replays=True)
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest() == before
        and rows[0]["quality"] == "best_effort"
    )
