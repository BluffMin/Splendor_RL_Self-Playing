from pathlib import Path

from splendor_env.recording import load_episode_log


def test_payment_only_text_is_best_effort():
    log = load_episode_log(
        Path(__file__).parent / "fixtures" / "legacy_v031_payment_only.txt"
    )
    assert not log.replay_verifiable and log.migration_quality == "best_effort"
    assert log.document["turns"][0]["primary_action_type"] == "inferred_purchase"
