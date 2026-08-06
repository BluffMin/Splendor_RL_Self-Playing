from splendor_env.core import NoLegalActionError
from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.rollout import RolloutCollector


def test_collector_identity_masks_and_discounts():
    actor = SharedActor(475, 373, [16])
    critic = PrivilegedCritic(475, [16])
    collector = RolloutCollector(actor, critic, num_envs=4, max_turns=20)
    batch, _adv, _ret, metrics = collector.collect(32)
    assert (
        len(batch) == 32
        and metrics["illegal_actions"] == 0
        and metrics["invariant_violations"] == 0
    )


def test_collector_truncates_no_official_action_deadlock():
    actor = SharedActor(475, 373, [8])
    critic = PrivilegedCritic(475, [8])
    collector = RolloutCollector(actor, critic, num_envs=1, num_players=2, max_turns=20)

    def deadlock():
        raise NoLegalActionError("forced test deadlock")

    collector.envs[0].action_mask = deadlock
    batch, _adv, _ret, metrics = collector.collect(4)
    assert len(batch) == 4
    assert collector.episodes[0]["truncation_reason"] == "training_no_legal_action"
    assert metrics["illegal_actions"] == 0
    assert metrics["invariant_violations"] == 0
    assert all(t.action_mask[t.action] for t in batch) and all(
        t.discount in {0, 0.997, 1} for t in batch
    )
