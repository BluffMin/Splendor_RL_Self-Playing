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
    assert all(t.action_mask[t.action] for t in batch) and all(
        t.discount in {0, 0.997, 1} for t in batch
    )
