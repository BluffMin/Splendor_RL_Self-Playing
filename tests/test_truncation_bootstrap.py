import math

from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.rollout import RolloutCollector


def test_truncation_bootstraps_each_pending_player_from_new_state():
    collector = RolloutCollector(
        SharedActor(475, 373, [8]),
        PrivilegedCritic(475, [8]),
        num_envs=1,
        max_turns=4,
        gamma=0.9,
    )
    calls = []
    collector._critic_value = lambda env, player: (
        calls.append(player) or (10.0 + player)
    )
    collector.collect(20)
    truncated = [
        t for values in collector.trajectories.values() for t in values if t.truncated
    ]
    assert {t.player_id for t in truncated} == {0, 1, 2, 3}
    assert all(
        not t.done
        and t.reward == 0
        and t.discount == 0.9
        and t.next_value == 10 + t.player_id
        for t in truncated
    )
    assert calls[:4] == [0, 1, 2, 3] and all(
        math.isfinite(t.next_value) for t in truncated
    )
