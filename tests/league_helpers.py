from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.league.config import LeagueConfig
from splendor_rl.models import PrivilegedCritic, SharedActor


def tiny_config(**values):
    config = LeagueConfig(
        num_players=2,
        num_envs=1,
        hidden_sizes=[8],
        transitions_per_update=8,
        update_epochs=1,
        minibatch_size=8,
        total_transitions=16,
        checkpoint_interval=8,
        evaluation_interval=8,
        evaluate_initial_policy=False,
        max_turns=20,
        recent_snapshot_interval=8,
        promotion_interval=8,
        promotion_pair_count=1,
        promotion_bootstrap_samples=20,
        promotion_anchor_games_per_opponent=1,
        matchup_matrix_interval=16,
        matchup_matrix_games_per_pair=2,
        matchup_matrix_max_policies=3,
    )
    config.update(values)
    config.validate()
    return config


def models():
    return (
        SharedActor(
            SelfPlayWrapper.actor_observation_size, SelfPlayWrapper.action_size, [8]
        ),
        PrivilegedCritic(SelfPlayWrapper.critic_state_size, [8]),
    )
