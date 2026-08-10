from __future__ import annotations

from dataclasses import dataclass

import torch

from splendor_rl.models import PrivilegedCritic, SharedActor


@dataclass
class Learner:
    role: str
    actor: SharedActor
    critic: PrivilegedCritic
    optimizer: torch.optim.Optimizer
    transitions: int = 0
    updates: int = 0
    generation_transitions: int = 0
    generation_updates: int = 0
    games: int = 0
    generation: int = 0
    archive_successes: int = 0
    respawns: int = 0
    stale_respawns: int = 0
    transitions_since_success: int = 0

    def state_summary(self):
        return {
            "role": self.role,
            "generation": self.generation,
            "learner_transitions": self.transitions,
            "updates": self.updates,
            "generation_transitions": self.generation_transitions,
            "generation_updates": self.generation_updates,
            "games": self.games,
            "archive_successes": self.archive_successes,
            "respawns": self.respawns,
            "stale_respawns": self.stale_respawns,
            "transitions_since_success": self.transitions_since_success,
        }


def make_learner(role, actor_state, critic_state, config, sizes, device):
    actor = SharedActor(sizes["actor"], sizes["action"], config.hidden_sizes).to(device)
    critic = PrivilegedCritic(sizes["critic"], config.hidden_sizes).to(device)
    actor.load_state_dict(actor_state)
    if critic_state is not None:
        critic.load_state_dict(critic_state)
    optimizer = torch.optim.Adam(
        [*actor.parameters(), *critic.parameters()], lr=config.learning_rate, eps=1e-5
    )
    return Learner(role, actor, critic, optimizer)


def reset_learner(learner, actor_state, critic_state, config):
    learner.actor.load_state_dict(actor_state)
    if critic_state is not None:
        learner.critic.load_state_dict(critic_state)
    learner.optimizer = torch.optim.Adam(
        [*learner.actor.parameters(), *learner.critic.parameters()],
        lr=config.learning_rate,
        eps=1e-5,
    )
    learner.generation_transitions = learner.generation_updates = 0
    learner.transitions_since_success = 0
    learner.generation += 1
    learner.respawns += 1
