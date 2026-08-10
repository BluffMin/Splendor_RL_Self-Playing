from __future__ import annotations

import numpy as np

from splendor_rl.league.rollout import LeagueRolloutCollector
from splendor_rl.league.types import LeagueEpisodeAssignment


class PopulationRolloutCollector(LeagueRolloutCollector):
    """Candidate-only PPO collector with a configurable frozen-opponent mixture."""

    def __init__(
        self,
        actor,
        critic,
        *,
        role,
        pool,
        records,
        config,
        selector,
        update_index,
        device,
    ):
        self.role = role
        self.selector = selector
        self.update_index = update_index
        # LeagueRolloutCollector calls _assign during initialization.
        super().__init__(
            actor, critic, pool=pool, records=records, config=config, device=device
        )

    def _assign(self, env_id):
        episode_id = self.episode_indices[env_id]
        seed = (
            self.seed + self.update_index * 10_000_019 + env_id * 100_003 + episode_id
        )
        mode, opponent_id, probability = self.selector(self.category_rng)
        seat = self.frozen_episode_count % 2
        self.frozen_episode_count += 1
        metadata = self.pool.metadata[opponent_id]
        assignment = LeagueEpisodeAssignment(
            env_id,
            episode_id,
            mode,
            seat,
            opponent_id,
            metadata.source_type,
            seed,
            probability,
        )
        self.assignments.append(assignment)
        return assignment

    def _learning_player(self, assignment, player):
        return player == assignment.candidate_seat

    def collect(self, target, gae_lambda=0.95):
        batch, advantages, returns, metrics = super().collect(target, gae_lambda)
        for episode in self.episodes:
            episode["population_role"] = self.role
        games = max(1, sum(self.mode_counts.values()))
        metrics["opponent_mode_fractions_actual"] = {
            key: value / games for key, value in self.mode_counts.items()
        }
        metrics["population_role"] = self.role
        rounds = [item["rounds"] for item in self.episodes]
        scored = [
            item for item in self.episodes if item.get("candidate_score") is not None
        ]
        metrics.update(
            average_final_round=float(np.mean(rounds)) if rounds else None,
            average_rounds_on_wins=_round_mean(scored, 1.0),
            average_rounds_on_losses=_round_mean(scored, 0.0),
            average_rounds_on_ties=_round_mean(scored, 0.5),
            average_player_turns=float(
                np.mean([item["turns"] for item in self.episodes])
            )
            if self.episodes
            else None,
            average_decisions=float(
                np.mean([item["decisions"] for item in self.episodes])
            )
            if self.episodes
            else None,
        )
        return batch, advantages, returns, metrics


def _round_mean(episodes, score):
    values = [item["rounds"] for item in episodes if item["candidate_score"] == score]
    return float(np.mean(values)) if values else None


def weighted_selector(categories, *, fallback_id):
    available = [
        (name, ids, weight, probabilities)
        for name, ids, weight, probabilities in categories
        if ids and weight > 0
    ]
    if not available:
        return lambda rng: ("fallback", fallback_id, 1.0)
    weights = np.asarray([item[2] for item in available], dtype=float)
    weights /= weights.sum()

    def select(rng):
        category_index = int(rng.choice(len(available), p=weights))
        name, ids, _, probabilities = available[category_index]
        probs = (
            np.asarray(probabilities, dtype=float)
            if probabilities is not None
            else np.full(len(ids), 1 / len(ids))
        )
        probs /= probs.sum()
        index = int(rng.choice(len(ids), p=probs))
        return name, ids[index], float(weights[category_index] * probs[index])

    return select
