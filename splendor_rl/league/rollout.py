from __future__ import annotations

from collections import Counter

import numpy as np

from splendor_env.actions import ACTIONS
from splendor_env.core import NoLegalActionError
from splendor_rl.rollout import PlayerTransition, RolloutCollector

from .sampling import pfsp_probabilities, pfsp_weight, sample_pfsp
from .types import LeagueEpisodeAssignment


class LeagueRolloutCollector(RolloutCollector):
    """Two-player collector that never places frozen-opponent decisions in PPO batches."""

    def __init__(self, actor, critic, *, pool, records, config, device="cpu"):
        super().__init__(
            actor,
            critic,
            num_envs=config.num_envs,
            num_players=2,
            seed=config.seed,
            gamma=config.gamma,
            payment_mode=config.payment_mode,
            device=device,
            max_turns=config.max_turns,
        )
        self.pool = pool
        self.records = records
        self.config = config
        self.champion_id = "champion_0000"
        self.category_rng = np.random.default_rng(config.seed + 10_001)
        self.pfsp_rng = np.random.default_rng(config.seed + 20_003)
        self.opponent_rng = np.random.default_rng(config.seed + 30_007)
        self.frozen_episode_count = 0
        self.assignments: list[LeagueEpisodeAssignment] = []
        self.env_assignments = [self._assign(index) for index in range(config.num_envs)]
        self.mode_counts = Counter()
        self.opponent_counts = Counter()
        self.candidate_seat_counts = Counter()
        self.current_transition = 0

    def _assign(self, env_id):
        historical = self.pool.historical_ids(self.champion_id)
        weights = np.array(
            [
                self.config.current_selfplay_fraction,
                self.config.champion_fraction,
                self.config.historical_pfsp_fraction if historical else 0.0,
            ],
            dtype=float,
        )
        weights /= weights.sum()
        category = int(self.category_rng.choice(3, p=weights))
        episode_id = self.episode_indices[env_id]
        seed = self.seed + env_id * 100_003 + episode_id
        if category == 0:
            assignment = LeagueEpisodeAssignment(
                env_id, episode_id, "current_selfplay", None, None, None, seed
            )
        else:
            seat = self.frozen_episode_count % 2
            self.frozen_episode_count += 1
            if category == 1:
                opponent_id = self.champion_id
                probability = 1.0
                mode = "candidate_vs_champion"
            else:
                scores = [
                    self.records.score(
                        item,
                        self.config.pfsp_prior_alpha,
                        self.config.pfsp_prior_beta,
                    )
                    for item in historical
                ]
                opponent_id, probability, _ = sample_pfsp(
                    historical,
                    scores,
                    self.pfsp_rng,
                    self.config.pfsp_alpha,
                    self.config.pfsp_epsilon,
                )
                mode = "candidate_vs_historical"
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
        return (
            assignment.mode == "current_selfplay" or player == assignment.candidate_seat
        )

    def _finish_league_env(self, env_id, env, completed):
        assignment = self.env_assignments[env_id]
        score = None
        if assignment.opponent_id is not None:
            rewards = env.rewards()
            reward = rewards[assignment.candidate_seat] if env.game.terminated else 0.0
            score = 1.0 if reward > 0 else 0.0 if reward < 0 else 0.5
            self.records.add(
                assignment.opponent_id,
                score,
                assignment.candidate_seat,
                self.current_transition,
                env.game.players[assignment.candidate_seat].score,
            )
        self.mode_counts[assignment.mode] += 1
        if assignment.opponent_id:
            self.opponent_counts[assignment.opponent_id] += 1
            self.candidate_seat_counts[assignment.candidate_seat] += 1
        super()._finish_env(env_id, env, completed)
        self.episodes[-1].update(
            {
                "league_mode": assignment.mode,
                "opponent_id": assignment.opponent_id,
                "candidate_seat": assignment.candidate_seat,
                "candidate_score": score,
            }
        )
        self.env_assignments[env_id] = self._assign(env_id)

    def collect(self, target: int, gae_lambda=0.95):
        # This mirrors the stable base collector while skipping frozen decisions.
        completed = list(self.ready)
        self.ready.clear()
        import time

        started = time.perf_counter()
        completed_turns = 0
        while len(completed) < target:
            for env_id, env in enumerate(self.envs):
                if len(completed) >= target:
                    break
                assignment = self.env_assignments[env_id]
                player = env.game.current_player
                learning = self._learning_player(assignment, player)
                try:
                    if learning:
                        obs, state, mask, action, log_prob, value = self._policy(
                            env, player
                        )
                    else:
                        mask = env.action_mask()
                        opponent = self.pool.load(assignment.opponent_id)
                        action = opponent.act(
                            env.actor_observation(player),
                            mask,
                            deterministic=False,
                            generator=self.opponent_rng,
                        )
                except NoLegalActionError:
                    env.game.truncate("training_no_legal_action")
                    self._finish_league_env(env_id, env, completed)
                    continue
                if not mask[action]:
                    self.illegal_actions += 1
                    raise AssertionError("policy selected illegal action")
                kind = ACTIONS[action].kind
                self.action_counts[kind] = self.action_counts.get(kind, 0) + 1
                if learning:
                    key = (env_id, player)
                    previous = self.pending.pop(key, None)
                    if previous is not None:
                        previous.next_value = value
                        previous.discount = (
                            1.0
                            if previous.player_turn_id == env.game.player_turn_id
                            else self.gamma
                        )
                        completed.append(previous)
                        self.trajectories.setdefault(key, []).append(previous)
                    transition = PlayerTransition(
                        obs,
                        state,
                        mask,
                        action,
                        log_prob,
                        value,
                        0.0,
                        0.0,
                        False,
                        False,
                        env_id,
                        player,
                        env.game.decision_id,
                        env.game.player_turn_id,
                        env.game.round_id,
                        env.game.phase.value,
                    )
                    transition.learning_role = (
                        "selfplay_candidate"
                        if assignment.mode == "current_selfplay"
                        else "candidate"
                    )
                    self.pending[(env_id, player)] = transition
                turn_before = env.game.turns_completed
                try:
                    env.step(action)
                    env.game.validate_invariants()
                except Exception:
                    self.invariant_violations += 1
                    raise
                completed_turns += env.game.turns_completed - turn_before
                if env.game.done:
                    self._finish_league_env(env_id, env, completed)
        batch = completed[:target]
        self.ready.extend(completed[target:])
        advantages, returns = self._advantages(batch, gae_lambda)
        elapsed = time.perf_counter() - started
        games = max(1, sum(self.mode_counts.values()))
        historical = self.pool.historical_ids(self.champion_id)
        historical_scores = [
            self.records.score(
                opponent_id,
                self.config.pfsp_prior_alpha,
                self.config.pfsp_prior_beta,
            )
            for opponent_id in historical
        ]
        probabilities = pfsp_probabilities(
            historical_scores, self.config.pfsp_alpha, self.config.pfsp_epsilon
        )
        opponent_sampling = {
            opponent_id: {
                "games": self.records.get(opponent_id).games,
                "score": score,
                "pfsp_weight": pfsp_weight(
                    score, self.config.pfsp_alpha, self.config.pfsp_epsilon
                ),
                "pfsp_probability": float(probability),
                "opponent_source_type": self.pool.metadata[opponent_id].source_type,
                "last_played_transition": self.records.get(
                    opponent_id
                ).last_played_transition,
            }
            for opponent_id, score, probability in zip(
                historical, historical_scores, probabilities, strict=True
            )
        }
        frozen_games = max(1, sum(self.candidate_seat_counts.values()))
        most_sampled = self.opponent_counts.most_common(1)
        metrics = {
            "rollout_seconds": elapsed,
            "decisions_per_second": len(batch) / elapsed,
            "player_turns_per_second": completed_turns / elapsed,
            "illegal_actions": self.illegal_actions,
            "invariant_violations": self.invariant_violations,
            "episode_mode_rates": {
                key: value / games for key, value in self.mode_counts.items()
            },
            "current_selfplay_episode_rate": self.mode_counts["current_selfplay"]
            / games,
            "champion_episode_rate": self.mode_counts["candidate_vs_champion"] / games,
            "historical_episode_rate": self.mode_counts["candidate_vs_historical"]
            / games,
            "unique_opponents_sampled": len(self.opponent_counts),
            "most_sampled_opponent": most_sampled[0][0] if most_sampled else None,
            "most_sampled_opponent_fraction": (
                most_sampled[0][1] / frozen_games if most_sampled else 0.0
            ),
            "candidate_vs_champion_score": self.records.score(self.champion_id),
            "candidate_vs_historical_score": (
                float(np.mean(historical_scores)) if historical_scores else None
            ),
            "candidate_seat_0_games": self.candidate_seat_counts[0],
            "candidate_seat_1_games": self.candidate_seat_counts[1],
            "candidate_seat_0_fraction": self.candidate_seat_counts[0] / frozen_games,
            "candidate_seat_1_fraction": self.candidate_seat_counts[1] / frozen_games,
            "candidate_seat_imbalance": abs(
                self.candidate_seat_counts[0] - self.candidate_seat_counts[1]
            ),
            "opponent_sampling": opponent_sampling,
        }
        return batch, advantages, returns, metrics

    def _advantages(self, batch, gae_lambda):
        from splendor_rl.gae import variable_discount_gae

        advantages = np.zeros(len(batch), dtype=np.float32)
        returns = np.zeros(len(batch), dtype=np.float32)
        by_player = {}
        for index, transition in enumerate(batch):
            by_player.setdefault((transition.env_id, transition.player_id), []).append(
                (index, transition)
            )
        for entries in by_player.values():
            adv, ret = variable_discount_gae(
                [item.reward for _, item in entries],
                [item.value for _, item in entries],
                [item.next_value for _, item in entries],
                [item.discount for _, item in entries],
                gae_lambda,
            )
            for offset, (index, _) in enumerate(entries):
                advantages[index], returns[index] = adv[offset], ret[offset]
        return advantages, returns

    def rng_state(self):
        return {
            "category": self.category_rng.bit_generator.state,
            "pfsp": self.pfsp_rng.bit_generator.state,
            "opponent": self.opponent_rng.bit_generator.state,
            "frozen_episode_count": self.frozen_episode_count,
        }

    def restore_rng_state(self, state):
        self.category_rng.bit_generator.state = state["category"]
        self.pfsp_rng.bit_generator.state = state["pfsp"]
        self.opponent_rng.bit_generator.state = state["opponent"]
        self.frozen_episode_count = state["frozen_episode_count"]
