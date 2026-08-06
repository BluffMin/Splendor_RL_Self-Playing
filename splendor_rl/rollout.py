from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from splendor_env.actions import ACTIONS
from splendor_env.core import NoLegalActionError
from splendor_env.wrappers import SelfPlayWrapper

from .distributions import MaskedCategorical
from .gae import variable_discount_gae


@dataclass
class PlayerTransition:
    actor_obs: np.ndarray
    critic_state: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    discount: float
    done: bool
    truncated: bool
    env_id: int
    player_id: int
    decision_id: int
    player_turn_id: int
    round_id: int
    phase: str
    next_value: float = 0.0
    learning_role: str = "selfplay_candidate"


class RolloutCollector:
    def __init__(
        self,
        actor,
        critic,
        *,
        num_envs=4,
        num_players=4,
        seed=42,
        gamma=0.997,
        payment_mode="canonical",
        device="cpu",
        max_turns=300,
    ):
        self.actor, self.critic, self.gamma, self.device = (
            actor,
            critic,
            gamma,
            torch.device(device),
        )
        self.num_players, self.seed, self.max_turns = num_players, seed, max_turns
        self.envs = [
            SelfPlayWrapper(
                num_players,
                seed=seed + i * 100003,
                payment_mode=payment_mode,
                max_turns=max_turns,
            )
            for i in range(num_envs)
        ]
        self.episode_indices = [0] * num_envs
        self.pending = {}
        self.trajectories = {}
        self.episodes = []
        self.illegal_actions = 0
        self.invariant_violations = 0
        self.phase_stats = {}
        self.action_counts = {}
        self.ready: list[PlayerTransition] = []

    def _finish_env(self, env_id, env, completed):
        rewards = env.rewards()
        for pending_key in [k for k in self.pending if k[0] == env_id]:
            item = self.pending.pop(pending_key)
            item.reward = rewards[item.player_id]
            item.done = env.game.terminated
            item.truncated = env.game.truncated
            item.discount = 0.0 if env.game.terminated else self.gamma
            item.next_value = (
                0.0 if env.game.terminated else self._critic_value(env, item.player_id)
            )
            completed.append(item)
            self.trajectories.setdefault(pending_key, []).append(item)
        self.episodes.append(
            {
                "turns": env.game.turns_completed,
                "decisions": env.game.decision_id,
                "rounds": env.game.round_id,
                "scores": [p.score for p in env.game.players],
                "winners": env.game.winner_ids(),
                "truncated": env.game.truncated,
                "truncation_reason": env.game.end_reason,
            }
        )
        self.episode_indices[env_id] += 1
        self.envs[env_id] = SelfPlayWrapper(
            self.num_players,
            seed=self.seed + env_id * 100003 + self.episode_indices[env_id],
            payment_mode=env.payment_mode,
            max_turns=self.max_turns,
        )

    def _policy(self, env: SelfPlayWrapper, player: int):
        obs = env.actor_observation(player)
        state = env.critic_state(player)
        mask = env.action_mask()
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0))
            dist = MaskedCategorical(
                logits, torch.as_tensor(mask, device=self.device).unsqueeze(0)
            )
            action = dist.sample()
            value = self.critic(torch.as_tensor(state, device=self.device).unsqueeze(0))
        phase = env.game.phase.value
        stats = self.phase_stats.setdefault(
            phase,
            {
                "decisions": 0,
                "legal_actions": 0,
                "entropy": 0.0,
                "max_probability": 0.0,
            },
        )
        stats["decisions"] += 1
        stats["legal_actions"] += int(mask.sum())
        stats["entropy"] += float(dist.entropy().item())
        stats["max_probability"] += float(dist.probs.max().item())
        return (
            obs,
            state,
            mask,
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
        )

    def _critic_value(self, env: SelfPlayWrapper, player: int) -> float:
        state = env.critic_state(player)
        with torch.no_grad():
            value = self.critic(
                torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            )
        return float(value.item())

    def collect(self, target: int, gae_lambda=0.95):
        completed = list(self.ready)
        self.ready.clear()
        started = time.perf_counter()
        completed_turns = 0
        while len(completed) < target:
            for env_id, env in enumerate(self.envs):
                if len(completed) >= target:
                    break
                player = env.game.current_player
                try:
                    obs, state, mask, action, log_prob, value = self._policy(
                        env, player
                    )
                except NoLegalActionError:
                    # An official-action deadlock is evaluator-owned truncation, not
                    # an engine action or invariant failure.
                    env.game.truncate("training_no_legal_action")
                    self._finish_env(env_id, env, completed)
                    continue
                if not mask[action]:
                    self.illegal_actions += 1
                    raise AssertionError("policy selected illegal action")
                kind = ACTIONS[action].kind
                self.action_counts[kind] = self.action_counts.get(kind, 0) + 1
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
                self.pending[key] = transition
                turn_before = env.game.turns_completed
                try:
                    env.step(action)
                    env.game.validate_invariants()
                except Exception:
                    self.invariant_violations += 1
                    raise
                completed_turns += env.game.turns_completed - turn_before
                if env.game.done:
                    self._finish_env(env_id, env, completed)
        batch = completed[:target]
        self.ready.extend(completed[target:])
        advantages = np.zeros(len(batch), dtype=np.float32)
        returns = np.zeros(len(batch), dtype=np.float32)
        by_player = {}
        for i, t in enumerate(batch):
            by_player.setdefault((t.env_id, t.player_id), []).append((i, t))
        for entries in by_player.values():
            adv, ret = variable_discount_gae(
                [t.reward for _, t in entries],
                [t.value for _, t in entries],
                [t.next_value for _, t in entries],
                [t.discount for _, t in entries],
                gae_lambda,
            )
            for j, (index, _) in enumerate(entries):
                advantages[index], returns[index] = adv[j], ret[j]
        elapsed = time.perf_counter() - started
        total = max(1, sum(self.action_counts.values()))
        metrics = {
            "rollout_seconds": elapsed,
            "decisions_per_second": len(batch) / elapsed,
            "player_turns_per_second": completed_turns / elapsed,
            "illegal_actions": self.illegal_actions,
            "invariant_violations": self.invariant_violations,
            "action_frequencies": {k: v / total for k, v in self.action_counts.items()},
            "phase_stats": {
                k: {
                    "decision_count": v["decisions"],
                    "mean_legal_actions": v["legal_actions"] / v["decisions"],
                    "mean_entropy": v["entropy"] / v["decisions"],
                    "mean_max_probability": v["max_probability"] / v["decisions"],
                }
                for k, v in self.phase_stats.items()
            },
        }
        return batch, advantages, returns, metrics
