from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections import Counter

import numpy as np
import torch

from splendor_env.actions import ACTIONS
from splendor_env.core import NoLegalActionError
from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.distributions import MaskedCategorical
from splendor_rl.gae import variable_discount_gae
from splendor_rl.league.types import LeagueEpisodeAssignment
from splendor_rl.rollout import PlayerTransition


def _snapshot(env, env_id):
    started = time.perf_counter()
    player = env.game.current_player
    obs = env.actor_observation(player)
    state = env.critic_state(player)
    observation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mask = env.action_mask()
    mask_seconds = time.perf_counter() - started
    return {
        "env_id": env_id,
        "player": player,
        "observation": obs,
        "critic_state": state,
        "mask": mask,
        "decision_id": env.game.decision_id,
        "player_turn_id": env.game.player_turn_id,
        "round_id": env.game.round_id,
        "phase": env.game.phase.value,
        "turns_completed": env.game.turns_completed,
        "observation_seconds": observation_seconds,
        "mask_seconds": mask_seconds,
    }


def _terminal(env, env_id, step_seconds):
    return {
        "env_id": env_id,
        "terminal": True,
        "rewards": env.rewards(),
        "critic_states": [env.critic_state(player) for player in range(2)],
        "terminated": env.game.terminated,
        "truncated": env.game.truncated,
        "truncation_reason": env.game.end_reason,
        "turns": env.game.turns_completed,
        "decisions": env.game.decision_id,
        "rounds": env.game.round_id,
        "scores": [player.score for player in env.game.players],
        "winners": env.game.winner_ids(),
        "step_seconds": step_seconds,
    }


def population_env_worker(connection, worker_id, specs, payment_mode, max_turns):
    """Spawn-safe environment-only worker; it never owns a PyTorch model."""
    try:
        torch.set_num_threads(1)
        envs = {
            env_id: SelfPlayWrapper(
                2, seed=seed, payment_mode=payment_mode, max_turns=max_turns
            )
            for env_id, seed in specs
        }
        connection.send(("ready", worker_id))
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("closed", worker_id))
                break
            if command == "observe":
                connection.send(("observations", [_snapshot(env, key) for key, env in envs.items()]))
                continue
            if command == "reset":
                values = []
                for env_id, seed in payload:
                    envs[env_id] = SelfPlayWrapper(
                        2, seed=seed, payment_mode=payment_mode, max_turns=max_turns
                    )
                    values.append(_snapshot(envs[env_id], env_id))
                connection.send(("observations", values))
                continue
            if command != "step":
                raise ValueError(f"unknown worker command: {command}")
            results = []
            for env_id, action in payload:
                env = envs[env_id]
                before = env.game.turns_completed
                started = time.perf_counter()
                try:
                    env.step(action)
                    env.game.validate_invariants()
                except Exception:  # noqa: BLE001 - preserve remote traceback context
                    raise RuntimeError(
                        f"worker={worker_id}, env={env_id}, state={env.game.state_hash()}"
                    ) from None
                elapsed = time.perf_counter() - started
                if env.game.done:
                    item = _terminal(env, env_id, elapsed)
                else:
                    try:
                        item = _snapshot(env, env_id)
                        item["terminal"] = False
                        item["step_seconds"] = elapsed
                    except NoLegalActionError:
                        env.game.truncate("training_no_legal_action")
                        item = _terminal(env, env_id, elapsed)
                item["completed_turns"] = env.game.turns_completed - before
                results.append(item)
            connection.send(("steps", results))
    except Exception as error:  # noqa: BLE001 - marshal any worker failure
        try:
            connection.send(
                (
                    "error",
                    {
                        "worker_id": worker_id,
                        "exception": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            return
    finally:
        connection.close()


class MultiprocessBatchedPopulationCollector:
    def __init__(
        self, actor, critic, *, role, pool, records, config, selector,
        update_index, device, profiler=None,
    ):
        self.actor, self.critic = actor, critic
        self.role, self.pool, self.records, self.config = role, pool, records, config
        self.selector, self.update_index = selector, update_index
        self.device, self.profiler = torch.device(device), profiler
        self.gamma, self.seed = config.gamma, config.seed
        self.num_workers = config.num_rollout_workers
        self.num_envs = self.num_workers * config.envs_per_worker
        self.episode_indices = [0] * self.num_envs
        self.category_rng = np.random.default_rng(config.seed + 10_001)
        self.opponent_rng = np.random.default_rng(config.seed + 30_007)
        self.frozen_episode_count = 0
        self.assignments = []
        self.env_assignments = [self._assign(i) for i in range(self.num_envs)]
        self.pending, self.trajectories, self.episodes = {}, {}, []
        self.mode_counts, self.opponent_counts = Counter(), Counter()
        self.candidate_seat_counts = Counter()
        self.action_counts, self.phase_stats = {}, {}
        self.illegal_actions = self.invariant_violations = 0
        self.current_transition = 0
        self.actor_batches = self.actor_items = 0
        self.critic_batches = self.critic_items = 0
        self._ctx = mp.get_context("spawn")
        self._connections, self._processes = [], []
        for worker_id in range(self.num_workers):
            first = worker_id * config.envs_per_worker
            ids = range(first, first + config.envs_per_worker)
            specs = [(i, self.env_assignments[i].seed) for i in ids]
            parent, child = self._ctx.Pipe()
            process = self._ctx.Process(
                target=population_env_worker,
                args=(child, worker_id, specs, config.payment_mode, config.max_turns),
                daemon=False,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        for connection in self._connections:
            self._receive(connection, "ready")

    def _assign(self, env_id):
        episode_id = self.episode_indices[env_id]
        seed = self.seed + self.update_index * 10_000_019 + env_id * 100_003 + episode_id
        mode, opponent_id, probability = self.selector(self.category_rng)
        seat = self.frozen_episode_count % 2
        self.frozen_episode_count += 1
        metadata = self.pool.metadata[opponent_id]
        assignment = LeagueEpisodeAssignment(
            env_id, episode_id, mode, seat, opponent_id,
            metadata.source_type, seed, probability,
        )
        self.assignments.append(assignment)
        return assignment

    def _receive(self, connection, expected):
        kind, payload = connection.recv()
        if kind == "error":
            raise RuntimeError(
                "population worker failed: " + payload["exception"] + "\n" + payload["traceback"]
            )
        if kind != expected:
            raise RuntimeError(f"worker protocol error: expected {expected}, got {kind}")
        return payload

    def _observe_all(self):
        for connection in self._connections:
            connection.send(("observe", None))
        return [item for connection in self._connections for item in self._receive(connection, "observations")]

    def _infer(self, snapshots):
        actions = {}
        learning = []
        frozen = {}
        for item in snapshots:
            assignment = self.env_assignments[item["env_id"]]
            if item["player"] == assignment.candidate_seat:
                learning.append(item)
            else:
                frozen.setdefault(assignment.opponent_id, []).append(item)
        if learning:
            obs = torch.as_tensor(np.stack([x["observation"] for x in learning]), device=self.device)
            states = torch.as_tensor(np.stack([x["critic_state"] for x in learning]), device=self.device)
            masks = torch.as_tensor(np.stack([x["mask"] for x in learning]), device=self.device)
            with torch.inference_mode():
                actor_started = time.perf_counter()
                distribution = MaskedCategorical(self.actor(obs), masks)
                sampled = distribution.sample()
                actor_elapsed = time.perf_counter() - actor_started
                critic_started = time.perf_counter()
                values = self.critic(states)
                critic_elapsed = time.perf_counter() - critic_started
            if self.profiler:
                self.profiler.add("actor_inference", actor_elapsed)
                self.profiler.add("critic_inference", critic_elapsed)
            self.actor_batches += 1
            self.actor_items += len(learning)
            self.critic_batches += 1
            self.critic_items += len(learning)
            log_probs = distribution.log_prob(sampled)
            for index, item in enumerate(learning):
                actions[item["env_id"]] = (
                    int(sampled[index].item()), float(log_probs[index].item()),
                    float(values[index].item()), True,
                )
        for opponent_id, items in frozen.items():
            opponent = self.pool.load(opponent_id)
            obs = torch.as_tensor(np.stack([x["observation"] for x in items]), device=opponent.device)
            masks = torch.as_tensor(np.stack([x["mask"] for x in items]), device=opponent.device)
            started = time.perf_counter()
            with torch.inference_mode():
                probabilities = MaskedCategorical(opponent.actor(obs), masks).probs.cpu().numpy()
            if self.profiler:
                self.profiler.add("actor_inference", time.perf_counter() - started)
            self.actor_batches += 1
            self.actor_items += len(items)
            for item, probs in zip(items, probabilities, strict=True):
                action = int(self.opponent_rng.choice(len(probs), p=probs))
                actions[item["env_id"]] = (action, 0.0, 0.0, False)
        return actions

    def _finish(self, terminal, completed):
        env_id = terminal["env_id"]
        assignment = self.env_assignments[env_id]
        rewards = terminal["rewards"]
        for key in [key for key in self.pending if key[0] == env_id]:
            item = self.pending.pop(key)
            item.reward = rewards[item.player_id]
            item.done = terminal["terminated"]
            item.truncated = terminal["truncated"]
            item.discount = 0.0 if terminal["terminated"] else self.gamma
            if terminal["terminated"]:
                item.next_value = 0.0
            else:
                state = torch.as_tensor(terminal["critic_states"][item.player_id], device=self.device).unsqueeze(0)
                with torch.inference_mode():
                    item.next_value = float(self.critic(state).item())
                self.critic_batches += 1
                self.critic_items += 1
            completed.append(item)
            self.trajectories.setdefault(key, []).append(item)
        score = None
        if assignment.opponent_id is not None:
            reward = rewards[assignment.candidate_seat] if terminal["terminated"] else 0.0
            score = 1.0 if reward > 0 else 0.0 if reward < 0 else 0.5
            self.records.add(assignment.opponent_id, score, assignment.candidate_seat, self.current_transition, terminal["scores"][assignment.candidate_seat])
        self.mode_counts[assignment.mode] += 1
        self.opponent_counts[assignment.opponent_id] += 1
        self.candidate_seat_counts[assignment.candidate_seat] += 1
        self.episodes.append({
            "turns": terminal["turns"], "decisions": terminal["decisions"],
            "rounds": terminal["rounds"], "scores": terminal["scores"],
            "winners": terminal["winners"], "truncated": terminal["truncated"],
            "truncation_reason": terminal["truncation_reason"],
            "league_mode": assignment.mode, "opponent_id": assignment.opponent_id,
            "candidate_seat": assignment.candidate_seat, "candidate_score": score,
            "population_role": self.role,
        })
        self.episode_indices[env_id] += 1
        self.env_assignments[env_id] = self._assign(env_id)

    def collect(self, target, gae_lambda=0.95):
        started = time.perf_counter()
        completed, completed_turns, raw_steps = [], 0, 0
        snapshots = self._observe_all()
        while len(completed) < target:
            actions = self._infer(snapshots)
            commands = [[] for _ in self._connections]
            for item in snapshots:
                env_id = item["env_id"]
                action, log_prob, value, learning = actions[env_id]
                if not item["mask"][action]:
                    self.illegal_actions += 1
                    raise AssertionError("batched policy selected illegal action")
                kind = ACTIONS[action].kind
                self.action_counts[kind] = self.action_counts.get(kind, 0) + 1
                if learning:
                    key = (env_id, item["player"])
                    previous = self.pending.pop(key, None)
                    if previous is not None:
                        previous.next_value = value
                        previous.discount = 1.0 if previous.player_turn_id == item["player_turn_id"] else self.gamma
                        completed.append(previous)
                        self.trajectories.setdefault(key, []).append(previous)
                    transition = PlayerTransition(
                        item["observation"], item["critic_state"], item["mask"], action,
                        log_prob, value, 0.0, 0.0, False, False, env_id,
                        item["player"], item["decision_id"], item["player_turn_id"],
                        item["round_id"], item["phase"],
                    )
                    transition.learning_role = "candidate"
                    self.pending[key] = transition
                commands[env_id // self.config.envs_per_worker].append((env_id, action))
            worker_round_started = time.perf_counter()
            for connection, values in zip(self._connections, commands, strict=True):
                connection.send(("step", values))
            worker_results = [
                self._receive(connection, "steps") for connection in self._connections
            ]
            worker_round_seconds = time.perf_counter() - worker_round_started
            results = [item for values in worker_results for item in values]
            raw_steps += len(results)
            completed_turns += sum(item["completed_turns"] for item in results)
            if self.profiler:
                environment = max(
                    sum(item["step_seconds"] for item in values)
                    for values in worker_results
                )
                observation = max(
                    sum(item.get("observation_seconds", 0) for item in values)
                    for values in worker_results
                )
                masks = max(
                    sum(item.get("mask_seconds", 0) for item in values)
                    for values in worker_results
                )
                self.profiler.add("environment_step", environment, len(results))
                self.profiler.add("observation_build", observation, len(results))
                self.profiler.add("action_mask_build", masks, len(results))
                self.profiler.add(
                    "IPC_wait",
                    max(0.0, worker_round_seconds - environment - observation - masks),
                )
            resets = [[] for _ in self._connections]
            snapshots = []
            for item in results:
                if item["terminal"]:
                    self._finish(item, completed)
                    env_id = item["env_id"]
                    resets[env_id // self.config.envs_per_worker].append((env_id, self.env_assignments[env_id].seed))
                else:
                    snapshots.append(item)
            for index, values in enumerate(resets):
                if values:
                    self._connections[index].send(("reset", values))
            for index, values in enumerate(resets):
                if values:
                    snapshots.extend(self._receive(self._connections[index], "observations"))
        batch = completed[:target]
        gae_started = time.perf_counter()
        advantages = np.zeros(len(batch), dtype=np.float32)
        returns = np.zeros(len(batch), dtype=np.float32)
        by_player = {}
        for index, transition in enumerate(batch):
            by_player.setdefault((transition.env_id, transition.player_id), []).append((index, transition))
        for entries in by_player.values():
            adv, ret = variable_discount_gae(
                [item.reward for _, item in entries], [item.value for _, item in entries],
                [item.next_value for _, item in entries], [item.discount for _, item in entries], gae_lambda,
            )
            for offset, (index, _) in enumerate(entries):
                advantages[index], returns[index] = adv[offset], ret[offset]
        if self.profiler:
            self.profiler.add("GAE", time.perf_counter() - gae_started)
        elapsed = time.perf_counter() - started
        games = len(self.episodes)
        metrics = {
            "rollout_seconds": elapsed,
            "decisions_per_second": raw_steps / elapsed,
            "learning_transitions_per_second": len(batch) / elapsed,
            "raw_env_steps_per_second": raw_steps / elapsed,
            "player_turns_per_second": completed_turns / elapsed,
            "games_per_second": games / elapsed,
            "illegal_actions": self.illegal_actions,
            "invariant_violations": self.invariant_violations,
            "actor_inference_batches_per_second": self.actor_batches / elapsed,
            "mean_actor_batch_size": self.actor_items / max(1, self.actor_batches),
            "critic_inference_batches_per_second": self.critic_batches / elapsed,
            "mean_critic_batch_size": self.critic_items / max(1, self.critic_batches),
            "collector_backend": "multiprocess_batched",
            "population_role": self.role,
            "opponent_mode_fractions_actual": {key: value / max(1, games) for key, value in self.mode_counts.items()},
        }
        return batch, advantages, returns, metrics

    def close(self):
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self._connections:
            try:
                if connection.poll(2):
                    self._receive(connection, "closed")
            except (BrokenPipeError, EOFError, OSError, RuntimeError):
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
