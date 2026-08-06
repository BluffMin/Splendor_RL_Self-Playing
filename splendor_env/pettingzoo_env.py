"""PettingZoo AEC adapter for :class:`splendor_env.core.SplendorGame`."""

from __future__ import annotations

from functools import cache
from typing import Any, ClassVar, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .actions import N_ACTIONS, describe_action
from .core import OBSERVATION_SIZE, SplendorGame

RewardMode = Literal["sparse", "score"]


class raw_env(AECEnv):
    """Turn-based Splendor environment following the PettingZoo AEC API.

    ``score`` reward mode adds zero-sum shaping whenever prestige is earned.
    ``sparse`` (the default) emits only the final game result.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "ansi"],
        "name": "splendor_selfplay_v3",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(
        self,
        num_players: int = 2,
        *,
        reward_mode: RewardMode = "sparse",
        shaping_scale: float = 0.1,
        max_turns: int | None = None,
        render_mode: str | None = None,
        render_omniscient: bool = False,
    ) -> None:
        super().__init__()
        if reward_mode not in ("sparse", "score"):
            raise ValueError("reward_mode must be 'sparse' or 'score'")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'")

        self.num_players_config = int(num_players)
        self.reward_mode = reward_mode
        self.shaping_scale = float(shaping_scale)
        self.render_mode = render_mode
        self.render_omniscient = bool(render_omniscient)
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive or None")
        self.max_turns = max_turns
        self.game = SplendorGame(num_players=num_players)

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        self.state_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    @cache  # noqa: B019 - spaces are immutable for the environment lifetime
    def observation_space(self, agent: str) -> spaces.Dict:
        del agent
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )

    @cache  # noqa: B019 - spaces are immutable for the environment lifetime
    def action_space(self, agent: str) -> spaces.Discrete:
        del agent
        return spaces.Discrete(N_ACTIONS)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        del options
        self.game.reset(seed=seed)
        self._last_turn_ended = False
        self._last_automatic_resolution = ()
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[self.game.current_player]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: self._info_for(agent) for agent in self.agents}

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        player_index = self.agent_name_mapping[agent]
        is_active = (
            agent == self.agent_selection
            and agent in self.agents
            and not self.terminations.get(agent, False)
            and not self.truncations.get(agent, False)
        )
        mask = (
            self.game.legal_action_mask(player_index)
            if is_active
            else np.zeros(N_ACTIONS, dtype=np.int8)
        )
        return {
            "observation": self.game.observation(player_index),
            "action_mask": mask,
        }

    def state(self) -> np.ndarray:
        return self.game.state()

    def step(self, action: int | None) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError("live agents must provide an integer action")

        agent = self.agent_selection
        actor_index = self.agent_name_mapping[agent]
        if actor_index != self.game.current_player:
            raise RuntimeError("PettingZoo agent selection and rules engine diverged")

        self._cumulative_rewards[agent] = 0.0
        self.rewards = {name: 0.0 for name in self.agents}
        result = self.game.step(int(action))
        self._last_turn_ended = result.turn_ended
        self._last_automatic_resolution = result.automatic_resolution
        if (
            self.max_turns is not None
            and result.turn_ended
            and self.game.turns_completed >= self.max_turns
            and not self.game.done
        ):
            self.game.truncate("max_turns_truncation")

        if self.reward_mode == "score" and result.score_delta:
            shaped = self.shaping_scale * float(result.score_delta)
            self.rewards[agent] += shaped
            if self.num_players_config > 1:
                penalty = shaped / (self.num_players_config - 1)
                for other in self.agents:
                    if other != agent:
                        self.rewards[other] -= penalty

        if self.game.done:
            terminal = self.game.terminal_rewards()
            for name, player_index in self.agent_name_mapping.items():
                self.rewards[name] += float(terminal[player_index])
            self.terminations = {
                name: bool(self.game.terminated) for name in self.agents
            }
            self.truncations = {
                name: bool(self.game.truncated) for name in self.agents
            }
        else:
            self.agent_selection = self.possible_agents[self.game.current_player]

        self.infos = {name: self._info_for(name) for name in self.agents}
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _info_for(self, agent: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "phase": self.game.phase.value,
            "decision_id": self.game.decision_id,
            "turn_id": self.game.turns_completed,
            "round_id": self.game.round_id,
            "acting_player": self.game.current_player,
            "automatic_resolution": list(
                getattr(self, "_last_automatic_resolution", ())
            ),
            "turns_completed": self.game.turns_completed,
            "current_player": self.game.current_player,
            "turn_completed": getattr(self, "_last_turn_ended", False),
        }
        if self.game.last_action is not None:
            info["last_action"] = self.game.last_action
            info["last_action_text"] = describe_action(self.game.last_action)
            info["action_text"] = describe_action(self.game.last_action)
            info["last_actor"] = self.game.last_actor
        if self.game.done:
            info.update(
                {
                    "end_reason": self.game.end_reason,
                    "winner_indices": self.game.winners(),
                    "is_winner": self.agent_name_mapping[agent]
                    in self.game.winners(),
                }
            )
        return info

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        text = self.game.render(
            perspective=self.game.current_player,
            omniscient=self.render_omniscient,
        )
        if self.render_mode == "human":
            print(text)
            print("-" * 100)
            return None
        if self.render_mode == "ansi":
            return text
        return None

    def close(self) -> None:
        return None


def env(**kwargs: Any) -> AECEnv:
    """Wrapped environment with bounds and call-order validation."""
    environment: AECEnv = raw_env(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment
