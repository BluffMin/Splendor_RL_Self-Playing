"""Small shared-network masked-DQN baseline for two-player self-play."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from splendor_env.actions import N_ACTIONS
from splendor_env.core import OBSERVATION_SIZE, SplendorGame


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_mask: np.ndarray
    continuation: float


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay: deque[Transition] = deque(maxlen=args.replay_size)
    games = SplendorGame(
        num_players=2,
        max_turns=args.max_turns,
        allow_deadlock_pass=True,
        seed=args.seed,
    )
    updates = 0

    for episode in range(1, args.episodes + 1):
        games.reset(seed=args.seed + episode)
        episode_loss = 0.0
        decisions = 0
        epsilon = max(
            args.epsilon_end,
            args.epsilon_start
            - (args.epsilon_start - args.epsilon_end) * episode / args.epsilon_decay,
        )

        while not games.done:
            actor = games.current_player
            observation = games.observation(actor)
            mask = games.legal_action_mask()
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise RuntimeError("no legal action; enable the deadlock safeguard")
            if rng.random() < epsilon:
                action = int(rng.choice(legal))
            else:
                with torch.no_grad():
                    values = online(torch.from_numpy(observation).to(device))
                    invalid = torch.from_numpy(mask == 0).to(device)
                    action = int(values.masked_fill(invalid, -torch.inf).argmax().item())

            result = games.step(action)
            if games.done:
                reward = float(games.terminal_rewards()[actor])
                next_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
                next_mask = np.zeros(N_ACTIONS, dtype=np.int8)
                continuation = 0.0
            else:
                next_actor = games.current_player
                next_observation = games.observation(next_actor)
                next_mask = games.legal_action_mask()
                reward = 0.0
                # Intermediate choices belong to the same real turn and receive
                # no extra time discount. At a two-player turn boundary, values
                # switch perspective and therefore enter with a negative sign.
                continuation = -args.gamma if result.turn_ended else 1.0
            replay.append(
                Transition(
                    observation.copy(), action, reward,
                    next_observation.copy(), next_mask.copy(), continuation,
                )
            )
            decisions += 1

            if len(replay) >= args.batch_size:
                batch = random.sample(replay, args.batch_size)
                obs = torch.from_numpy(np.stack([t.observation for t in batch])).to(device)
                actions = torch.tensor([t.action for t in batch], device=device)
                rewards = torch.tensor([t.reward for t in batch], device=device)
                next_obs = torch.from_numpy(np.stack([t.next_observation for t in batch])).to(device)
                next_masks = torch.from_numpy(np.stack([t.next_mask for t in batch]) == 0).to(device)
                continuation = torch.tensor([t.continuation for t in batch], device=device)

                predicted = online(obs).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_values = target(next_obs).masked_fill(next_masks, -torch.inf).max(1).values
                    next_values = torch.where(torch.isfinite(next_values), next_values, 0.0)
                    expected = rewards + continuation * next_values
                loss = nn.functional.smooth_l1_loss(predicted, expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                episode_loss += float(loss.item())
                updates += 1
                if updates % args.target_interval == 0:
                    target.load_state_dict(online.state_dict())

        if episode == 1 or episode % args.log_interval == 0:
            print(
                f"episode={episode:5d} turns={games.turns_completed:3d} "
                f"decisions={decisions:3d} epsilon={epsilon:.3f} "
                f"loss={episode_loss / max(decisions, 1):.5f} "
                f"winner={games.winners()} reason={games.end_reason} device={device}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": online.state_dict(), "episodes": args.episodes, "seed": args.seed},
        output,
    )
    print(f"saved checkpoint: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=400.0)
    parser.add_argument("--target-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="checkpoints/dqn_self_play.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
