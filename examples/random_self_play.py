from __future__ import annotations

import numpy as np

from splendor_env import env


def main() -> None:
    rng = np.random.default_rng(7)
    game = env(
        num_players=2,
        reward_mode="sparse",
        max_turns=200,
        render_mode="ansi",
    )
    game.reset(seed=42)

    final_rewards = {}
    for agent in game.agent_iter(max_iter=5000):
        observation, reward, terminated, truncated, _info = game.last()
        final_rewards[agent] = reward

        if terminated or truncated:
            action = None
        else:
            legal = np.flatnonzero(observation["action_mask"])
            action = int(rng.choice(legal))

        game.step(action)

    print(game.render())
    print("Final AEC rewards seen by agents:", final_rewards)
    game.close()


if __name__ == "__main__":
    main()
