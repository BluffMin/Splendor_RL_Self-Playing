from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splendor_rl.config import PPOConfig
from splendor_rl.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--resume")
    p.add_argument("--seed", type=int)
    p.add_argument("--num-players", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--total-transitions", type=int)
    p.add_argument("--payment-mode", choices=("canonical", "exact"))
    p.add_argument("--device")
    a = p.parse_args()
    config = PPOConfig.load(a.config)
    config.update(
        {
            "seed": a.seed,
            "num_players": a.num_players,
            "num_envs": a.num_envs,
            "total_transitions": a.total_transitions,
            "payment_mode": a.payment_mode,
            "device": a.device,
        }
    )
    train(config, a.run_dir, a.resume)


if __name__ == "__main__":
    main()
