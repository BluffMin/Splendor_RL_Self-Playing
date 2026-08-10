from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_rl.population.config import PopulationConfig
from splendor_rl.population.train import train_population
from splendor_rl.progress import ProgressConfig, ProgressMode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a PPO-based population league")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--stop-at-population-transitions", type=int)
    parser.add_argument(
        "--progress", choices=["auto", "always", "never"], default="auto"
    )
    args = parser.parse_args(argv)
    if bool(args.bootstrap_run_dir) == bool(args.resume):
        parser.error("exactly one of --bootstrap-run-dir or --resume is required")
    config = PopulationConfig.load(args.config)
    train_population(
        config,
        args.run_dir,
        bootstrap_run_dir=args.bootstrap_run_dir,
        resume=args.resume,
        stop_at_population_transitions=args.stop_at_population_transitions,
        device=args.device,
        progress_config=ProgressConfig(ProgressMode(args.progress), 1.0),
    )


if __name__ == "__main__":
    main()
