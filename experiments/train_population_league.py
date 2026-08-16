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
    parser.add_argument("--evaluation-device")
    parser.add_argument("--collector-backend", choices=["single_process", "multiprocess_batched"])
    parser.add_argument("--num-rollout-workers", type=int)
    parser.add_argument("--envs-per-worker", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output")
    parser.add_argument("--resume-dry-run", action="store_true")
    parser.add_argument("--stop-at-population-transitions", type=int)
    parser.add_argument(
        "--progress", choices=["auto", "always", "never"], default="auto"
    )
    args = parser.parse_args(argv)
    if bool(args.bootstrap_run_dir) == bool(args.resume):
        parser.error("exactly one of --bootstrap-run-dir or --resume is required")
    config = PopulationConfig.load(args.config)
    if args.collector_backend:
        config.collector_backend = args.collector_backend
    if args.num_rollout_workers:
        config.num_rollout_workers = args.num_rollout_workers
    if args.envs_per_worker:
        config.envs_per_worker = args.envs_per_worker
    if args.evaluation_device:
        config.evaluation_device = args.evaluation_device
    if args.profile:
        config.profiling_enabled = True
    try:
        train_population(
            config,
            args.run_dir,
            bootstrap_run_dir=args.bootstrap_run_dir,
            resume=args.resume,
            stop_at_population_transitions=args.stop_at_population_transitions,
            device=args.device,
            progress_config=ProgressConfig(ProgressMode(args.progress), 1.0),
            resume_dry_run=args.resume_dry_run,
            profile_output=args.profile_output,
        )
    except KeyboardInterrupt:
        print("Graceful shutdown complete.")
        if args.resume:
            print(f"Resume checkpoint: {args.resume}")
        return 130


if __name__ == "__main__":
    main()
