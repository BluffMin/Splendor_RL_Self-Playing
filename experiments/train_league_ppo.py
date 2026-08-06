from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_rl.league.config import LeagueConfig
from splendor_rl.league.train import train_league
from splendor_rl.progress import ProgressConfig, ProgressMode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train two-player PPO-based league self-play"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--bootstrap-manifest")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-at-transitions", type=int)
    parser.add_argument(
        "--progress", choices=[mode.value for mode in ProgressMode], default="auto"
    )
    parser.add_argument("--progress-refresh-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if (
        sum(
            bool(value)
            for value in (args.initial_checkpoint, args.bootstrap_manifest, args.resume)
        )
        > 1
    ):
        parser.error(
            "--initial-checkpoint, --bootstrap-manifest, and --resume are mutually exclusive"
        )
    config = LeagueConfig.load(args.config)
    config.update({"device": args.device, "seed": args.seed})
    train_league(
        config,
        args.run_dir,
        initial_checkpoint=args.initial_checkpoint,
        bootstrap_manifest=args.bootstrap_manifest,
        resume=args.resume,
        stop_at_transitions=args.stop_at_transitions,
        progress_config=ProgressConfig(
            ProgressMode(args.progress), args.progress_refresh_seconds
        ),
    )


if __name__ == "__main__":
    main()
