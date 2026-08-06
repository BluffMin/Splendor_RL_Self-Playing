from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_rl.league.bootstrap import BootstrapConfig, run_bootstrap
from splendor_rl.progress import ProgressConfig, ProgressMode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap a two-player PPO league")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-summary")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--progress", choices=[mode.value for mode in ProgressMode], default="auto"
    )
    parser.add_argument("--progress-refresh-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    run_bootstrap(
        BootstrapConfig.load(args.config),
        args.source_run_dir,
        args.output_dir,
        device=args.device,
        reference_summary=args.reference_summary,
        progress_config=ProgressConfig(
            ProgressMode(args.progress), args.progress_refresh_seconds
        ),
    )


if __name__ == "__main__":
    main()
