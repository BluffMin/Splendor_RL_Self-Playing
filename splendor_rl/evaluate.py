from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_checkpoint
from .evaluation import evaluate_ladder
from .models import SharedActor
from .progress import ProgressConfig, ProgressMode


def resolve_checkpoint_num_players(data, requested=None):
    checkpoint_players = data.get(
        "num_players", data.get("config", {}).get("num_players")
    )
    if checkpoint_players is None:
        raise ValueError("checkpoint does not contain num_players")
    if requested is not None and requested != checkpoint_players:
        raise ValueError(
            f"Checkpoint was trained with num_players={checkpoint_players}, but evaluation requested {requested}."
        )
    return checkpoint_players


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games-per-matchup", type=int, default=100)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--actor-only", action="store_true")
    p.add_argument("--num-players", type=int)
    p.add_argument(
        "--progress", choices=[mode.value for mode in ProgressMode], default="auto"
    )
    p.add_argument("--progress-refresh-seconds", type=float, default=1.0)
    a = p.parse_args(argv)
    progress_config = ProgressConfig(
        ProgressMode(a.progress), a.progress_refresh_seconds
    )
    data = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    checkpoint_players = resolve_checkpoint_num_players(data, a.num_players)
    sizes = data["observation_sizes"]
    actor = SharedActor(
        sizes["actor"], sizes["action"], data["config"]["hidden_sizes"]
    ).to(a.device)
    load_checkpoint(
        a.checkpoint,
        actor,
        expected_sizes=sizes,
        map_location=a.device,
        restore_rng=False,
    )
    print(
        json.dumps(
            evaluate_ladder(
                actor,
                output_dir=a.output_dir,
                games_per_matchup=a.games_per_matchup,
                evaluation_seed_base=100000,
                device=a.device,
                num_players=checkpoint_players,
                checkpoint_path=a.checkpoint,
                transition_count=data.get("global_transition_count", 0),
                progress_config=progress_config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
