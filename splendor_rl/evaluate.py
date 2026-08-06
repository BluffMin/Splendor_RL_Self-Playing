from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_checkpoint
from .evaluation import evaluate_ladder
from .models import SharedActor


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games-per-matchup", type=int, default=100)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--actor-only", action="store_true")
    a = p.parse_args(argv)
    data = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
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
            evaluate_ladder(actor, a.output_dir, a.games_per_matchup, device=a.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
