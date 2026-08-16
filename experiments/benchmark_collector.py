from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.records import MatchRecords
from splendor_rl.population.config import PopulationConfig
from splendor_rl.population.multiprocess_collector import (
    MultiprocessBatchedPopulationCollector,
)
from splendor_rl.population.rollout import PopulationRolloutCollector
from splendor_rl.population.train import _load_learner, _selector


def _csv_ints(value):
    return [int(item) for item in value.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark population collectors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--envs-per-worker", default="4,8,16")
    parser.add_argument("--transitions", type=int, default=100_000)
    parser.add_argument("--output")
    parser.add_argument("--progress", choices=["auto", "always", "never"], default="auto")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    config = PopulationConfig.load(args.config)
    state = json.loads((run / "population_state.json").read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint) if args.checkpoint else run / state["learners"]["main"]["checkpoint"]
    raw = torch.load(checkpoint, map_location=args.device, weights_only=False)
    sizes = raw["observation_sizes"]
    config.hidden_sizes = list(raw["config"]["hidden_sizes"])
    pool = OpponentPool(run / "population/pool", config.hidden_sizes, args.device)
    learner = _load_learner(checkpoint, config, sizes, args.device)
    records = MatchRecords(run / "metrics/matchup_records.json", 400)
    champion = state["main"]["champion_id"]
    meta = state["meta_strategy"]
    combinations = [(0, config.num_envs)] + [
        (workers, envs)
        for workers in _csv_ints(args.workers)
        for envs in _csv_ints(args.envs_per_worker)
    ]
    results = []
    for workers, envs in combinations:
        candidate = copy.copy(config)
        candidate.seed = config.seed + 91_000_000
        candidate.num_rollout_workers = max(1, workers)
        candidate.envs_per_worker = envs
        candidate.num_envs = envs if workers == 0 else workers * envs
        selector = _selector("main", candidate, pool, records, champion, meta)
        collector_class = PopulationRolloutCollector if workers == 0 else MultiprocessBatchedPopulationCollector
        torch.manual_seed(candidate.seed)
        started = time.perf_counter()
        collector = collector_class(
            learner.actor, learner.critic, role="main", pool=pool, records=records,
            config=candidate, selector=selector, update_index=learner.updates,
            device=args.device,
        )
        try:
            batch, _, _, metrics = collector.collect(args.transitions, candidate.gae_lambda)
        finally:
            close = getattr(collector, "close", None)
            if close:
                close()
        wall = time.perf_counter() - started
        row = {
            "backend": "single_process" if workers == 0 else "multiprocess_batched",
            "workers": workers,
            "envs_per_worker": envs,
            "total_envs": candidate.num_envs,
            "learning_transitions": len(batch),
            "learning_transitions_per_second": len(batch) / wall,
            "decisions_per_second": metrics["decisions_per_second"],
            "games_per_second": metrics.get("games_per_second"),
            "mean_actor_batch_size": metrics.get("mean_actor_batch_size", 1.0),
            "illegal_actions": metrics["illegal_actions"],
            "invariant_violations": metrics["invariant_violations"],
            "wall_seconds": wall,
        }
        results.append(row)
        print(json.dumps(row))
    baseline = results[0]["learning_transitions_per_second"]
    stable = [
        row for row in results[1:]
        if row["illegal_actions"] == 0 and row["invariant_violations"] == 0
    ]
    winner = max(stable, key=lambda row: row["learning_transitions_per_second"])
    payload = {
        "schema_version": "0.6.1",
        "source_checkpoint": str(checkpoint),
        "transitions_per_trial": args.transitions,
        "results": results,
        "recommended": winner,
        "measured_speedup": winner["learning_transitions_per_second"] / baseline,
    }
    output = Path(args.output) if args.output else run / "benchmarks/collector_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Recommended: workers={winner['workers']}, envs_per_worker={winner['envs_per_worker']}")
    print(f"Measured speedup: {payload['measured_speedup']:.2f}x")


if __name__ == "__main__":
    main()

