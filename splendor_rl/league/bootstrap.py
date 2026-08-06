from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.models import SharedActor
from splendor_rl.progress import ProgressConfig, make_evaluation_progress

from .pool import FrozenOpponent, actor_sha256
from .promotion import actor_vs_bot_score, paired_actor_evaluation
from .state import atomic_json_write
from .types import OpponentMetadata


@dataclass(frozen=True)
class AnchorGroups:
    saturated: tuple[str, ...] = ("random", "shortest")
    hard: tuple[str, ...] = ("greedy", "noble", "blocking")


@dataclass
class BootstrapConfig:
    fixed_bot_games_per_matchup: int = 1000
    pair_count: int = 500
    pool_size: int = 4
    score_tolerance: float = 0.01
    saturated_anchors: list[str] = field(default_factory=lambda: ["random", "shortest"])
    hard_anchors: list[str] = field(
        default_factory=lambda: ["greedy", "noble", "blocking"]
    )
    numbered_checkpoint_limit: int = 6
    include_best_average_rank: bool = True
    include_best_vs_random: bool = True
    include_best_vs_greedy: bool = True
    include_latest: bool = True
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    evaluation_seed_base: int = 3_000_000

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw = raw.get("bootstrap", raw)
        sources = raw.pop("checkpoint_sources", {})
        aliases = {
            "fixed_bot_games_per_matchup": raw.pop("fixed_bot_games_per_matchup", 1000),
            "pair_count": raw.pop("pair_count", 500),
            "pool_size": raw.pop("pool_size", 4),
            "score_tolerance": raw.pop("score_tolerance", 0.01),
            **raw,
            **sources,
        }
        value = cls(**aliases)
        value.validate()
        return value

    def validate(self):
        for name in (
            "fixed_bot_games_per_matchup",
            "pair_count",
            "pool_size",
            "numbered_checkpoint_limit",
            "bootstrap_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.hard_anchors or not self.saturated_anchors:
            raise ValueError("hard and saturated anchors must not be empty")

    @property
    def anchors(self):
        return AnchorGroups(tuple(self.saturated_anchors), tuple(self.hard_anchors))


def _checkpoint_info(path, source_selection):
    data = torch.load(path, map_location="cpu", weights_only=False)
    players = data.get("num_players", data.get("config", {}).get("num_players"))
    if players != 2:
        raise ValueError(f"bootstrap checkpoint must be two-player: {path}")
    sizes = data.get("observation_sizes")
    expected = {
        "actor": SelfPlayWrapper.actor_observation_size,
        "critic": SelfPlayWrapper.critic_state_size,
        "action": SelfPlayWrapper.action_size,
    }
    if sizes != expected:
        raise ValueError(f"incompatible observation/action schema: {path}")
    return {
        "candidate_id": source_selection,
        "checkpoint_path": str(Path(path).resolve()),
        "transition_count": int(data.get("global_transition_count", 0)),
        "actor_sha256": actor_sha256(data["actor_state_dict"]),
        "source_selection": source_selection,
        "hidden_sizes": list(data["config"]["hidden_sizes"]),
        "num_players": 2,
        "observation_sizes": sizes,
    }


def discover_checkpoints(source_run_dir, config: BootstrapConfig):
    checkpoint_dir = Path(source_run_dir) / "checkpoints"
    named = []
    selections = (
        ("best_average_rank", config.include_best_average_rank),
        ("best_vs_random", config.include_best_vs_random),
        ("best_vs_greedy", config.include_best_vs_greedy),
        ("latest", config.include_latest),
    )
    for name, enabled in selections:
        path = checkpoint_dir / f"{name}.pt"
        if enabled and path.exists():
            named.append((path, name))
    numbered = sorted(checkpoint_dir.glob("step_*.pt"))
    if len(numbered) > config.numbered_checkpoint_limit:
        indices = np.linspace(
            0, len(numbered) - 1, config.numbered_checkpoint_limit, dtype=int
        )
        numbered = [numbered[index] for index in dict.fromkeys(indices)]
    named.extend((path, path.stem) for path in numbered)
    unique = []
    seen = set()
    for path, source in named:
        info = _checkpoint_info(path, source)
        if info["actor_sha256"] not in seen:
            unique.append(info)
            seen.add(info["actor_sha256"])
    if not unique:
        raise ValueError("no compatible unique bootstrap checkpoints were found")
    return unique


def load_frozen_candidate(info, device):
    data = torch.load(info["checkpoint_path"], map_location=device, weights_only=False)
    actor = SharedActor(475, 373, info["hidden_sizes"])
    actor.load_state_dict(data["actor_state_dict"])
    metadata = OpponentMetadata(
        info["candidate_id"],
        "bootstrap_candidate",
        info["transition_count"],
        None,
        data.get("config", {}).get("seed", 0),
        475,
        373,
        2,
        info["checkpoint_path"],
        info["actor_sha256"],
    )
    return FrozenOpponent(actor, metadata, device)


def summarize_anchors(results, groups: AnchorGroups):
    hard = [results[name]["fractional_first_place_rate"] for name in groups.hard]
    saturated = [
        results[name]["fractional_first_place_rate"] for name in groups.saturated
    ]
    all_scores = hard + saturated
    win_rounds = [
        value["average_rounds_on_wins"]
        for value in results.values()
        if value["average_rounds_on_wins"] is not None
    ]
    return {
        "hard_anchor_score": float(np.mean(hard)),
        "saturated_anchor_score": float(np.mean(saturated)),
        "aggregate_anchor_score": float(np.mean(all_scores)),
        "worst_hard_anchor_score": float(min(hard)),
        "seat_swapped_score": float(np.mean(all_scores)),
        "average_final_round": float(
            np.mean([value["average_final_round"] for value in results.values()])
        ),
        "average_rounds_on_wins": float(np.mean(win_rounds)) if win_rounds else None,
    }


def paired_checkpoint_matrix(candidates, actors, config, progress=None):
    ids = [item["candidate_id"] for item in candidates]
    matrix = np.full((len(ids), len(ids)), 0.5)
    details = {}
    for row in range(len(ids)):
        for column in range(row + 1, len(ids)):
            result = paired_actor_evaluation(
                actors[ids[row]],
                actors[ids[column]],
                pair_count=config.pair_count,
                seed_base=config.evaluation_seed_base + row * 100_000 + column * 1_000,
                progress=progress,
                matchup_label=f"{ids[row]}_vs_{ids[column]}",
            )
            score = float(np.mean(result["pair_scores"]))
            matrix[row, column], matrix[column, row] = score, 1.0 - score
            details[f"{ids[row]}__vs__{ids[column]}"] = result
    return ids, matrix, details


def _tolerance_filter(items, key, tolerance, *, higher=True):
    values = [key(item) for item in items]
    target = max(values) if higher else min(values)
    return [
        item
        for item in items
        if (target - key(item) if higher else key(item) - target) <= tolerance
    ]


def select_bootstrap_champion(candidates, metrics, matrix, tolerance=0.01):
    ids = [item["candidate_id"] for item in candidates]
    enriched = []
    for index, item in enumerate(candidates):
        enriched.append(
            item
            | metrics[item["candidate_id"]]
            | {
                "population_paired_score": float(
                    np.mean(np.delete(matrix[index], index))
                )
                if len(ids) > 1
                else 0.5
            }
        )
    remaining = enriched
    criteria = [
        ("hard_anchor_score", True),
        ("population_paired_score", True),
        ("worst_hard_anchor_score", True),
        ("aggregate_anchor_score", True),
        ("average_rounds_on_wins", False),
        ("transition_count", True),
    ]
    trace = []
    for name, higher in criteria:
        if len(remaining) <= 1:
            break
        before = [item["candidate_id"] for item in remaining]
        if name == "average_rounds_on_wins":
            usable = [item for item in remaining if item[name] is not None]
            if usable:
                remaining = _tolerance_filter(
                    usable, lambda item, key=name: item[key], 0, higher=False
                )
        else:
            threshold = tolerance if name != "transition_count" else 0
            remaining = _tolerance_filter(
                remaining,
                lambda item, key=name: item[key],
                threshold,
                higher=higher,
            )
        trace.append(
            {
                "criterion": name,
                "before": before,
                "after": [item["candidate_id"] for item in remaining],
            }
        )
    selected = remaining[0]
    return selected, enriched, trace


def select_diverse_pool(candidates, enriched, matrix, champion_id, pool_size):
    ids = [item["candidate_id"] for item in candidates]
    by_id = {item["candidate_id"]: item for item in enriched}
    selected = [champion_id]
    rankings = [
        sorted(ids, key=lambda item: by_id[item]["hard_anchor_score"], reverse=True),
        sorted(
            ids, key=lambda item: by_id[item]["population_paired_score"], reverse=True
        ),
    ]
    for ranking in rankings:
        for candidate_id in ranking:
            if candidate_id not in selected:
                selected.append(candidate_id)
                break
    champion_vector = matrix[ids.index(champion_id)]
    diversity = sorted(
        (
            (float(np.mean(np.abs(matrix[ids.index(item)] - champion_vector))), item)
            for item in ids
            if item not in selected
        ),
        reverse=True,
    )
    selected.extend(item for _, item in diversity[: max(0, pool_size - len(selected))])
    if len(selected) < pool_size:
        selected.extend(item for item in ids if item not in selected)
    return selected[:pool_size], {item: distance for distance, item in diversity}


def run_bootstrap(
    config,
    source_run_dir,
    output_dir,
    *,
    device="cpu",
    progress_config=None,
    reference_summary=None,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = discover_checkpoints(source_run_dir, config)
    actors = {
        item["candidate_id"]: load_frozen_candidate(item, device) for item in candidates
    }
    total_games = (
        len(candidates)
        * len(config.anchors.hard + config.anchors.saturated)
        * config.fixed_bot_games_per_matchup
    )
    progress = make_evaluation_progress(
        total_games, 0, progress_config or ProgressConfig()
    )
    fixed, metrics = {}, {}
    for item in candidates:
        candidate_id = item["candidate_id"]
        fixed[candidate_id] = {
            bot: actor_vs_bot_score(
                actors[candidate_id],
                bot,
                games=config.fixed_bot_games_per_matchup,
                seed_base=config.evaluation_seed_base + index * 100_000,
                progress=progress,
                label=f"{candidate_id}_vs_{bot}",
                include_games=True,
            )
            for index, bot in enumerate(config.anchors.hard + config.anchors.saturated)
        }
        metrics[candidate_id] = summarize_anchors(fixed[candidate_id], config.anchors)
    progress.close()
    pair_total = len(candidates) * (len(candidates) - 1) // 2 * config.pair_count * 2
    pair_progress = make_evaluation_progress(
        pair_total, 0, progress_config or ProgressConfig()
    )
    ids, matrix, pair_details = paired_checkpoint_matrix(
        candidates, actors, config, pair_progress
    )
    pair_progress.close()
    selected, enriched, trace = select_bootstrap_champion(
        candidates, metrics, matrix, config.score_tolerance
    )
    pool_ids, diversity = select_diverse_pool(
        candidates,
        enriched,
        matrix,
        selected["candidate_id"],
        min(config.pool_size, len(candidates)),
    )
    manifest = {"schema_version": "0.5.1", "candidates": candidates}
    selection = {
        "schema_version": "0.5.1",
        "selected_champion_id": selected["candidate_id"],
        "selected": selected,
        "selection_trace": trace,
    }
    pool_manifest = {
        "schema_version": "0.5.1",
        "num_players": 2,
        "selected_champion_id": selected["candidate_id"],
        "policies": [
            next(item for item in candidates if item["candidate_id"] == candidate_id)
            | {
                "opponent_id": (
                    "champion_0000"
                    if candidate_id == selected["candidate_id"]
                    else f"bootstrap_{candidate_id}"
                ),
                "source_type": "bootstrap_champion"
                if candidate_id == selected["candidate_id"]
                else "bootstrap_historical",
                "matchup_vector_distance": diversity.get(candidate_id, 0.0),
            }
            for candidate_id in pool_ids
        ],
    }
    matrix_payload = {
        "schema_version": "0.5.1",
        "policy_ids": ids,
        "matrix": matrix.tolist(),
        "pair_count": config.pair_count,
        "pair_details": pair_details,
    }
    atomic_json_write(output / "candidate_manifest.json", manifest)
    atomic_json_write(
        output / "fixed_bot_results.json", {"results": fixed, "metrics": metrics}
    )
    atomic_json_write(output / "checkpoint_matchup_matrix.json", matrix_payload)
    atomic_json_write(output / "bootstrap_selection.json", selection)
    atomic_json_write(output / "bootstrap_pool_manifest.json", pool_manifest)
    with (output / "fixed_bot_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "anchor",
                "score",
                "average_final_round",
                "average_rounds_on_wins",
            ]
        )
        for candidate_id, anchors in fixed.items():
            for anchor, result in anchors.items():
                writer.writerow(
                    [
                        candidate_id,
                        anchor,
                        result["score"],
                        result["average_final_round"],
                        result["average_rounds_on_wins"],
                    ]
                )
    with (output / "checkpoint_matchup_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", *ids])
        for candidate_id, values in zip(ids, matrix, strict=True):
            writer.writerow([candidate_id, *values])
    reference = (
        json.loads(Path(reference_summary).read_text(encoding="utf-8"))
        if reference_summary and Path(reference_summary).exists()
        else None
    )
    report = [
        "# Empirical league bootstrap",
        "",
        f"Selected Champion: `{selected['candidate_id']}`",
        "",
        "## Hard anchors first",
        "",
        "| Checkpoint | Hard score | Saturated | Aggregate | Population paired | Rounds on wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in enriched:
        report.append(
            f"| {item['candidate_id']} | {item['hard_anchor_score']:.3f} | {item['saturated_anchor_score']:.3f} | {item['aggregate_anchor_score']:.3f} | {item['population_paired_score']:.3f} | {item['average_rounds_on_wins']} |"
        )
    report.extend(["", "## Pool", "", *[f"- {item}" for item in pool_ids]])
    if reference:
        report.extend(
            [
                "",
                "## Reference baseline",
                "",
                f"- source: `{reference_summary}`",
                f"- aggregate fractional first-place rate: {reference.get('aggregate', {}).get('fractional_first_place_rate')}",
            ]
        )
    (output / "bootstrap_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "candidates": candidates,
        "fixed": fixed,
        "metrics": metrics,
        "matrix": matrix_payload,
        "selection": selection,
        "pool_manifest": pool_manifest,
    }
