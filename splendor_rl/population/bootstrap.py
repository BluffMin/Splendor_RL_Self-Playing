from __future__ import annotations

import json
from pathlib import Path

import torch

from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.league.pool import OpponentPool, actor_sha256


def bootstrap_population(source_run_dir, destination_pool, config, device):
    source = Path(source_run_dir)
    state = json.loads((source / "league_state.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") not in {"0.5.0", "0.5.1"}
        or state.get("num_players") != 2
    ):
        raise ValueError("bootstrap source must be a compatible v0.5 two-player league")
    champion_id = state["champion"]["opponent_id"]
    source_index = json.loads(
        (source / "opponent_pool/index.json").read_text(encoding="utf-8")
    )
    metadata = {item["opponent_id"]: item for item in source_index["opponents"]}
    if champion_id not in metadata:
        raise ValueError(f"bootstrap Champion is missing: {champion_id}")
    latest = source / "candidate/checkpoints/latest.pt"
    candidate = torch.load(latest, map_location=device, weights_only=False)
    sizes = {
        "actor": SelfPlayWrapper.actor_observation_size,
        "critic": SelfPlayWrapper.critic_state_size,
        "action": SelfPlayWrapper.action_size,
    }
    if candidate.get("observation_sizes") != sizes or candidate.get("num_players") != 2:
        raise ValueError("bootstrap Candidate tensor schema is incompatible")
    config.hidden_sizes = list(candidate["config"]["hidden_sizes"])
    pool = OpponentPool(destination_pool, config.hidden_sizes, device)
    source_pool = OpponentPool(source / "opponent_pool", config.hidden_sizes, device)
    for opponent_id in [
        *state["pool"]["hall_of_fame_ids"],
        *state["pool"]["recent_ids"],
    ]:
        if opponent_id not in source_pool.metadata:
            continue
        frozen = source_pool.load(opponent_id)
        item = source_pool.metadata[opponent_id]
        pool.add_snapshot(
            frozen.actor,
            opponent_id=opponent_id,
            source_type=item.source_type,
            created_transition=item.created_transition,
            champion_version=item.champion_version,
            training_seed=item.training_seed,
            actor_obs_size=sizes["actor"],
            action_size=sizes["action"],
            bootstrap_champion=opponent_id == champion_id,
            source_checkpoint=str(
                (source / "opponent_pool" / item.file_name).resolve()
            ),
        )
    champion = pool.load(champion_id)
    candidate_hash = actor_sha256(candidate["actor_state_dict"])
    champion_hash = champion.metadata.sha256
    actor_state = champion.actor.state_dict()
    return {
        "source_rl_version": candidate.get("rl_version", "0.5.1"),
        "source_transition": int(candidate["global_transition_count"]),
        "source_checkpoint": str(latest.resolve()),
        "source_actor_sha256": champion_hash,
        "candidate_actor_sha256": candidate_hash,
        "candidate_matches_champion": candidate_hash == champion_hash,
        "champion_id": champion_id,
        "champion_version": int(state["champion"]["version"]),
        "actor_state_dict": actor_state,
        "critic_state_dict": candidate.get("critic_state_dict"),
        "sizes": sizes,
        "pool": pool,
    }
