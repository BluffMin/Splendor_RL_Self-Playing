from __future__ import annotations

import json
import shutil
from pathlib import Path


def _better_average(candidate, current):
    key = lambda s: (
        -s["aggregate"]["average_rank"],
        s["aggregate"]["fractional_first_place_rate"],
        s["aggregate"]["average_score"],
        s["transition_count"],
    )
    return current is None or key(candidate) > key(current)


def _better_random(candidate, current):
    stats = candidate["matchups"]["policy_vs_random"]
    key = (
        stats["fractional_first_place_rate"],
        -stats["average_rank"],
        stats["average_score"],
        candidate["transition_count"],
    )
    if current is None:
        return True
    old = current["matchups"]["policy_vs_random"]
    return key > (
        old["fractional_first_place_rate"],
        -old["average_rank"],
        old["average_score"],
        current["transition_count"],
    )


def _better_greedy(candidate, current):
    stats = candidate["matchups"]["policy_vs_greedy"]
    key = (
        -stats["average_rank"],
        stats["fractional_first_place_rate"],
        stats["average_score"],
        candidate["transition_count"],
    )
    if current is None:
        return True
    old = current["matchups"]["policy_vs_greedy"]
    return key > (
        -old["average_rank"],
        old["fractional_first_place_rate"],
        old["average_score"],
        current["transition_count"],
    )


def update_best_checkpoints(checkpoint_path, summary, checkpoint_dir, state):
    checkpoint_dir = Path(checkpoint_dir)
    updates = []
    criteria = {
        "best_average_rank": _better_average,
        "best_vs_random": _better_random,
        "best_vs_greedy": _better_greedy,
    }
    for name, better in criteria.items():
        old = state.get(name, {}).get("summary")
        if better(summary, old):
            shutil.copy2(checkpoint_path, checkpoint_dir / f"{name}.pt")
            state[name] = {
                "checkpoint": Path(checkpoint_path).name,
                "transition_count": summary["transition_count"],
                "summary": summary,
            }
            updates.append(name)
    (checkpoint_dir / "best_checkpoints.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return updates


def load_best_state(checkpoint_dir, fallback=None):
    path = Path(checkpoint_dir) / "best_checkpoints.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else dict(fallback or {})
    )
