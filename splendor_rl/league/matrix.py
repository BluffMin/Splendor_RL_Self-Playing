from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .promotion import paired_actor_evaluation


def detect_cycle(matrix, policy_ids, margin=0.05):
    for a in range(len(policy_ids)):
        for b in range(len(policy_ids)):
            for c in range(len(policy_ids)):
                if len({a, b, c}) < 3:
                    continue
                if (
                    matrix[a, b] > 0.5 + margin
                    and matrix[b, c] > 0.5 + margin
                    and matrix[c, a] > 0.5 + margin
                ):
                    return [policy_ids[a], policy_ids[b], policy_ids[c]]
    return None


def build_matchup_matrix(
    policies, *, games_per_pair, seed_base, output_dir, progress=None
):
    ids = list(policies)
    matrix = np.full((len(ids), len(ids)), 0.5, dtype=float)
    pair_count = max(1, (games_per_pair + 1) // 2)
    for row in range(len(ids)):
        for column in range(row + 1, len(ids)):
            result = paired_actor_evaluation(
                policies[ids[row]],
                policies[ids[column]],
                pair_count=pair_count,
                seed_base=seed_base + row * 100_000 + column * 1_000,
            )
            score = float(np.mean(result["pair_scores"]))
            matrix[row, column] = score
            matrix[column, row] = 1.0 - score
            if progress:
                progress.update(result["games"], matchup=f"{ids[row]}_vs_{ids[column]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.5.1",
        "policy_ids": ids,
        "matrix": matrix.tolist(),
        "games_per_pair": pair_count * 2,
        "potential_non_transitive_cycle": detect_cycle(matrix, ids),
    }
    (output / "matchup_matrix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output / "matchup_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", *ids])
        for policy_id, values in zip(ids, matrix, strict=True):
            writer.writerow([policy_id, *values])
    warning = (
        "\n\nPotential non-transitive cycle detected."
        if payload["potential_non_transitive_cycle"]
        else ""
    )
    (output / "report.md").write_text(
        "# League matchup matrix\n\n"
        + "\n".join(
            f"- {policy_id}: {dict(zip(ids, values, strict=True))}"
            for policy_id, values in zip(ids, matrix.tolist(), strict=True)
        )
        + warning,
        encoding="utf-8",
    )
    return payload
