from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MetaStrategyResult:
    probabilities: np.ndarray
    estimated_value: float
    iterations: int
    convergence_gap: float

    def to_dict(self):
        value = asdict(self)
        value["probabilities"] = self.probabilities.tolist()
        return value


def antisymmetrize_score_matrix(score_matrix):
    raw = np.asarray(score_matrix, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.isfinite(raw).all():
        raise ValueError("payoff matrix must be a finite square matrix")
    centered = 2.0 * raw - 1.0
    result = 0.5 * (centered - centered.T)
    np.fill_diagonal(result, 0.0)
    return result


def solve_symmetric_meta_strategy(payoff_matrix, *, iterations=20_000, seed=0):
    del seed  # deterministic regret matching intentionally has no random tie breaking.
    matrix = np.asarray(payoff_matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("payoff matrix must be finite and square")
    count = len(matrix)
    if count == 0:
        raise ValueError("meta population must not be empty")
    regrets = np.zeros(count, dtype=float)
    strategy_sum = np.zeros(count, dtype=float)
    strategy = np.full(count, 1.0 / count)
    for _ in range(iterations):
        positive = np.maximum(regrets, 0.0)
        strategy = (
            positive / positive.sum() if positive.sum() else np.full(count, 1.0 / count)
        )
        strategy_sum += strategy
        action_values = matrix @ strategy
        regrets += action_values - float(strategy @ action_values)
    probabilities = strategy_sum / strategy_sum.sum()
    values = matrix @ probabilities
    value = float(probabilities @ values)
    gap = float(values.max() - value)
    return MetaStrategyResult(probabilities, value, iterations, gap)


def farthest_point_selection(policy_ids, matrix, required, maximum, hashes=None):
    ids = list(policy_ids)
    hashes = hashes or {key: key for key in ids}
    selected, seen_hashes = [], set()
    for policy_id in required:
        if policy_id in ids and hashes[policy_id] not in seen_hashes:
            selected.append(policy_id)
            seen_hashes.add(hashes[policy_id])
    while len(selected) < min(maximum, len(ids)):
        choices = [
            key for key in ids if key not in selected and hashes[key] not in seen_hashes
        ]
        if not choices:
            break
        if not selected:
            chosen = choices[0]
        else:
            chosen = max(
                choices,
                key=lambda key: min(
                    float(
                        np.mean(
                            np.abs(matrix[ids.index(key)] - matrix[ids.index(item)])
                        )
                    )
                    for item in selected
                ),
            )
        selected.append(chosen)
        seen_hashes.add(hashes[chosen])
    return selected


def empirical_exploitability_proxy(exploiter_scores):
    return max((float(score) for score in exploiter_scores), default=0.5) - 0.5


def detect_cycles(matrix, policy_ids, threshold=0.55, maximum=10):
    matrix = np.asarray(matrix, dtype=float)
    cycles = []
    for a in range(len(policy_ids)):
        for b in range(a + 1, len(policy_ids)):
            for c in range(b + 1, len(policy_ids)):
                for x, y, z in ((a, b, c), (a, c, b)):
                    if (
                        matrix[x, y] >= threshold
                        and matrix[y, z] >= threshold
                        and matrix[z, x] >= threshold
                    ):
                        cycles.append(
                            {
                                "policies": [
                                    policy_ids[x],
                                    policy_ids[y],
                                    policy_ids[z],
                                ],
                                "scores": [
                                    float(matrix[x, y]),
                                    float(matrix[y, z]),
                                    float(matrix[z, x]),
                                ],
                            }
                        )
                        if len(cycles) >= maximum:
                            return cycles
    return cycles
