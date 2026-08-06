from __future__ import annotations

import numpy as np


def posterior_score(games, wins, ties, prior_alpha=1.0, prior_beta=1.0):
    return (wins + 0.5 * ties + prior_alpha) / (games + prior_alpha + prior_beta)


def pfsp_weight(score: float, alpha=1.0, epsilon=0.05) -> float:
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("PFSP score must be finite and between zero and one")
    return float((epsilon + 4.0 * score * (1.0 - score)) ** alpha)


def pfsp_probabilities(scores, alpha=1.0, epsilon=0.05) -> np.ndarray:
    values = np.asarray([pfsp_weight(s, alpha, epsilon) for s in scores], dtype=float)
    if not len(values):
        return values
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid PFSP weights")
    return values / total


def sample_pfsp(ids, scores, rng, alpha=1.0, epsilon=0.05):
    if not ids:
        raise ValueError("cannot sample an empty opponent pool")
    probabilities = pfsp_probabilities(scores, alpha, epsilon)
    index = int(rng.choice(len(ids), p=probabilities))
    return ids[index], float(probabilities[index]), probabilities
