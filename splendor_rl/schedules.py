from __future__ import annotations


def linear_learning_rate(
    initial: float, minimum: float, transitions: int, total: int, enabled: bool = True
) -> float:
    if not enabled:
        return initial
    progress = min(max(transitions / max(total, 1), 0.0), 1.0)
    return initial + progress * (minimum - initial)


def entropy_coefficient(
    start: float, end: float, fraction: float, transitions: int, total: int
) -> float:
    anneal_transitions = max(1.0, total * fraction)
    progress = min(max(transitions / anneal_transitions, 0.0), 1.0)
    return start + progress * (end - start)


def next_interval_threshold(transitions: int, interval: int | None) -> int | None:
    if interval is None:
        return None
    return (transitions // interval + 1) * interval


def apply_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value
