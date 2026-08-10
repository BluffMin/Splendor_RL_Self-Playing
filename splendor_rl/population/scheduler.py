from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeficitScheduler:
    weights: dict[str, float]
    counts: dict[str, int] | None = None
    steps: int = 0

    def __post_init__(self):
        self.weights = {
            key: float(value) for key, value in self.weights.items() if value > 0
        }
        if not self.weights or abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("enabled scheduler weights must sum to one")
        self.counts = {key: 0 for key in self.weights} | (self.counts or {})

    def next(self) -> str:
        role = max(
            self.weights,
            key=lambda key: (
                (self.steps + 1) * self.weights[key] - self.counts.get(key, 0),
                -list(self.weights).index(key),
            ),
        )
        self.counts[role] = self.counts.get(role, 0) + 1
        self.steps += 1
        return role

    def state_dict(self):
        return {"weights": self.weights, "counts": self.counts, "steps": self.steps}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["weights"], dict(state["counts"]), int(state["steps"]))
