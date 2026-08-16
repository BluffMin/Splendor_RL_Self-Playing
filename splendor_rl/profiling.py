from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

PROFILE_CATEGORIES = (
    "environment_step", "observation_build", "action_mask_build",
    "actor_inference", "critic_inference", "IPC_wait", "worker_wait",
    "batch_assembly", "rollout_bookkeeping", "trajectory_finalize", "GAE",
    "PPO_forward", "PPO_backward", "optimizer_step", "checkpoint_save",
    "fixed_bot_evaluation", "promotion_evaluation", "exploiter_evaluation",
    "meta_matchup_evaluation", "meta_solver", "other",
)


class WallProfiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.started = time.perf_counter()
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def measure(self, category):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[category] += time.perf_counter() - started
            self.calls[category] += 1

    def add(self, category, seconds, calls=1):
        if self.enabled:
            self.seconds[category] += float(seconds)
            self.calls[category] += int(calls)

    def report(self):
        wall = max(time.perf_counter() - self.started, 1e-12)
        accounted = sum(self.seconds.values())
        denominator = max(wall, accounted)
        rows = {}
        for key in PROFILE_CATEGORIES:
            seconds = self.seconds[key]
            calls = self.calls[key]
            rows[key] = {
                "seconds": seconds,
                "calls": calls,
                "average_seconds": seconds / calls if calls else 0.0,
                "wall_fraction": seconds / denominator,
            }
        rows["other"]["seconds"] += max(0.0, wall - accounted)
        rows["other"]["wall_fraction"] = rows["other"]["seconds"] / denominator
        return {"wall_seconds": wall, "components": rows}

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
