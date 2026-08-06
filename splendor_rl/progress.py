from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from tqdm.auto import tqdm


class ProgressMode(str, Enum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True)
class ProgressConfig:
    mode: ProgressMode = ProgressMode.AUTO
    refresh_seconds: float = 1.0
    dynamic_ncols: bool = True

    def __post_init__(self) -> None:
        if self.refresh_seconds <= 0:
            raise ValueError("progress_refresh_seconds must be positive")


def progress_enabled(config: ProgressConfig, stream: TextIO = sys.stderr) -> bool:
    if config.mode == ProgressMode.NEVER:
        return False
    if config.mode == ProgressMode.ALWAYS:
        return True
    return "PYTEST_CURRENT_TEST" not in os.environ and bool(stream.isatty())


class NullProgress:
    enabled = False
    n = 0
    total = 0

    def update(self, amount=1, **_fields) -> None:
        self.n += amount

    def status(self, _description: str, **_fields) -> None:
        pass

    def update_training(self, amount, **_fields) -> None:
        self.update(amount)

    def write(self, message: str) -> None:
        print(message)

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def close(self) -> None:
        pass


class _TqdmProgress:
    enabled = True

    def __init__(
        self, description, total, initial, config, *, position=0, leave=True, unit="tr"
    ):
        self.bar = tqdm(
            total=total,
            initial=initial,
            desc=description,
            mininterval=config.refresh_seconds,
            dynamic_ncols=config.dynamic_ncols,
            position=position,
            leave=leave,
            file=sys.stderr,
            unit=unit,
        )

    @property
    def n(self):
        return self.bar.n

    @property
    def total(self):
        return self.bar.total

    def _fields(self, fields):
        return {key: value for key, value in fields.items() if value is not None}

    def update(self, amount=1, **fields) -> None:
        target = self.bar.n + amount
        if self.bar.total is not None and target > self.bar.total:
            self.bar.total = target
        if fields:
            self.bar.set_postfix(self._fields(fields), refresh=False)
        self.bar.update(amount)

    def status(self, description: str, **fields) -> None:
        self.bar.set_description_str(description, refresh=False)
        if fields:
            self.bar.set_postfix(self._fields(fields), refresh=False)
        self.bar.refresh()

    def write(self, message: str) -> None:
        tqdm.write(message, file=sys.stderr)

    def pause(self) -> None:
        self.bar.clear()

    def resume(self) -> None:
        self.bar.unpause()

    def close(self) -> None:
        self.bar.close()


class TrainingProgress(_TqdmProgress):
    def __init__(self, total, initial, schedule_total, config):
        self.schedule_total = schedule_total
        super().__init__(f"Training to {total:,}", total, initial, config)

    def update_training(self, amount, *, transitions, update_index, episodes, metrics):
        schedule = 100.0 * transitions / max(1, self.schedule_total)
        self.update(
            amount,
            upd=update_index,
            games=episodes,
            schedule=f"{schedule:.1f}%",
            **metrics,
        )


class EvaluationProgress(_TqdmProgress):
    def __init__(self, total, transition_count, config, *, position=1):
        super().__init__(
            f"Evaluation step {transition_count:,}",
            total,
            0,
            config,
            position=position,
            leave=False,
            unit="game",
        )


def make_training_progress(total, initial, schedule_total, config):
    if not progress_enabled(config):
        progress = NullProgress()
        progress.total = total
        progress.n = initial
        return progress
    return TrainingProgress(total, initial, schedule_total, config)


def make_evaluation_progress(total, transition_count, config):
    if not progress_enabled(config):
        progress = NullProgress()
        progress.total = total
        return progress
    return EvaluationProgress(total, transition_count, config)
