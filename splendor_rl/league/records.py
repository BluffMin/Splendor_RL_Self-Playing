from __future__ import annotations

import json
from pathlib import Path

from .sampling import posterior_score
from .types import MatchRecord


class MatchRecords:
    def __init__(self, path, recent_window=400):
        self.path = Path(path)
        self.recent_window = recent_window
        self.records: dict[str, MatchRecord] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = {
                key: MatchRecord.from_dict(value) for key, value in raw.items()
            }

    def get(self, opponent_id):
        return self.records.setdefault(opponent_id, MatchRecord(opponent_id))

    def add(self, opponent_id, score, seat, transition, final_score=None):
        self.get(opponent_id).add(
            score, seat, transition, self.recent_window, final_score
        )

    def score(self, opponent_id, prior_alpha=1.0, prior_beta=1.0):
        record = self.get(opponent_id)
        if record.recent_games:
            wins = sum(value == 1 for value in record.recent_results)
            ties = sum(value == 0.5 for value in record.recent_results)
            return posterior_score(
                record.recent_games, wins, ties, prior_alpha, prior_beta
            )
        return posterior_score(
            record.games, record.wins, record.ties, prior_alpha, prior_beta
        )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {key: value.to_dict() for key, value in self.records.items()}, indent=2
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)
