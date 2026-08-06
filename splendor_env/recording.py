"""Passive JSON/CSV episode recording for deterministic Splendor games."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .core import SplendorGame

RecordLevel = Literal["summary", "actions", "full"]


class EpisodeRecorder:
    """Observe a game without changing observations, masks, or rewards."""

    def __init__(
        self,
        output_path: str | Path,
        record_level: RecordLevel = "full",
        omniscient: bool = True,
        snapshot_on_turn_end: bool = True,
    ) -> None:
        if record_level not in {"summary", "actions", "full"}:
            raise ValueError("record_level must be summary, actions, or full")
        self.output_path = Path(output_path)
        self.record_level = record_level
        self.omniscient = bool(omniscient)
        self.snapshot_on_turn_end = bool(snapshot_on_turn_end)
        self.game: SplendorGame | None = None
        self.events: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    def attach(self, game: SplendorGame) -> None:
        """Attach once and capture the initial deterministic state."""
        if self.game is not None:
            raise RuntimeError("recorder is already attached")
        self.game = game
        self.initial_state_hash = game.state_hash()
        game.add_event_listener(self._on_event)

    def _on_event(self, event: dict[str, Any]) -> None:
        if self.record_level in {"actions", "full"}:
            self.events.append(dict(event))
        if self.record_level == "full" and (
            not self.snapshot_on_turn_end or event["turn_completed"]
        ):
            assert self.game is not None
            self.snapshots.append(
                {
                    "decision_id": event["decision_id"],
                    "turn_id": event["turn_id"],
                    "state": self.game.to_state_dict(omniscient=self.omniscient),
                    "state_hash": self.game.state_hash(),
                }
            )

    def finalize(self) -> dict[str, Any]:
        """Write the episode JSON and return the document."""
        if self.game is None:
            raise RuntimeError("recorder is not attached")
        self.game.remove_event_listener(self._on_event)
        document: dict[str, Any] = {
            "schema_version": "1.0",
            "record_level": self.record_level,
            "omniscient": self.omniscient,
            "config": {"num_players": self.game.num_players},
            "seed": self.game.seed,
            "started_at": self.started_at,
            "initial_state_hash": self.initial_state_hash,
            "events": self.events,
            "snapshots": self.snapshots,
            "result": {
                "end_reason": self.game.end_reason,
                "turns": self.game.turns_completed,
                "rounds": self.game.round_id,
                "decisions": self.game.decision_id,
                "ranking": self.game.final_ranking(),
                "winner_ids": self.game.winner_ids(),
                "final_state_hash": self.game.state_hash(),
                "final_summary": self.game.final_summary(),
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return document


def append_games_summary_csv(
    path: str | Path, game_id: str, seed: int, game: SplendorGame
) -> None:
    """Append one stable, four-seat-wide game summary row."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rank_by_player = {
        player: group["rank"]
        for group in game.final_ranking()
        for player in group["players"]
    }
    row: dict[str, Any] = {
        "game_id": game_id,
        "seed": seed,
        "num_players": game.num_players,
        "end_reason": game.end_reason,
        "rounds": game.round_id,
        "turns": game.turns_completed,
        "decisions": game.decision_id,
        "winner_ids": ",".join(map(str, game.winner_ids())),
        "final_state_hash": game.state_hash(),
    }
    for index in range(4):
        player = game.players[index] if index < game.num_players else None
        row[f"score_p{index}"] = "" if player is None else player.score
        row[f"rank_p{index}"] = "" if player is None else rank_by_player[index]
        row[f"cards_p{index}"] = "" if player is None else len(player.purchased)
        row[f"reserves_p{index}"] = "" if player is None else len(player.reserved)
        row[f"nobles_p{index}"] = "" if player is None else len(player.nobles)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
