"""Schema-versioned episode recording and backward-compatible loading."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .core import SplendorGame
from .event_schema import SCHEMA_VERSION, DecisionEvent, TurnRecord, build_turn_records

RecordLevel = Literal["summary", "actions", "full"]


@dataclass(frozen=True, slots=True)
class EpisodeLog:
    document: dict[str, Any]
    source_schema_version: str
    normalized_schema_version: str = SCHEMA_VERSION
    is_legacy: bool = False
    migration_warnings: tuple[str, ...] = ()
    migration_quality: str = "exact"
    replay_verifiable: bool = True


class EpisodeRecorder:
    def __init__(
        self,
        output_path: str | Path,
        record_level: RecordLevel = "full",
        omniscient: bool = True,
        snapshot_on_turn_end: bool = True,
    ) -> None:
        if record_level not in {"summary", "actions", "full"}:
            raise ValueError("record_level must be summary, actions, or full")
        self.output_path, self.record_level, self.omniscient = (
            Path(output_path),
            record_level,
            bool(omniscient),
        )
        self.snapshot_on_turn_end = bool(snapshot_on_turn_end)
        self.game: SplendorGame | None = None
        self.events: list[dict[str, Any]] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    def attach(self, game: SplendorGame) -> None:
        if self.game is not None:
            raise RuntimeError("recorder is already attached")
        self.game = game
        self.initial_state = game.to_state_dict(omniscient=self.omniscient)
        self.initial_state_hash = game.state_hash()
        game.add_event_listener(self._on_event)

    def _on_event(self, event: dict[str, Any]) -> None:
        if self.record_level in {"actions", "full"}:
            self.events.append(dict(event))

    def finalize(self) -> dict[str, Any]:
        if self.game is None:
            raise RuntimeError("recorder is not attached")
        self.game.remove_event_listener(self._on_event)
        turns = [t.to_dict() for t in build_turn_records(self.events)]
        summary = self.game.final_summary()
        metadata = {
            "num_players": self.game.num_players,
            "seed": self.game.seed,
            "record_level": self.record_level,
            "omniscient": self.omniscient,
            "started_at": self.started_at,
        }
        doc = {
            "schema_version": SCHEMA_VERSION,
            "game_metadata": metadata,
            "initial_state": self.initial_state,
            "initial_state_hash": self.initial_state_hash,
            "events": self.events,
            "turns": turns,
            "final_summary": summary,
            "final_state_hash": self.game.state_hash(),
            "replay_verifiable": True,
            # Compatibility aliases for v0.3.1 consumers.
            "config": {"num_players": self.game.num_players},
            "seed": self.game.seed,
            "snapshots": [
                {
                    "decision_id": e["decision_id"],
                    "turn_id": e["player_turn_id"],
                    "state": e["post_snapshot"],
                    "state_hash": e["post_state_hash"],
                }
                for e in self.events
                if e["turn_completed"]
            ],
            "result": {
                "end_reason": self.game.end_reason,
                "turns": self.game.turns_completed,
                "rounds": self.game.round_id,
                "decisions": self.game.decision_id,
                "ranking": self.game.final_ranking(),
                "winner_ids": self.game.winner_ids(),
                "final_state_hash": self.game.state_hash(),
                "final_summary": summary,
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return doc


def _detect_json_schema(raw: dict[str, Any]) -> str:
    version = raw.get("schema_version")
    if version == SCHEMA_VERSION:
        return SCHEMA_VERSION
    if version in {"0.3.0", "0.3.1"}:
        return str(version)
    if version == "1.0":
        return "0.3.1-compatible"
    return "legacy-unversioned"


def _normalize_legacy_json(raw: dict[str, Any], source: str) -> EpisodeLog:
    warnings = ["Legacy decision events were normalized to the v0.3.2 field names."]
    config = raw.get("config", raw.get("game_metadata", {}))
    seed = raw.get("seed", raw.get("game_metadata", {}).get("seed"))
    events: list[dict[str, Any]] = []
    sub: dict[int, int] = {}
    old = raw.get("events", [])
    for i, item in enumerate(old):
        turn = int(item.get("player_turn_id", item.get("turn_id", 0)))
        si = sub.get(turn, 0)
        sub[turn] = si + 1
        next_player = None
        if item.get("turn_completed"):
            next_player = (int(item.get("player", 0)) + 1) % int(
                config.get("num_players", 2)
            )
        event = dict(item)
        event.update(
            {
                "schema_version": SCHEMA_VERSION,
                "event_sequence_id": i,
                "player_turn_id": turn,
                "subdecision_index": si,
                "acting_player": int(item.get("player", 0)),
                "phase_before": item.get("phase", "normal"),
                "phase_after": item.get("phase_after", "unknown"),
                "turn_started": si == 0,
                "next_player": next_player,
            }
        )
        events.append(DecisionEvent.from_dict(event).to_dict())
    exact = bool(
        events
        and all(
            e.get("action_id") is not None
            and e.get("pre_state_hash")
            and e.get("post_state_hash")
            for e in events
        )
    )
    # Re-execution restores exact native snapshots and phase/next metadata when possible.
    if exact:
        try:
            game = SplendorGame(int(config["num_players"]), seed=seed)
            regenerated: list[dict[str, Any]] = []
            game.add_event_listener(lambda e: regenerated.append(dict(e)))
            for old_event in old:
                if game.state_hash() != old_event["pre_state_hash"]:
                    raise ValueError("pre-state hash mismatch")
                game.step(int(old_event["action_id"]))
                if game.state_hash() != old_event["post_state_hash"]:
                    raise ValueError("post-state hash mismatch")
            events = regenerated
        except (KeyError, TypeError, ValueError, RuntimeError, AssertionError) as exc:
            exact = False
            warnings.append(f"Exact replay reconstruction failed: {exc}")
    if not exact:
        warnings.append(
            "Original subdecision ordering or snapshots cannot be reconstructed exactly."
        )
    turns = [t.to_dict() for t in build_turn_records(events)]
    final_summary = raw.get(
        "final_summary", raw.get("result", {}).get("final_summary", {})
    )
    final_hash = raw.get(
        "final_state_hash", raw.get("result", {}).get("final_state_hash", "")
    )
    doc = {
        "schema_version": SCHEMA_VERSION,
        "game_metadata": {**config, "seed": seed},
        "initial_state": raw.get("initial_state", {}),
        "initial_state_hash": raw.get("initial_state_hash", ""),
        "events": events,
        "turns": turns,
        "final_summary": final_summary,
        "final_state_hash": final_hash,
        "replay_verifiable": exact,
        "config": {"num_players": int(config.get("num_players", 2))},
        "seed": seed,
        "result": raw.get(
            "result",
            {
                "final_state_hash": final_hash,
                "final_summary": final_summary,
                "ranking": final_summary.get("ranking", []),
            },
        ),
    }
    return EpisodeLog(
        doc,
        source,
        is_legacy=True,
        migration_warnings=tuple(warnings),
        migration_quality="exact" if exact else "best_effort",
        replay_verifiable=exact,
    )


def _load_legacy_text(path: Path) -> EpisodeLog:
    text = path.read_text(encoding="utf-8", errors="replace")
    actor_match = re.search(r"(?:actor|player)\s*[=:]\s*P?(\d+)", text, re.IGNORECASE)
    turn_match = re.search(r"turn(?:_id| counter)?\s*[=:]\s*(\d+)", text, re.IGNORECASE)
    actor, turn = (
        int(actor_match.group(1)) if actor_match else 0,
        int(turn_match.group(1)) if turn_match else 0,
    )
    event = DecisionEvent(
        SCHEMA_VERSION,
        0,
        None,
        max(0, turn - 1),
        0,
        0,
        actor,
        "payment",
        "unknown",
        None,
        "choose_payment_plan",
        {},
        "choose payment plan 0",
        False,
        True,
        True,
        None,
        "",
        "",
    ).to_dict()
    record = build_turn_records([event])[0]
    record = TurnRecord(**record.__dict__) if False else record
    td = record.to_dict()
    td["primary_action_type"] = "inferred_purchase"
    td["primary_action_text"] = "purchase inferred from legacy text"
    warnings = (
        "Primary buy action was not present; inferred from the payment-only legacy entry.",
        "Original subdecision ordering cannot be reconstructed exactly.",
    )
    doc = {
        "schema_version": SCHEMA_VERSION,
        "game_metadata": {"source": "legacy_text"},
        "initial_state": {},
        "events": [event],
        "turns": [td],
        "final_summary": {},
        "final_state_hash": "",
        "replay_verifiable": False,
    }
    return EpisodeLog(
        doc,
        "legacy-text",
        is_legacy=True,
        migration_warnings=warnings,
        migration_quality="best_effort",
        replay_verifiable=False,
    )


def load_episode_log(path: str | Path, *, allow_legacy: bool = True) -> EpisodeLog:
    source_path = Path(path)
    if source_path.suffix.lower() == ".txt":
        if not allow_legacy:
            raise ValueError("legacy logs are disabled")
        return _load_legacy_text(source_path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    source = _detect_json_schema(raw)
    if source == SCHEMA_VERSION:
        return EpisodeLog(raw, source)
    if not allow_legacy:
        raise ValueError(f"legacy schema {source} is disabled")
    return _normalize_legacy_json(raw, source)


def assert_compatible_log_schemas(
    logs: list[EpisodeLog],
    *,
    include_exact_migrations: bool = True,
    include_best_effort: bool = False,
) -> None:
    for log in logs:
        if log.migration_quality == "best_effort" and not include_best_effort:
            raise ValueError("best-effort log excluded from quantitative aggregation")
        if (
            log.is_legacy
            and log.migration_quality == "exact"
            and not include_exact_migrations
        ):
            raise ValueError("exact migration excluded")


def append_games_summary_csv(
    path: str | Path, game_id: str, seed: int, game: SplendorGame
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rank = {p: g["rank"] for g in game.final_ranking() for p in g["players"]}
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
    for i in range(4):
        p = game.players[i] if i < game.num_players else None
        for key, value in {
            "score": None if p is None else p.score,
            "rank": None if p is None else rank[i],
            "cards": None if p is None else len(p.purchased),
            "reserves": None if p is None else len(p.reserved),
            "nobles": None if p is None else len(p.nobles),
        }.items():
            row[f"{key}_p{i}"] = "" if value is None else value
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
