"""Replay and verify v0.3.2 and legacy Splendor episode logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import SplendorGame
from .event_schema import DecisionEvent, build_turn_records, summarize_turn
from .recording import load_episode_log


class ReplayVerificationError(RuntimeError):
    pass


def load_recording(path: str | Path) -> dict[str, Any]:
    return load_episode_log(path).document


def verify_recording(document: dict[str, Any]) -> SplendorGame:
    metadata = document.get("game_metadata", {})
    game = SplendorGame(
        int(metadata.get("num_players", document.get("config", {})["num_players"])),
        seed=metadata.get("seed", document.get("seed")),
    )
    initial_hash = document.get("initial_state_hash", "")
    if initial_hash and game.state_hash() != initial_hash:
        raise ReplayVerificationError("initial state hash mismatch")
    emitted: list[dict[str, Any]] = []
    game.add_event_listener(lambda e: emitted.append(e))
    for event in document["events"]:
        if event.get("action_id") is None:
            raise ReplayVerificationError("event is not replay-verifiable")
        if game.state_hash() != event["pre_state_hash"]:
            raise ReplayVerificationError(
                f"event {event.get('decision_id')} pre-state hash mismatch"
            )
        game.step(event["action_id"])
        if game.state_hash() != event["post_state_hash"]:
            raise ReplayVerificationError(
                f"event {event.get('decision_id')} post-state hash mismatch"
            )
    expected = [DecisionEvent.from_dict(e) for e in document["events"]]
    actual = [DecisionEvent.from_dict(e) for e in emitted]
    for e, a in zip(expected, actual, strict=True):
        if (
            e.player_turn_id,
            e.round_id,
            e.acting_player,
            e.turn_completed,
            e.next_player,
        ) != (
            a.player_turn_id,
            a.round_id,
            a.acting_player,
            a.turn_completed,
            a.next_player,
        ):
            raise ReplayVerificationError(
                f"turn semantics mismatch at decision {e.decision_id}"
            )
    calculated = build_turn_records(expected)
    stored = document.get("turns", [])
    if stored and [
        (t.player_turn_id, t.round_id, t.acting_player, t.next_player, t.decision_ids)
        for t in calculated
    ] != [
        (
            t["player_turn_id"],
            t["round_id"],
            t["acting_player"],
            t.get("next_player"),
            tuple(t["decision_ids"]),
        )
        for t in stored
    ]:
        raise ReplayVerificationError("turn grouping mismatch")
    if any(t.round_id != t.player_turn_id // game.num_players for t in calculated):
        raise ReplayVerificationError("round progression mismatch")
    final_hash = document.get(
        "final_state_hash", document.get("result", {}).get("final_state_hash")
    )
    if final_hash and game.state_hash() != final_hash:
        raise ReplayVerificationError("final state hash mismatch")
    return game


def verification_report(document: dict[str, Any]) -> str:
    verify_recording(document)
    return ("Decision events: PASS\nTurn grouping: PASS\nRound progression: PASS\n"
            "Acting player sequence: PASS\nFinal state hash: PASS")


def replay_text(
    document: dict[str, Any],
    *,
    perspective: int = 0,
    omniscient: bool = False,
    turn_only: bool = False,
    from_event: int = 0,
    to_event: int | None = None,
    show_action_mask: bool = False,
) -> str:
    events = document["events"]
    if turn_only:
        turns = build_turn_records(events)
        return (
            "\n\n".join(summarize_turn(t) for t in turns)
            + "\n\n"
            + json.dumps(
                document.get("final_summary", {}), ensure_ascii=False, indent=2
            )
        )
    metadata = document.get("game_metadata", {})
    game = SplendorGame(
        int(metadata.get("num_players", document.get("config", {})["num_players"])),
        seed=metadata.get("seed", document.get("seed")),
    )
    frames = []
    stop = len(events) if to_event is None else min(to_event + 1, len(events))
    per_turn = {}
    for e in events:
        per_turn[e.get("player_turn_id", e.get("turn_id", 0))] = (
            per_turn.get(e.get("player_turn_id", e.get("turn_id", 0)), 0) + 1
        )
    for index, event in enumerate(events[:stop]):
        before = f"\nlegal_actions={game.legal_actions()}" if show_action_mask else ""
        game.step(event["action_id"])
        if index >= from_event:
            turn = event.get("player_turn_id", event.get("turn_id", 0))
            sub = event.get("subdecision_index", 0)
            header = f"Round {event.get('round_id', 0) + 1} · Turn {turn + 1} · Decision {sub + 1}/{per_turn[turn]} · P{event.get('acting_player', event.get('player'))}"
            frames.append(
                f"=== {header}\nPhase: {event.get('phase_before', event.get('phase'))}\nDecision: {event['action_text']} ===\n"
                + game.render(perspective=perspective, omniscient=omniscient)
                + before
            )
    frames.append("\n" + game.render_final_summary())
    return "\n\n".join(frames)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("recording")
    p.add_argument("--perspective", type=int, default=0)
    p.add_argument("--omniscient", action="store_true")
    p.add_argument("--step", action="store_true")
    p.add_argument("--delay", type=float, default=0)
    p.add_argument("--from-event", type=int, default=0)
    p.add_argument("--to-event", type=int)
    p.add_argument("--turn-only", action="store_true")
    p.add_argument("--show-action-mask", action="store_true")
    p.add_argument("--output-text")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)
    try:
        doc = load_episode_log(a.recording).document
        if a.verify:
            print(verification_report(doc))
            return 0
        value = replay_text(
            doc,
            perspective=a.perspective,
            omniscient=a.omniscient,
            turn_only=a.turn_only,
            from_event=a.from_event,
            to_event=a.to_event,
            show_action_mask=a.show_action_mask,
        )
        if a.output_text:
            Path(a.output_text).write_text(value, encoding="utf-8")
        else:
            print(value)
        return 0
    except (ReplayVerificationError, ValueError, KeyError) as exc:
        print(f"verification: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
