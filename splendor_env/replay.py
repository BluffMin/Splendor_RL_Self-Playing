"""Replay and verify recorded Splendor episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .core import SplendorGame


class ReplayVerificationError(RuntimeError):
    """Raised at the first deterministic replay mismatch."""


def load_recording(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_recording(document: dict[str, Any]) -> SplendorGame:
    """Re-run actions and verify every hash and final holding."""
    game = SplendorGame(document["config"]["num_players"], seed=document["seed"])
    if game.state_hash() != document["initial_state_hash"]:
        raise ReplayVerificationError("initial state hash mismatch")
    for event in document["events"]:
        if game.state_hash() != event["pre_state_hash"]:
            raise ReplayVerificationError(
                f"event {event['decision_id']} pre-state hash mismatch"
            )
        game.step(event["action_id"])
        if game.state_hash() != event["post_state_hash"]:
            raise ReplayVerificationError(
                f"event {event['decision_id']} post-state hash mismatch"
            )
    result = document["result"]
    if game.state_hash() != result["final_state_hash"]:
        raise ReplayVerificationError("final state hash mismatch")
    if game.final_ranking() != result["ranking"]:
        raise ReplayVerificationError("final ranking mismatch")
    expected_players = result["final_summary"]["players"]
    actual_players = game.final_summary()["players"]
    for expected, actual in zip(expected_players, actual_players, strict=True):
        for field in ("score", "purchased_cards", "reserved_cards", "nobles"):
            if actual[field] != expected[field]:
                raise ReplayVerificationError(
                    f"player {actual['player_id']} {field} mismatch"
                )
    return game


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
    """Re-execute and return perspective-safe textual frames."""
    game = SplendorGame(document["config"]["num_players"], seed=document["seed"])
    frames: list[str] = []
    events = document["events"]
    stop = len(events) if to_event is None else min(to_event + 1, len(events))
    for index, event in enumerate(events[:stop]):
        mask_text = ""
        if show_action_mask:
            mask_text = f"\nlegal_actions={game.legal_actions()}"
        result = game.step(event["action_id"])
        if index >= from_event and (not turn_only or result.turn_ended):
            frames.append(
                f"=== event {index}: {event['action_text']} ===\n"
                + game.render(perspective=perspective, omniscient=omniscient)
                + mask_text
            )
    frames.append("\n" + game.render_final_summary())
    return "\n\n".join(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--perspective", type=int, default=0)
    parser.add_argument("--omniscient", action="store_true")
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--from-event", type=int, default=0)
    parser.add_argument("--to-event", type=int)
    parser.add_argument("--turn-only", action="store_true")
    parser.add_argument("--show-action-mask", action="store_true")
    parser.add_argument("--output-text")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    document = load_recording(args.recording)
    try:
        if args.verify:
            game = verify_recording(document)
            print(f"verification: PASS ({game.state_hash()})")
            return 0
        text = replay_text(
            document,
            perspective=args.perspective,
            omniscient=args.omniscient,
            turn_only=args.turn_only,
            from_event=args.from_event,
            to_event=args.to_event,
            show_action_mask=args.show_action_mask,
        )
        if args.output_text:
            Path(args.output_text).write_text(text, encoding="utf-8")
        elif args.step:
            chunks = text.split("\n\n=== ")
            frames = [chunks[0]] + ["=== " + chunk for chunk in chunks[1:]]
            index = 0
            while 0 <= index < len(frames):
                print(frames[index])
                command = input("Enter/n=next, p=previous, q=quit, f=final: ").strip().lower()
                if command == "q":
                    break
                if command == "p":
                    index = max(0, index - 1)
                elif command == "f":
                    index = len(frames) - 1
                else:
                    index += 1
        else:
            for frame in text.split("\n\n==="):
                print(frame if frame.startswith("===") else "===" + frame)
                if args.delay:
                    time.sleep(args.delay)
        return 0
    except ReplayVerificationError as exc:
        print(f"verification: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
