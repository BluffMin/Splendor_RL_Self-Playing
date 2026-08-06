"""Self-contained browser replay and live-state HTML exporter."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from ..core import NoLegalActionError, SplendorGame
from ..replay import load_recording
from .view_model import (
    Perspective,
    board_view_from_game,
    compute_state_delta,
)

DataMode = Literal["omniscient-data", "perspective-sanitized-data"]


def _legal_actions_or_empty(game: SplendorGame) -> list[int]:
    try:
        return game.legal_actions()
    except NoLegalActionError:
        return []


def _asset_text(name: str) -> str:
    return files("splendor_env.visualization.assets").joinpath(name).read_text(encoding="utf-8")


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_document(payload: dict[str, Any]) -> str:
    template = _asset_text("viewer_template.html")
    return (
        template.replace("/*__VIEWER_CSS__*/", _asset_text("viewer.css"))
        .replace("/*__REPLAY_DATA__*/", _safe_json(payload))
        .replace("/*__VIEWER_JS__*/", _asset_text("viewer.js"))
    )


def _views_for_game(
    game: SplendorGame,
    *,
    data_mode: DataMode,
    perspective: Perspective,
) -> tuple[dict[str, Any], list[str]]:
    allowed: list[Perspective]
    if data_mode == "perspective-sanitized-data":
        allowed = [perspective]
    else:
        allowed = ["current", *range(game.num_players), "omniscient"]
    views: dict[str, Any] = {}
    if data_mode == "omniscient-data":
        views["omniscient"] = board_view_from_game(
            game, perspective="omniscient", layout="table"
        ).to_dict()
    else:
        views[str(perspective)] = board_view_from_game(
            game, perspective=perspective, layout="table"
        ).to_dict()
    return views, [str(item) for item in allowed]


def export_replay(
    recording_path: str | Path,
    output_path: str | Path,
    *,
    data_mode: DataMode = "omniscient-data",
    perspective: Perspective = "current",
) -> Path:
    """Re-execute a recording and export decision-level browser frames."""
    document = load_recording(recording_path)
    game = SplendorGame(document["config"]["num_players"], seed=document["seed"])
    initial_views, perspectives = _views_for_game(
        game, data_mode=data_mode, perspective=perspective
    )
    frames: list[dict[str, Any]] = [
        {"views": initial_views, "event": None, "delta": {}, "turn_completed": False, "legal_actions": _legal_actions_or_empty(game)}
    ]
    for event in document["events"]:
        pre = game.to_state_dict()
        result = game.step(event["action_id"])
        post = game.to_state_dict()
        legal_actions = [] if game.done else _legal_actions_or_empty(game)
        views, _ = _views_for_game(game, data_mode=data_mode, perspective=perspective)
        delta = compute_state_delta(pre, post).to_dict()
        if data_mode == "perspective-sanitized-data":
            # IDs can identify a private deck reservation even when its CardView is masked.
            delta["reserved_card_ids"] = []
        safe_event = {
            key: event.get(key)
            for key in (
                "decision_id",
                "turn_id",
                "round_id",
                "phase",
                "player",
                "action_type",
                "action_text",
                "action_params",
                "automatic_resolution",
            )
        }
        frames.append(
            {
                "views": views,
                "event": safe_event,
                "delta": delta,
                "turn_completed": result.turn_ended,
                "legal_actions": legal_actions,
            }
        )
    payload = {
        "schema": "splendor-visual-replay-1",
        "dataMode": data_mode,
        "defaultPerspective": str(perspective),
        "perspectives": perspectives,
        "frames": frames,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_document(payload), encoding="utf-8")
    return output


def export_game_view(
    game: SplendorGame,
    output_path: str | Path,
    perspective: Perspective = "current",
) -> Path:
    """Export one current live state as a self-contained HTML viewer."""
    views, perspectives = _views_for_game(
        game,
        data_mode="perspective-sanitized-data" if perspective != "omniscient" else "omniscient-data",
        perspective=perspective,
    )
    payload = {
        "schema": "splendor-live-view-1",
        "dataMode": "snapshot",
        "defaultPerspective": str(perspective),
        "perspectives": perspectives,
        "frames": [{"views": views, "event": None, "delta": {}, "turn_completed": False, "legal_actions": [] if game.done else _legal_actions_or_empty(game)}],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_document(payload), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-mode", choices=("omniscient-data", "perspective-sanitized-data"), default="omniscient-data")
    parser.add_argument("--perspective", default="current")
    parser.add_argument("--single-file", action="store_true", default=True)
    args = parser.parse_args()
    perspective: Perspective = int(args.perspective) if args.perspective.isdigit() else args.perspective
    output = export_replay(args.recording, args.output, data_mode=args.data_mode, perspective=perspective)
    print(f"exported {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
