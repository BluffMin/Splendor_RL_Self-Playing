"""Browser visualization exports for live and recorded Splendor games."""

from .view_model import (
    BoardView,
    CardView,
    NobleView,
    PlayerView,
    StateDelta,
    board_view_from_game,
    board_view_from_snapshot,
    compute_state_delta,
)


def export_game_view(*args, **kwargs):
    from .html_export import export_game_view as implementation

    return implementation(*args, **kwargs)


def export_replay(*args, **kwargs):
    from .html_export import export_replay as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "BoardView",
    "CardView",
    "NobleView",
    "PlayerView",
    "StateDelta",
    "board_view_from_game",
    "board_view_from_snapshot",
    "compute_state_delta",
    "export_game_view",
    "export_replay",
]
