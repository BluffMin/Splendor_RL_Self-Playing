"""Self-play-ready Splendor reinforcement-learning environment."""

from .actions import ACTIONS, COLORS, N_ACTIONS, TOKEN_COLORS, describe_action
from .core import OBSERVATION_SIZE, InvalidActionError, SplendorGame

__all__ = [
    "ACTIONS",
    "COLORS",
    "N_ACTIONS",
    "OBSERVATION_SIZE",
    "TOKEN_COLORS",
    "InvalidActionError",
    "SplendorGame",
    "describe_action",
]


def env(**kwargs):
    """Create the wrapped PettingZoo AEC environment lazily."""
    from .pettingzoo_env import env as make_env

    return make_env(**kwargs)


def raw_env(**kwargs):
    """Create the unwrapped PettingZoo AEC environment lazily."""
    from .pettingzoo_env import raw_env as RawEnv

    return RawEnv(**kwargs)
