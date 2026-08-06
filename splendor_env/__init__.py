"""Self-play-ready Splendor reinforcement-learning environment."""

from .actions import (
    ACTIONS,
    COLORS,
    GEM_COLORS,
    N_ACTIONS,
    TOKEN_COLORS,
    describe_action,
)
from .core import (
    OBSERVATION_SIZE,
    DiscardPlan,
    InvalidActionError,
    NoLegalActionError,
    PaymentPlan,
    Phase,
    SplendorGame,
    enumerate_legal_discard_plans,
    enumerate_legal_payment_plans,
)

__version__ = "0.4.1"

__all__ = [
    "ACTIONS",
    "COLORS",
    "GEM_COLORS",
    "N_ACTIONS",
    "OBSERVATION_SIZE",
    "TOKEN_COLORS",
    "DiscardPlan",
    "InvalidActionError",
    "NoLegalActionError",
    "PaymentPlan",
    "Phase",
    "SplendorGame",
    "describe_action",
    "enumerate_legal_discard_plans",
    "enumerate_legal_payment_plans",
]


def env(**kwargs):
    """Create the wrapped PettingZoo AEC environment lazily."""
    from .pettingzoo_env import env as make_env

    return make_env(**kwargs)


def raw_env(**kwargs):
    """Create the unwrapped PettingZoo AEC environment lazily."""
    from .pettingzoo_env import raw_env as RawEnv

    return RawEnv(**kwargs)
