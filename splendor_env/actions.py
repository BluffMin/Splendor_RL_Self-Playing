"""Fixed discrete action vocabulary for Splendor.

The action space is deliberately flat so that DQN/PPO-style policies can use a
single categorical head together with an invalid-action mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

COLORS = ("green", "white", "blue", "black", "red")
TOKEN_COLORS = COLORS + ("gold",)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    kind: str
    payload: Any


ACTIONS: list[ActionSpec] = []

# Take 1-3 distinct non-gold colors. Only the largest legal subset size is
# enabled by the mask (normally 3; 1 or 2 only when fewer stacks are nonempty).
for size in (1, 2, 3):
    for combo in combinations(range(5), size):
        ACTIONS.append(ActionSpec("take_distinct", combo))

# Take two of one color.
for color in range(5):
    ACTIONS.append(ActionSpec("take_double", color))

# Buy one of 12 visible cards (tier-major, four slots per tier).
for slot in range(12):
    ACTIONS.append(ActionSpec("buy_visible", slot))

# Reserve one of 12 visible cards.
for slot in range(12):
    ACTIONS.append(ActionSpec("reserve_visible", slot))

# Blind-reserve from the top of a tier deck.
for tier in range(3):
    ACTIONS.append(ActionSpec("reserve_deck", tier))

# Buy one of the current player's three reserved slots.
for slot in range(3):
    ACTIONS.append(ActionSpec("buy_reserved", slot))

# During the forced discard phase, return one token at a time.
for color in range(6):
    ACTIONS.append(ActionSpec("discard_one", color))

# If multiple nobles are eligible, choose one of up to five visible nobles.
for slot in range(5):
    ACTIONS.append(ActionSpec("choose_noble", slot))

# Safeguard for pathological deadlocks. This is enabled only when no official
# action is legal. A full table rotation of passes truncates the episode.
ACTIONS.append(ActionSpec("pass", None))

# During a purchase, choose how many gold tokens replace each required color.
# A player can hold at most five gold tokens, so allocations with total <= 5
# cover every legal payment without making the action vocabulary unbounded.
PAYMENT_OFFSET = len(ACTIONS)
PAYMENT_ALLOCATIONS = tuple(
    allocation
    for allocation in product(range(6), repeat=5)
    if sum(allocation) <= 5
)
for allocation in PAYMENT_ALLOCATIONS:
    ACTIONS.append(ActionSpec("choose_payment", allocation))

N_ACTIONS = len(ACTIONS)
assert N_ACTIONS == 324

OFFSETS = {
    "take_distinct": 0,
    "take_double": 25,
    "buy_visible": 30,
    "reserve_visible": 42,
    "reserve_deck": 54,
    "buy_reserved": 57,
    "discard_one": 60,
    "choose_noble": 66,
    "pass": 71,
    "choose_payment": PAYMENT_OFFSET,
}


def action_id(kind: str, payload: Any) -> int:
    """Return the integer id for a known action specification."""
    target = ActionSpec(kind, payload)
    try:
        return ACTIONS.index(target)
    except ValueError as exc:
        raise KeyError(f"Unknown action: {target}") from exc


def describe_action(action: int) -> str:
    """Human-readable action description useful for debugging rollouts."""
    if not 0 <= int(action) < N_ACTIONS:
        return f"invalid_action({action})"
    spec = ACTIONS[int(action)]
    if spec.kind == "take_distinct":
        names = ", ".join(COLORS[i] for i in spec.payload)
        return f"take distinct: {names}"
    if spec.kind == "take_double":
        return f"take two: {COLORS[spec.payload]}"
    if spec.kind in {"buy_visible", "reserve_visible"}:
        tier, slot = divmod(spec.payload, 4)
        verb = "buy" if spec.kind == "buy_visible" else "reserve"
        return f"{verb} visible: tier {tier + 1}, slot {slot}"
    if spec.kind == "reserve_deck":
        return f"blind reserve: tier {spec.payload + 1}"
    if spec.kind == "buy_reserved":
        return f"buy reserved: slot {spec.payload}"
    if spec.kind == "discard_one":
        return f"discard one: {TOKEN_COLORS[spec.payload]}"
    if spec.kind == "choose_noble":
        return f"choose noble: slot {spec.payload}"
    if spec.kind == "pass":
        return "pass (deadlock safeguard)"
    if spec.kind == "choose_payment":
        parts = ", ".join(
            f"{COLORS[i]}={amount}"
            for i, amount in enumerate(spec.payload)
            if amount
        )
        return f"pay with gold: {parts or 'none'}"
    return repr(spec)
