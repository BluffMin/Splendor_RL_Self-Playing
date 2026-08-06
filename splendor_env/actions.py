"""Stable fixed action layout for every Splendor decision phase."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

GEM_COLORS = ("white", "blue", "green", "red", "black")
COLORS = GEM_COLORS  # Backward-compatible alias.
TOKEN_COLORS = GEM_COLORS + ("gold",)

NORMAL_OFFSET = 0
NORMAL_SIZE = 60
PAYMENT_OFFSET = 60
PAYMENT_SIZE = 252  # C(5 gold + 5 colors, 5 colors)
DISCARD_OFFSET = PAYMENT_OFFSET + PAYMENT_SIZE
DISCARD_SIZE = 56  # C(3 returned + 6 token types - 1, 6 - 1)
NOBLE_OFFSET = DISCARD_OFFSET + DISCARD_SIZE
NOBLE_SIZE = 5
N_ACTIONS = NOBLE_OFFSET + NOBLE_SIZE
assert N_ACTIONS == 373


@dataclass(frozen=True, slots=True)
class ActionSpec:
    kind: str
    payload: Any


NORMAL_ACTIONS: list[ActionSpec] = []
for size in (1, 2, 3):
    for combo in combinations(range(5), size):
        NORMAL_ACTIONS.append(ActionSpec("take_distinct", combo))
for color in range(5):
    NORMAL_ACTIONS.append(ActionSpec("take_double", color))
for slot in range(12):
    NORMAL_ACTIONS.append(ActionSpec("buy_visible", slot))
for slot in range(12):
    NORMAL_ACTIONS.append(ActionSpec("reserve_visible", slot))
for tier in range(3):
    NORMAL_ACTIONS.append(ActionSpec("reserve_deck", tier))
for slot in range(3):
    NORMAL_ACTIONS.append(ActionSpec("buy_reserved", slot))
assert len(NORMAL_ACTIONS) == NORMAL_SIZE

ACTIONS = (
    NORMAL_ACTIONS
    + [ActionSpec("choose_payment", i) for i in range(PAYMENT_SIZE)]
    + [ActionSpec("choose_discard", i) for i in range(DISCARD_SIZE)]
    + [ActionSpec("choose_noble", i) for i in range(NOBLE_SIZE)]
)

OFFSETS = {
    "take_distinct": 0,
    "take_double": 25,
    "buy_visible": 30,
    "reserve_visible": 42,
    "reserve_deck": 54,
    "buy_reserved": 57,
    "choose_payment": PAYMENT_OFFSET,
    "choose_discard": DISCARD_OFFSET,
    "choose_noble": NOBLE_OFFSET,
}


def action_id(kind: str, payload: Any) -> int:
    """Return a stable action ID; phase plans use their deterministic index."""
    if kind in {"choose_payment", "choose_discard", "choose_noble"}:
        action = OFFSETS[kind] + int(payload)
        if not 0 <= action < N_ACTIONS:
            raise KeyError(f"Unknown action: {kind} {payload}")
        return action
    target = ActionSpec(kind, payload)
    try:
        return NORMAL_ACTIONS.index(target)
    except ValueError as exc:
        raise KeyError(f"Unknown action: {target}") from exc


def describe_action(action: int) -> str:
    """Return a deterministic human-readable action description."""
    if not 0 <= int(action) < N_ACTIONS:
        return f"invalid_action({action})"
    spec = ACTIONS[int(action)]
    if spec.kind == "take_distinct":
        return "take distinct: " + ", ".join(GEM_COLORS[i] for i in spec.payload)
    if spec.kind == "take_double":
        return f"take two: {GEM_COLORS[spec.payload]}"
    if spec.kind in {"buy_visible", "reserve_visible"}:
        tier, slot = divmod(spec.payload, 4)
        verb = "buy" if spec.kind == "buy_visible" else "reserve"
        return f"{verb} visible: tier {tier + 1}, slot {slot}"
    if spec.kind == "reserve_deck":
        return f"blind reserve: tier {spec.payload + 1}"
    if spec.kind == "buy_reserved":
        return f"buy reserved: slot {spec.payload}"
    if spec.kind == "choose_payment":
        return f"choose payment plan {spec.payload}"
    if spec.kind == "choose_discard":
        return f"choose discard plan {spec.payload}"
    return f"choose noble: slot {spec.payload}"
