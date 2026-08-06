"""Numerical card and noble definitions for the original Splendor base game.

Color order throughout the package is:
    green, white, blue, black, red
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .actions import GEM_COLORS


@dataclass(frozen=True, slots=True)
class Card:
    card_id: str
    tier: int  # 0, 1, 2
    points: int
    bonus_color: str
    cost: tuple[int, int, int, int, int]

    @property
    def bonus(self) -> int:
        """Backward-compatible numeric bonus index."""
        return GEM_COLORS.index(self.bonus_color)

    def cost_by_color(self) -> dict[str, int]:
        return dict(zip(GEM_COLORS, self.cost, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "card_id": self.card_id,
            "tier": self.tier + 1,
            "bonus_color": self.bonus_color,
            "points": self.points,
            "cost": self.cost_by_color(),
        }

    def short_text(self) -> str:
        costs = ", ".join(f"{c}={n}" for c, n in self.cost_by_color().items() if n)
        return f"{self.card_id}: {self.bonus_color} bonus, {self.points} pt, cost {costs or 'free'}"


@dataclass(frozen=True, slots=True)
class Noble:
    noble_id: str
    requirements: tuple[int, int, int, int, int]
    points: int = 3

    @property
    def cost(self) -> tuple[int, int, int, int, int]:
        return self.requirements

    def requirements_by_color(self) -> dict[str, int]:
        return dict(zip(GEM_COLORS, self.requirements, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "noble_id": self.noble_id,
            "points": self.points,
            "requirements": self.requirements_by_color(),
        }


_CARD_CSV = """tier,value,type,green,white,blue,black,red
1,0,1,0,2,1,0,0
1,0,2,0,0,0,1,2
1,0,3,0,1,0,2,0
1,0,4,2,0,0,0,1
1,0,5,1,0,2,0,0
1,0,1,0,0,0,0,3
1,0,2,0,0,3,0,0
1,0,3,0,0,0,3,0
1,0,4,3,0,0,0,0
1,0,5,0,3,0,0,0
1,1,1,0,0,0,4,0
1,1,2,4,0,0,0,0
1,1,3,0,0,0,0,4
1,1,4,0,0,4,0,0
1,1,5,0,4,0,0,0
1,0,1,0,1,1,1,1
1,0,2,1,0,1,1,1
1,0,3,1,1,0,1,1
1,0,4,1,1,1,0,1
1,0,5,1,1,1,1,0
1,0,1,0,0,2,0,2
1,0,2,0,0,2,2,0
1,0,3,2,0,0,2,0
1,0,4,2,2,0,0,0
1,0,5,0,2,0,0,2
1,0,1,1,1,3,0,0
1,0,2,0,3,1,1,0
1,0,3,3,0,1,0,1
1,0,4,1,0,0,1,3
1,0,5,0,1,0,3,1
1,0,1,0,0,1,2,2
1,0,2,2,0,2,1,0
1,0,3,2,1,0,0,2
1,0,4,0,2,2,0,1
1,0,5,1,2,0,2,0
1,0,1,0,1,1,2,1
1,0,2,2,0,1,1,1
1,0,3,1,1,0,1,2
1,0,4,1,1,2,0,1
1,0,5,1,2,1,1,0
2,1,1,0,2,3,2,0
2,1,2,3,0,0,2,2
2,1,3,2,0,2,0,3
2,1,4,2,3,2,0,0
2,1,5,0,2,0,3,2
2,1,1,2,3,0,0,3
2,1,2,0,2,3,0,3
2,1,3,3,0,2,3,0
2,1,4,3,3,0,2,0
2,1,5,0,0,3,3,2
2,2,1,5,0,0,0,0
2,2,2,0,0,0,0,5
2,2,3,0,0,5,0,0
2,2,4,0,5,0,0,0
2,2,5,0,0,0,5,0
2,2,1,0,4,2,1,0
2,2,2,1,0,0,2,4
2,2,3,0,2,0,4,1
2,2,4,4,0,1,0,2
2,2,5,2,1,4,0,0
2,2,1,3,0,5,0,0
2,2,2,0,0,0,3,5
2,2,3,0,5,3,0,0
2,2,4,5,0,0,0,3
2,2,5,0,3,0,5,0
2,3,1,6,0,0,0,0
2,3,2,0,6,0,0,0
2,3,3,0,0,6,0,0
2,3,4,0,0,0,6,0
2,3,5,0,0,0,0,6
3,3,1,0,5,3,3,3
3,3,2,3,0,3,3,5
3,3,3,3,3,0,5,3
3,3,4,5,3,3,0,3
3,3,5,3,3,5,3,0
3,4,1,0,0,7,0,0
3,4,2,0,0,0,7,0
3,4,3,0,7,0,0,0
3,4,4,0,0,0,0,7
3,4,5,7,0,0,0,0
3,4,1,3,3,6,0,0
3,4,2,0,3,0,6,3
3,4,3,0,6,3,3,0
3,4,4,3,0,0,3,6
3,4,5,6,0,3,0,3
3,5,1,3,0,7,0,0
3,5,2,0,3,0,7,0
3,5,3,0,7,3,0,0
3,5,4,0,0,0,3,7
3,5,5,7,0,0,0,3
"""

_NOBLE_CSV = """green,white,blue,black,red
4,0,4,0,0
4,0,0,0,4
0,0,0,4,4
0,4,0,4,0
0,4,4,0,0
3,3,3,0,0
3,0,3,0,3
3,0,0,3,3
0,3,0,3,3
0,3,3,3,0
"""


def load_cards() -> tuple[Card, ...]:
    cards: list[Card] = []
    counters: dict[tuple[int, str], int] = {}
    type_colors = ("green", "white", "blue", "black", "red")
    for row in csv.DictReader(io.StringIO(_CARD_CSV)):
        tier = int(row["tier"])
        bonus_color = type_colors[int(row["type"]) - 1]
        key = (tier, bonus_color)
        counters[key] = counters.get(key, 0) + 1
        cards.append(
            Card(
                card_id=f"T{tier}-{bonus_color.upper()}-{counters[key]:02d}",
                tier=tier - 1,
                points=int(row["value"]),
                bonus_color=bonus_color,
                cost=tuple(int(row[color]) for color in GEM_COLORS),
            )
        )
    assert len(cards) == 90
    assert [sum(card.tier == tier for card in cards) for tier in range(3)] == [40, 30, 20]
    assert len({card.card_id for card in cards}) == 90
    for tier, expected in enumerate((8, 6, 4)):
        assert all(
            sum(card.tier == tier and card.bonus_color == color for card in cards) == expected
            for color in GEM_COLORS
        )
    return tuple(cards)


def load_nobles() -> tuple[Noble, ...]:
    nobles: list[Noble] = []
    for noble_index, row in enumerate(csv.DictReader(io.StringIO(_NOBLE_CSV)), 1):
        nobles.append(
            Noble(
                noble_id=f"N-{noble_index:02d}",
                requirements=tuple(int(row[color]) for color in GEM_COLORS),
            )
        )
    assert len(nobles) == 10
    assert len({noble.noble_id for noble in nobles}) == 10
    assert len({noble.requirements for noble in nobles}) == 10
    assert all(noble.points == 3 for noble in nobles)
    return tuple(nobles)


CARDS = load_cards()
NOBLES = load_nobles()
