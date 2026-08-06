"""Numerical card and noble definitions for the original Splendor base game.

Color order throughout the package is:
    green, white, blue, black, red
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .actions import COLORS


@dataclass(frozen=True, slots=True)
class Card:
    card_id: int
    tier: int  # 0, 1, 2
    points: int
    bonus: int  # color index 0..4
    cost: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class Noble:
    noble_id: int
    cost: tuple[int, int, int, int, int]
    points: int = 3


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
    for card_id, row in enumerate(csv.DictReader(io.StringIO(_CARD_CSV))):
        cards.append(
            Card(
                card_id=card_id,
                tier=int(row["tier"]) - 1,
                points=int(row["value"]),
                bonus=int(row["type"]) - 1,
                cost=tuple(int(row[color]) for color in COLORS),
            )
        )
    assert len(cards) == 90
    assert [sum(card.tier == tier for card in cards) for tier in range(3)] == [40, 30, 20]
    return tuple(cards)


def load_nobles() -> tuple[Noble, ...]:
    nobles: list[Noble] = []
    for noble_id, row in enumerate(csv.DictReader(io.StringIO(_NOBLE_CSV))):
        nobles.append(
            Noble(
                noble_id=noble_id,
                cost=tuple(int(row[color]) for color in COLORS),
            )
        )
    assert len(nobles) == 10
    return tuple(nobles)


CARDS = load_cards()
NOBLES = load_nobles()
