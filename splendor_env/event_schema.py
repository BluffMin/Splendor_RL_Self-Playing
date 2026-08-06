"""Versioned decision and player-turn records for replay tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "0.3.2"


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    schema_version: str
    event_sequence_id: int | None
    decision_id: int | None
    player_turn_id: int
    round_id: int
    subdecision_index: int
    acting_player: int
    phase_before: str
    phase_after: str
    action_id: int | None
    action_type: str
    action_params: dict[str, Any]
    action_text: str
    automatic: bool
    turn_started: bool
    turn_completed: bool
    next_player: int | None
    pre_state_hash: str
    post_state_hash: str
    pre_snapshot: dict[str, Any] | None = None
    post_snapshot: dict[str, Any] | None = None
    automatic_resolution: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["automatic_resolution"] = list(self.automatic_resolution)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionEvent:
        return cls(
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
            event_sequence_id=value.get("event_sequence_id", value.get("decision_id")),
            decision_id=value.get("decision_id"),
            player_turn_id=int(value.get("player_turn_id", value.get("turn_id", 0))),
            round_id=int(value.get("round_id", 0)),
            subdecision_index=int(value.get("subdecision_index", 0)),
            acting_player=int(value.get("acting_player", value.get("player", 0))),
            phase_before=str(value.get("phase_before", value.get("phase", "normal"))),
            phase_after=str(value.get("phase_after", value.get("phase", "normal"))),
            action_id=value.get("action_id"),
            action_type=str(value.get("action_type", "unknown")),
            action_params=dict(value.get("action_params") or {}),
            action_text=str(value.get("action_text", "")),
            automatic=bool(value.get("automatic", False)),
            turn_started=bool(value.get("turn_started", False)),
            turn_completed=bool(value.get("turn_completed", False)),
            next_player=value.get("next_player"),
            pre_state_hash=str(value.get("pre_state_hash", "")),
            post_state_hash=str(value.get("post_state_hash", "")),
            pre_snapshot=value.get("pre_snapshot"),
            post_snapshot=value.get("post_snapshot"),
            automatic_resolution=tuple(value.get("automatic_resolution") or ()),
        )


@dataclass(frozen=True, slots=True)
class TurnRecord:
    schema_version: str
    player_turn_id: int
    round_id: int
    acting_player: int
    next_player: int | None
    start_decision_id: int
    end_decision_id: int
    decision_ids: tuple[int, ...]
    primary_action_type: str
    primary_action_text: str
    payment_summary: dict[str, Any] | None
    discard_summary: dict[str, int] | None
    purchased_card: dict[str, Any] | None
    reserved_card: dict[str, Any] | None
    taken_tokens: dict[str, int] | None
    acquired_noble: dict[str, Any] | None
    score_before: int
    score_after: int
    pre_state_hash: str
    post_state_hash: str
    pre_snapshot: dict[str, Any] | None
    post_snapshot: dict[str, Any] | None
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision_ids"] = list(self.decision_ids)
        return value


def _player(snapshot: dict[str, Any] | None, actor: int) -> dict[str, Any]:
    if not snapshot or actor >= len(snapshot.get("players", [])):
        return {}
    return snapshot["players"][actor]


def _added(before: list[Any], after: list[Any]) -> Any | None:
    before_ids = {x.get("card_id", x) if isinstance(x, dict) else x for x in before}
    return next(
        (
            x
            for x in after
            if (x.get("card_id", x) if isinstance(x, dict) else x) not in before_ids
        ),
        None,
    )


def build_turn_records(
    events: list[DecisionEvent | dict[str, Any]],
) -> list[TurnRecord]:
    """Group ordered decision events without inventing missing decisions."""
    normalized = [
        e if isinstance(e, DecisionEvent) else DecisionEvent.from_dict(e)
        for e in events
    ]
    groups: list[list[DecisionEvent]] = []
    for event in normalized:
        if not groups or groups[-1][0].player_turn_id != event.player_turn_id:
            groups.append([event])
        else:
            groups[-1].append(event)
    records: list[TurnRecord] = []
    for group in groups:
        first, last = group[0], group[-1]
        primary = next(
            (e for e in group if e.phase_before == "normal" and not e.automatic),
            group[0],
        )
        pre, post = first.pre_snapshot, last.post_snapshot
        pb, pa = _player(pre, first.acting_player), _player(post, first.acting_player)
        purchased = _added(pb.get("purchased", []), pa.get("purchased", []))
        reserved = _added(pb.get("reserved", []), pa.get("reserved", []))
        noble = _added(pb.get("nobles", []), pa.get("nobles", []))
        payment_event = next((e for e in group if e.phase_before == "payment"), None)
        discard_event = next((e for e in group if e.phase_before == "discard"), None)
        taken = None
        if primary.action_type == "take_distinct":
            taken = {c: 1 for c in primary.action_params.get("colors", [])}
        elif primary.action_type == "take_double":
            taken = {str(primary.action_params.get("color")): 2}
        ids = tuple(int(e.decision_id) for e in group if e.decision_id is not None)
        records.append(
            TurnRecord(
                SCHEMA_VERSION,
                first.player_turn_id,
                first.round_id,
                first.acting_player,
                last.next_player,
                ids[0] if ids else -1,
                ids[-1] if ids else -1,
                ids,
                primary.action_type,
                primary.action_text,
                payment_event.action_params if payment_event else None,
                discard_event.action_params if discard_event else None,
                {"card_id": purchased} if isinstance(purchased, str) else purchased,
                {"card_id": reserved} if isinstance(reserved, str) else reserved,
                taken,
                {"noble_id": noble} if isinstance(noble, str) else noble,
                int(pb.get("score", 0)),
                int(pa.get("score", pb.get("score", 0))),
                first.pre_state_hash,
                last.post_state_hash,
                pre,
                post,
                bool(last.turn_completed),
            )
        )
    return records


def summarize_turn(turn: TurnRecord) -> str:
    card = turn.purchased_card and turn.purchased_card.get("card_id")
    reserved = turn.reserved_card and turn.reserved_card.get("card_id")
    if card:
        action = f"P{turn.acting_player} purchased {card}"
    elif reserved:
        action = f"P{turn.acting_player} reserved {reserved}"
    elif turn.taken_tokens:
        action = f"P{turn.acting_player} took " + ", ".join(
            f"{n} {c}" for c, n in turn.taken_tokens.items()
        )
    elif turn.primary_action_type == "inferred_purchase":
        action = (
            f"P{turn.acting_player} purchase inferred (primary decision unavailable)"
        )
    else:
        action = f"P{turn.acting_player} {turn.primary_action_text}"
    lines = [
        f"Turn {turn.player_turn_id + 1} · Round {turn.round_id + 1} · P{turn.acting_player}",
        f"Action: {action}",
    ]
    if turn.payment_summary:
        lines.append(f"Payment: {turn.payment_summary}")
    if turn.discard_summary:
        lines.append(f"Discard: {turn.discard_summary}")
    if turn.acquired_noble:
        lines.append(f"Noble: {turn.acquired_noble}")
    lines.append(f"Score: {turn.score_before} -> {turn.score_after}")
    lines.append(
        "Game ended"
        if turn.next_player is None and turn.completed
        else f"Next player: {('P' + str(turn.next_player)) if turn.next_player is not None else 'not decided'}"
    )
    return "\n".join(lines)
