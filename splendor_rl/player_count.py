from __future__ import annotations

SUPPORTED_PLAYER_COUNTS = (2, 3, 4)


def validate_num_players(num_players: int) -> int:
    if num_players not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(
            f"num_players must be one of {SUPPORTED_PLAYER_COUNTS}, got {num_players}"
        )
    return num_players


def balanced_policy_seats(num_players: int, num_games: int) -> tuple[int, ...]:
    validate_num_players(num_players)
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    return tuple(index % num_players for index in range(num_games))


def build_fixed_bot_matchups(num_players: int) -> dict[str, tuple[str, ...]]:
    validate_num_players(num_players)
    matchups = {
        f"policy_vs_{name}": tuple([name] * (num_players - 1))
        for name in ("random", "greedy", "shortest", "noble", "blocking")
    }
    if num_players >= 3:
        matchups["mixed_ladder"] = ("greedy", "shortest", "blocking")[: num_players - 1]
    if any(len(opponents) != num_players - 1 for opponents in matchups.values()):
        raise AssertionError("invalid fixed-bot matchup")
    return matchups
