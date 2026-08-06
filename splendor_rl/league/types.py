from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class OpponentMetadata:
    opponent_id: str
    source_type: str
    created_transition: int
    champion_version: int | None
    training_seed: int
    actor_obs_size: int
    action_size: int
    num_players: int
    file_name: str
    sha256: str
    bootstrap_champion: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LeagueEpisodeAssignment:
    env_id: int
    episode_id: int
    mode: str
    candidate_seat: int | None
    opponent_id: str | None
    opponent_source_type: str | None
    seed: int
    pfsp_probability: float | None = None


@dataclass
class MatchRecord:
    opponent_id: str
    games: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    candidate_as_p0_games: int = 0
    candidate_as_p1_games: int = 0
    score_sum: float = 0.0
    recent_results: list[float] = field(default_factory=list)
    last_played_transition: int = 0

    @property
    def average_score(self) -> float:
        return self.score_sum / self.games if self.games else 0.5

    @property
    def recent_games(self) -> int:
        return len(self.recent_results)

    @property
    def recent_score(self) -> float:
        return (
            sum(self.recent_results) / len(self.recent_results)
            if self.recent_results
            else 0.5
        )

    def add(
        self,
        score: float,
        candidate_seat: int,
        transition: int,
        window: int,
        final_score: float | None = None,
    ):
        self.games += 1
        self.wins += score == 1.0
        self.ties += score == 0.5
        self.losses += score == 0.0
        self.candidate_as_p0_games += candidate_seat == 0
        self.candidate_as_p1_games += candidate_seat == 1
        self.score_sum += score if final_score is None else final_score
        self.recent_results.append(score)
        del self.recent_results[:-window]
        self.last_played_transition = transition

    def to_dict(self):
        value = asdict(self)
        value.update(
            average_score=self.average_score,
            recent_games=self.recent_games,
            recent_score=self.recent_score,
        )
        return value

    @classmethod
    def from_dict(cls, value):
        fields = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        return cls(**fields)
