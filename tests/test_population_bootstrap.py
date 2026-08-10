import json

import pytest

from splendor_rl.population.bootstrap import bootstrap_population
from splendor_rl.population.config import PopulationConfig


def test_bootstrap_rejects_incompatible_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "league_state.json").write_text(
        json.dumps({"schema_version": "0.5.1", "num_players": 4})
    )
    with pytest.raises(ValueError, match="two-player"):
        bootstrap_population(
            source, tmp_path / "pool", PopulationConfig(hidden_sizes=[8]), "cpu"
        )
