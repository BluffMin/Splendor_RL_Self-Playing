from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor_rl.distributions import MaskedCategorical
from splendor_rl.models import SharedActor

from .types import OpponentMetadata


def actor_sha256(actor_or_state) -> str:
    state = (
        actor_or_state.state_dict()
        if hasattr(actor_or_state, "state_dict")
        else actor_or_state
    )
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class FrozenOpponent:
    def __init__(self, actor: SharedActor, metadata: OpponentMetadata, device):
        self.actor = actor.to(device)
        self.metadata = metadata
        self.device = torch.device(device)
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def action_probabilities(self, observation, action_mask):
        obs = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
        return (
            MaskedCategorical(self.actor(obs), mask)
            .probs.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def act(self, observation, action_mask, *, deterministic, generator=None):
        obs = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
        distribution = MaskedCategorical(self.actor(obs), mask)
        if deterministic:
            action = distribution.mode()
        elif isinstance(generator, np.random.Generator):
            probabilities = distribution.probs.squeeze(0).detach().cpu().numpy()
            action = torch.tensor(
                int(generator.choice(len(probabilities), p=probabilities)),
                device=self.device,
            )
        else:
            action = torch.multinomial(
                distribution.probs.squeeze(0), 1, generator=generator
            ).squeeze(0)
        result = int(action.item())
        if not bool(action_mask[result]):
            raise AssertionError("frozen opponent selected an illegal action")
        return result


class OpponentPool:
    def __init__(self, root, hidden_sizes, device="cpu"):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.hidden_sizes = list(hidden_sizes)
        self.device = torch.device(device)
        self.metadata: dict[str, OpponentMetadata] = {}
        self.loaded: dict[str, FrozenOpponent] = {}
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.metadata = {
                item["opponent_id"]: OpponentMetadata(**item)
                for item in raw.get("opponents", [])
            }

    def _save_index(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "0.5.0",
                    "opponents": [asdict(item) for item in self.metadata.values()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.index_path)

    def add_snapshot(
        self,
        actor,
        *,
        opponent_id,
        source_type,
        created_transition,
        champion_version,
        training_seed,
        actor_obs_size,
        action_size,
        num_players=2,
        bootstrap_champion=False,
        source_checkpoint="",
    ):
        if opponent_id in self.metadata:
            raise ValueError(f"duplicate opponent ID: {opponent_id}")
        folder = "hall_of_fame" if source_type == "champion" else "recent"
        relative = Path(folder) / f"{opponent_id}_actor.pt"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        checksum = actor_sha256(actor)
        payload = {
            "schema_version": "0.5.0",
            "actor_state_dict": actor.state_dict(),
            "actor_config": {"hidden_sizes": self.hidden_sizes},
            "actor_obs_size": actor_obs_size,
            "action_size": action_size,
            "num_players": num_players,
            "source_transition": created_transition,
            "source_checkpoint": source_checkpoint,
            "opponent_id": opponent_id,
            "sha256": checksum,
        }
        torch.save(payload, path)
        metadata = OpponentMetadata(
            opponent_id,
            source_type,
            created_transition,
            champion_version,
            training_seed,
            actor_obs_size,
            action_size,
            num_players,
            relative.as_posix(),
            checksum,
            bootstrap_champion,
        )
        self.metadata[opponent_id] = metadata
        self._save_index()
        return metadata

    def remove(self, opponent_id):
        metadata = self.metadata[opponent_id]
        if metadata.source_type == "champion":
            raise ValueError("Hall-of-Fame champions cannot be removed")
        path = self.root / metadata.file_name
        if path.exists():
            path.unlink()
        self.metadata.pop(opponent_id)
        self.loaded.pop(opponent_id, None)
        self._save_index()

    def trim_recent(self, maximum):
        recent = sorted(
            (item for item in self.metadata.values() if item.source_type == "recent"),
            key=lambda item: item.created_transition,
        )
        for item in recent[:-maximum] if maximum else recent:
            self.remove(item.opponent_id)

    def load(self, opponent_id):
        if opponent_id in self.loaded:
            return self.loaded[opponent_id]
        if opponent_id not in self.metadata:
            raise KeyError(f"unknown opponent: {opponent_id}")
        metadata = self.metadata[opponent_id]
        path = self.root / metadata.file_name
        if not path.exists():
            raise FileNotFoundError(f"opponent snapshot is missing: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema_version") != "0.5.0":
            raise ValueError("unsupported opponent snapshot schema")
        checksum = actor_sha256(payload["actor_state_dict"])
        if checksum != metadata.sha256 or checksum != payload.get("sha256"):
            raise ValueError(f"opponent snapshot SHA-256 mismatch: {opponent_id}")
        actor = SharedActor(
            metadata.actor_obs_size, metadata.action_size, self.hidden_sizes
        )
        actor.load_state_dict(payload["actor_state_dict"])
        result = FrozenOpponent(actor, metadata, self.device)
        self.loaded[opponent_id] = result
        return result

    @property
    def hall_of_fame_ids(self):
        return [
            key for key, item in self.metadata.items() if item.source_type == "champion"
        ]

    @property
    def recent_ids(self):
        return [
            key for key, item in self.metadata.items() if item.source_type == "recent"
        ]

    def historical_ids(self, current_champion_id=None):
        return [
            key
            for key in [*self.hall_of_fame_ids, *self.recent_ids]
            if key != current_champion_id
        ]
