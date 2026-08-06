from __future__ import annotations

import json
from pathlib import Path


def atomic_json_write(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def load_league_state(path):
    target = Path(path)
    if not target.exists():
        return None
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("schema_version") != "0.5.0":
        raise ValueError("unsupported league state schema")
    return value
