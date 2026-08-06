from __future__ import annotations

import json
from pathlib import Path


class JsonlMetrics:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, values):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False) + "\n")
