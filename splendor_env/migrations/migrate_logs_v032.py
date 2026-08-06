"""Migrate older JSON/text replay logs to normalized v0.3.2 documents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..recording import load_episode_log
from ..replay import verify_recording


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(
    source: str | Path,
    output_dir: str | Path,
    *,
    recursive=False,
    dry_run=False,
    verify=False,
    include_text_replays=False,
    strict=False,
    report: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(source)
    output = Path(output_dir)
    candidates = (
        [root]
        if root.is_file()
        else list(root.rglob("*") if recursive else root.glob("*"))
    )
    candidates = [
        p
        for p in candidates
        if p.is_file()
        and (
            p.suffix.lower() == ".json"
            or include_text_replays
            and p.suffix.lower() == ".txt"
        )
    ]
    rows = []
    for path in candidates:
        try:
            log = load_episode_log(path)
            source_hash = _sha(path)
            doc = dict(log.document)
            doc["migration"] = {
                "source_path": path.name,
                "source_sha256": source_hash,
                "source_schema_version": log.source_schema_version,
                "migration_tool_version": "0.3.2",
                "migration_quality": log.migration_quality,
                "warnings": list(log.migration_warnings),
            }
            verified = False
            if verify and log.replay_verifiable:
                verify_recording(doc)
                verified = True
            target = output / f"{path.stem}.v032.json"
            if not dry_run:
                output.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            rows.append(
                {
                    "source": path.name,
                    "target": target.name,
                    "source_sha256": source_hash,
                    "source_schema": log.source_schema_version,
                    "quality": log.migration_quality,
                    "replay_verifiable": log.replay_verifiable,
                    "verified": verified,
                    "warnings": " | ".join(log.migration_warnings),
                    "status": "dry-run" if dry_run else "written",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": path.name,
                    "target": "",
                    "source_sha256": _sha(path),
                    "source_schema": "unknown",
                    "quality": "failed",
                    "replay_verifiable": False,
                    "verified": False,
                    "warnings": str(exc),
                    "status": "failed",
                }
            )
            if strict:
                raise
    if not dry_run:
        report_path = Path(report) if report else output / "migration_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0]) if rows else ["source", "status"]
        with report_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        md = report_path.with_suffix(".md")
        md.write_text(
            "# Migration report\n\n"
            + "\n".join(
                f"- {r['source']}: {r['quality']} ({r['status']})" for r in rows
            ),
            encoding="utf-8",
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--include-text-replays", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--report")
    a = p.parse_args(argv)
    rows = migrate(
        a.source,
        a.output_dir,
        recursive=a.recursive,
        dry_run=a.dry_run,
        verify=a.verify,
        include_text_replays=a.include_text_replays,
        strict=a.strict,
        report=a.report,
    )
    for row in rows:
        print(f"{row['source']}: {row['quality']} ({row['status']})")
    return 1 if any(r["status"] == "failed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
