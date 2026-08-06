# Legacy log migration

Migration never modifies source files. It writes `.v032.json` files with the source SHA-256 and a CSV/Markdown report.

`exact` means action IDs and state hashes allowed deterministic re-execution and exact reconstruction. `best_effort` means information was missing; warnings identify the inference, and `replay_verifiable` is false. Legacy text is intended for human visualization only and must not be included in quantitative turn statistics by default.

Dry run:

```shell
python -m splendor_env.migrations.migrate_logs_v032 runs/old_logs --output-dir runs/migrated_v032 --dry-run
```

Add `--recursive --verify` for migration and deterministic verification. Add `--include-text-replays` only when best-effort text conversion is desired.
