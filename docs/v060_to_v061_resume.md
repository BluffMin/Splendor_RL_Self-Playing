# Resume v0.6.0 with v0.6.1

Run `--resume-dry-run` first. It loads every learner network, critic, optimizer,
counter, scheduler state, population scheduler, Champion, pool, records, and meta
strategy without training or writing the run.

The first real resume writes `pre_v061_backup/migration_manifest.json` containing
the source state SHA-256. Tensor files are not duplicated. Saving the next
authoritative boundary upgrades learner and population schemas to v0.6.1.

Performance-only settings may change on resume. PPO/GAE/reward/network settings and
the 50M total horizon are preserved. New evaluation thresholds start at the next
regular boundary; missed historical gates are not replayed.

State continuity is preserved. Exact stochastic rollout ordering is not guaranteed
with the new backend; use `--collector-backend single_process` for legacy ordering.
