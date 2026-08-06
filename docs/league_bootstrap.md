# Empirical league bootstrap

v0.5.1 discovers named and sampled numbered checkpoints, removes duplicate actor hashes, evaluates identical seeds with balanced seats, and runs a paired seat-swapped checkpoint tournament.

Champion selection is lexicographic: hard-anchor score, population paired score, worst hard anchor, aggregate score, win-round efficiency for practical ties, then transition count. A checkpoint name never selects the Champion by itself. Historical policies combine strength with matchup-vector diversity.

```powershell
python .\experiments\bootstrap_league.py --config .\configs\league_bootstrap_2p.yaml --source-run-dir .\runs\shared_ppo_2p_1m_seed42 --output-dir .\runs\league_bootstrap_2p_seed42 --device cuda --progress always
```

Use the resulting pool with `train_league_ppo.py --bootstrap-manifest PATH`. This option is mutually exclusive with `--initial-checkpoint` and `--resume`; player count, tensor schema, architecture, and actor SHA-256 are verified.
