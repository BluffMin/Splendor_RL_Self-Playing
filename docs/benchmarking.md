# Collector benchmarking

Use a separate benchmark output under the ignored `runs/` directory:

```powershell
python .\experiments\benchmark_collector.py --config .\configs\population_league_2p_50m_fast.yaml --run-dir .\runs\population_league_2p_50m_seed42 --device cuda --workers 1,2,4,8 --envs-per-worker 8 --transitions 4096 --progress always
```

The command loads one checkpoint for all trials and does not save learner,
optimizer, population, or matchup state. Compare longer trials before choosing a
machine-specific worker count; short trials overstate spawn overhead.

For an interactive desktop benchmark, include worker 6 and request BALANCED
selection:

```powershell
python .\experiments\benchmark_collector.py --config .\configs\population_league_2p_50m_balanced.yaml --run-dir .\runs\population_league_2p_50m_seed42 --device cuda --workers 4,6,8 --envs-per-worker 8 --transitions 16384 --preset balanced --write-local-config .\configs\population_league_2p_50m_fast_local.yaml
```

The sampler uses Windows system counters and `nvidia-smi`. BALANCED rejects unsafe
CPU/RAM/VRAM/temperature results. Among configurations within 10% of the best safe
throughput, it selects the lower-resource option. MAXIMUM selects the fastest stable
result.
