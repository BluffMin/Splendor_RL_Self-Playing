# Population league

v0.6.0 trains five independent PPO roles sequentially: Main, two Main Exploiters, and two League Exploiters. Every role owns its actor, critic, optimizer, learning-rate/entropy schedule, counters, and deterministic update stream. Frozen snapshots are used as opponents; opponent actions never enter the learner PPO batch.

The default scheduler allocates 60% of population transitions to Main and 10% to each exploiter. `total_population_transitions` counts the sum of transitions actually used by all five PPO learners.

The v0.5.1 source supplies `champion_0006`, Hall of Fame, recent snapshots, actor weights, and a compatible critic. v0.6 optimizer state and schedules start from zero; source lifetime transitions remain separate metadata.

```powershell
python .\experiments\train_population_league.py --config .\configs\population_league_2p_50m.yaml --run-dir .\runs\population_league_2p_50m_seed42 --bootstrap-run-dir .\runs\league_ppo_2p_20m_seed42 --device cuda --progress always
```

To stop without changing the 50M schedule horizon, add `--stop-at-population-transitions 5000000`. Continue with `--resume .\runs\population_league_2p_50m_seed42\population_state.json` instead of `--bootstrap-run-dir`.
