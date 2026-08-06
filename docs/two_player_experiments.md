# Two-player experiments

Recommended progression: 2P smoke → 114,688-transition probe → 300k → 1M → seeds 43 and 44 → final fixed-bot evaluation → opponent pool → 4P expansion.

```powershell
python experiments/train_shared_ppo.py --config configs/shared_ppo_2p_smoke.yaml --run-dir runs/shared_ppo_2p_smoke --device cpu
python .\experiments\train_shared_ppo.py --config .\configs\shared_ppo_2p_1m.yaml --run-dir .\runs\shared_ppo_2p_1m_seed42 --device cuda --progress always --stop-at-transitions 114688
```

The 1M schedule uses 16,384 transitions per update. The first 100k threshold is crossed after update 7 at 114,688 transitions. The run may be stopped after `step_000114688` evaluation and resumed without changing the schedule:

```powershell
python .\experiments\train_shared_ppo.py --config .\configs\shared_ppo_2p_1m.yaml --run-dir .\runs\shared_ppo_2p_1m_seed42 --resume .\runs\shared_ppo_2p_1m_seed42\checkpoints\step_000114688.pt --device cuda --progress always --stop-at-transitions 311296
python .\experiments\train_shared_ppo.py --config .\configs\shared_ppo_2p_1m.yaml --run-dir .\runs\shared_ppo_2p_1m_seed42 --resume .\runs\shared_ppo_2p_1m_seed42\checkpoints\step_000311296.pt --device cuda --progress always
python .\experiments\evaluate_shared_ppo.py --checkpoint .\runs\shared_ppo_2p_1m_seed42\checkpoints\best_average_rank.pt --games-per-matchup 1000 --output-dir .\runs\shared_ppo_2p_1m_seed42\final_evaluation --device cuda --actor-only --progress always
```

Each staged run keeps the full one-million-transition LR and entropy horizons. Use
`--progress never` for redirected/non-interactive jobs; `auto` is the default.
