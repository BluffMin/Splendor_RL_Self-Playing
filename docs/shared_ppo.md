# Shared-parameter PPO

The original two-, three-, and four-player mode is named `shared_current` in v0.5.0
and remains available through `train_shared_ppo.py`. The separate `league_2p` mode
reuses the same masked PPO, GAE, actor, critic, schedules, and checkpoint machinery.

v0.4.1 stabilizes the single-process synchronous PPO baseline with numbered checkpoints, initial/periodic fixed-bot evaluation, best-checkpoint selection, resume-safe LR/entropy schedules, and epoch-mean target-KL stopping. The critic exposes private reservations but not hidden deck order.

v0.4.2 uses the same shared policy for 2, 3, or 4 players. Evaluation matchups, policy-seat rotation, seat metrics, replay metadata, and checkpoint validation all follow `num_players`.

Learning rate is computed immediately before each update from the transition count: `initial_lr + progress * (min_lr - initial_lr)`.

The smoke configuration is for pipeline verification, not evidence of learned play. Progress must be assessed against the fixed random, greedy, shortest-purchase, noble-focused, and reserve-blocking ladder rather than aggregate shared-policy self-play wins.

PyTorch remains an optional dependency: `pip install -e ".[rl]"`.

## Progress and staged runs

v0.4.3 displays collection/update state, transitions, schedule percentage, update and
episode counts, throughput, LR, entropy coefficient, KL, and safety counters. Fixed-bot
evaluation displays total games, matchup, policy seat, running rank, and first-place rate.
Progress is written only to stderr; JSONL and result files remain machine-readable.

`--progress auto` enables bars only on an interactive stderr, `always` forces them on
(useful in PowerShell), and `never` disables them. ETA is an estimate based only on the
current process. `--stop-at-transitions` is an absolute run target: it does not change
the LR or entropy schedule horizon stored in `total_transitions`.
