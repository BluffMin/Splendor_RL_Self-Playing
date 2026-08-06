# Shared-parameter PPO

v0.4.1 stabilizes the single-process synchronous PPO baseline with numbered checkpoints, initial/periodic fixed-bot evaluation, best-checkpoint selection, resume-safe LR/entropy schedules, and epoch-mean target-KL stopping. The critic exposes private reservations but not hidden deck order.

Learning rate is computed immediately before each update from the transition count: `initial_lr + progress * (min_lr - initial_lr)`.

The smoke configuration is for pipeline verification, not evidence of learned play. Progress must be assessed against the fixed random, greedy, shortest-purchase, noble-focused, and reserve-blocking ladder rather than aggregate shared-policy self-play wins.

PyTorch remains an optional dependency: `pip install -e ".[rl]"`.
