# v0.6.1 performance

The optimized collector keeps game engines in spawned CPU workers. Trainable and
frozen PyTorch actors stay in the trainer, where observations are grouped by policy
ID and inferred in batches. This avoids CUDA contexts and model copies in workers.

On the local RTX system, a 4,096-transition collector benchmark from the same
checkpoint measured 77.59 transitions/s for `single_process` and 309.97
transitions/s for 8 workers × 8 environments (`3.99x`). A real migrated PPO update
with 4 × 8 measured 322.23 transitions/s in the progress timer. These are local
measurements, not a guarantee for another machine.

The remaining 29,618,304 transitions require roughly 26.5 hours at 310
transitions/s before scheduled evaluation overhead. Evaluation and machine load can
increase actual wall time.

Training-fast also changes orchestration costs: meta/promotion run every 2M,
meta uses at most 16 policies and 50 games per new pair, and each exploiter is
evaluated after its own 1M learner transitions. PPO hyperparameters remain unchanged.
