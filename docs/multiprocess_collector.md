# Multiprocess batched collector

`multiprocess_batched` uses the Windows-compatible `spawn` context. Module-level
workers own only Splendor environments and return observations, critic states, and
legal masks in batches. The trainer groups requests by policy ID, performs batched
inference, and sends actions back through process pipes.

Workers never own a CUDA model or print progress. Failures include worker ID,
environment ID, state hash, exception, and traceback. Normal completion and
KeyboardInterrupt close pipes, join workers, and terminate only an unresponsive
worker. CPU PyTorch threads are restricted to one inside each worker.

Both backends preserve legal-action masking, candidate-only PPO trajectories,
terminal rewards, variable discounts, and GAE semantics. Stochastic RNG ordering
can differ between backends.
