# Shared-parameter PPO

v0.4.0 provides a single-process synchronous PPO baseline. Every seat uses one shared actor and one shared privileged critic. The default actor and critic are separate 512×512×512 LayerNorm/SiLU MLPs with orthogonal initialization. Legal actions are enforced by a masked categorical distribution.

The smoke configuration is for pipeline verification, not evidence of learned play. Progress must be assessed against the fixed random, greedy, shortest-purchase, noble-focused, and reserve-blocking ladder rather than aggregate shared-policy self-play wins.

PyTorch remains an optional dependency: `pip install -e ".[rl]"`.
