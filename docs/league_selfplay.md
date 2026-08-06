# PPO-based league self-play

An empirical bootstrap manifest may initialize the Candidate, `champion_0000`, and a diverse historical pool. Fixed bots remain evaluation anchors and do not generate PPO trajectories when `scripted_training_fraction` is its default `0.0`.

`league_2p` supports exactly two players. The Candidate owns the trainable actor,
privileged critic, optimizer, and full schedule. The Champion is the latest actor-only
snapshot that passed promotion. Candidate training may fluctuate; Champion changes
only after a gate, which is a practical regression guard rather than a proof of
monotonic strength against every strategy.

Episodes mix Candidate-vs-Candidate (20%), Candidate-vs-Champion (40%), and
Candidate-vs-historical PFSP (40%). When history is empty, the available categories
are deterministically renormalized. Both seats learn in current self-play; only the
Candidate seat enters PPO against frozen opponents. Frozen actors receive public
egocentric observations and legal masks, never critic state or hidden reservations.

Recent snapshots are capped; Hall-of-Fame Champions are permanent. Atomic
`league_state.json`, pool index, match records, separated RNG streams, thresholds,
seat balancing, and Candidate checkpoints support resume. `--stop-at-transitions`
does not shorten LR or entropy horizons.

The 1M diagnostic sequence attempts promotion at 250k, 500k, 750k, and 1M, with
matrices at 500k and 1M. Game length is diagnostic only and never reward shaping.

This shares self-play without demonstrations, stronger checkpoint preservation, and
final-outcome learning with AlphaGo Zero-style motivations. It differs fundamentally:
there is no MCTS or visit-count target, PPO is used, Splendor has private information,
PFSP drives a league, and the privileged critic exists only during training.
