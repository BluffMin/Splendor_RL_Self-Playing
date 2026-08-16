# Changelog

## v0.6.1 - High-Throughput Population Training

- Added BALANCED and MAXIMUM resource presets. BALANCED reserves CPU, RAM, and
  VRAM headroom for Excel and other foreground desktop applications.
- Added resource-aware auto-tuning; configurations within 10% throughput favor
  materially lower CPU/RAM use in BALANCED mode.
- Added Windows-spawn multiprocess environment collection with centralized,
  policy-wise batched actor and critic inference.
- Added bounded frozen-actor caching, profiling, throughput metrics, collector
  benchmarking, clean worker shutdown, and explicit worker failure propagation.
- Added backward-compatible v0.6.0 state migration while preserving optimizers,
  schedules, learner counters, population state, and the 50M horizon.
- Moved exploiter gates to learner-local cadence and reduced training-time meta
  and promotion evaluation cadence in the fast preset.
- Reduced full tensor checkpoints from every PPO update to an authoritative 1M
  cadence, with diagnostic-only lightweight state at 250k.
- Reuse population meta-game pair evaluations by immutable actor hash, including
  deterministic recovery of legacy results at the exact replayed boundary.
- Treat meta-mixture and League Exploiter pair counts as fixed total evaluation
  budgets instead of multiplying them by the number of policies in the pool.

## v0.6.0 - Population League, Exploiters, and PSRO-lite

- Added five independent sequential PPO learners: Main, two Main Exploiters, and two League Exploiters.
- Added archived exploiters, empirical meta-game matrices, regret-matching PSRO-lite, diverse population sampling, and robust Champion gates.
- Added empirical exploitability proxy, cycle diagnostics, and population checkpoint/resume.
- Preserved the v0.4 shared-PPO and v0.5 league modes.

## v0.5.1 - Empirical league bootstrap and robust Champion selection

- Added tournament-based bootstrap Champion selection and a diverse initial population.
- Added hard/saturated anchor groups, robust promotion gates, paired evaluation, and real round metrics.
- Preserved self-play-only PPO training by default.

## v0.5.0 - Two-player PPO league self-play

- Separated Candidate and Champion policies.
- Added a frozen historical opponent pool and PFSP selection.
- Added mixed current/champion/historical self-play and Candidate-only PPO trajectories.
- Added paired-seat Champion promotion with bootstrap confidence intervals.
- Added fixed-bot and historical regression gates and a Hall-of-Fame archive.
- Added atomic league state, resume, matchup matrices, and population diagnostics.
- Preserved existing two- and four-player `shared_current` PPO modes.

## v0.4.3 - Progress monitoring and staged training runs

- Added tqdm-based training and fixed-bot evaluation progress.
- Added PowerShell-friendly `auto`, `always`, and `never` progress modes.
- Added transition ETA and compact PPO status metrics.
- Added absolute `--stop-at-transitions` staged runs without shortening schedules.
- Added resume-aware progress and progress-safe checkpoint/evaluation messages.
- Converted no-official-action collector deadlocks into recorded safety truncations.
- Fixed CUDA resume by restoring CPU/CUDA RNG states from CPU ByteTensors.

## v0.4.2 - First-class two-player self-play support

- Removed four-player assumptions from fixed-bot evaluation.
- Added player-count-aware matchups, balanced seat rotation, and dynamic seat metrics.
- Added two-player smoke and one-million-transition configurations.
- Added player-count checkpoint metadata and validation.
- Added two-player replay validation while preserving four-player compatibility.

## v0.4.1 - PPO training stabilization

- Corrected truncation-state critic bootstrapping.
- Added initial and periodic fixed-bot evaluations and numbered checkpoints.
- Added automatic best-checkpoint selection.
- Implemented resume-safe transition-based LR and entropy schedules.
- Changed target-KL stopping to epoch-level statistics.
- Clarified privileged critic information scope and v0.4.0 compatibility.
- Added a one-million-transition experiment configuration.

## v0.4.0 - Shared-parameter PPO self-play baseline

- Shared actor across all players and seats with a separate CTDE critic.
- Egocentric actor observations and legal-action masked categorical policy.
- Canonical payment wrapper over the exact engine.
- Constant-sum terminal rank rewards and player-specific trajectories.
- Variable-discount GAE, PPO checkpoints/resume, and fixed-bot evaluation.
- v0.3.2-native evaluation replay export.

## v0.3.2 - Correct player-turn semantics and legacy-log migration

- Separated decision, player turn, and round counters.
- Added `DecisionEvent` and `TurnRecord` schemas.
- Fixed acting-player versus next-player rendering and turn navigation.
- Grouped payment, discard, and noble choices into one player turn.
- Added schema-versioned recording, legacy loading, and non-destructive migration.

## 0.3.1 — Visual board and replay viewer

- Added an offline browser board visualization with original CSS card/token graphics.
- Added adaptive 2/3/4-player table and egocentric layouts.
- Added perspective-safe private reservation rendering and sanitized exports.
- Added decision/turn replay controls, autoplay, event log, and state-delta highlights.
- Added final-hand detail views and multi-game final-board comparison.
- No game rule, action mask, reward, observation, or state-transition changes.

## 0.3.0

- Standardized gem order and stable human-readable card/noble IDs.
- Added exhaustive deterministic payment and discard plans.
- Replaced pass and single-token discard actions with a 373-action phase layout.
- Moved optional `max_turns` truncation into the PettingZoo adapter.
- Added explicit phase/turn metadata, invariants, rankings, summaries, state hashes.
- Added passive JSON/CSV recording, perspective replay, and hash verification.
- Added random/greedy validation agents and multi-game demo generation.
- Expanded observations to include payment/discard pending context.

## 0.2.0

- Added perspective-correct reservation visibility in observations and rendering.
