# Changelog

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
