# Changelog

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
