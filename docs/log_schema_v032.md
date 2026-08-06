# Episode log schema v0.3.2

Native logs contain `game_metadata`, `initial_state`, decision-level `events`, player-level `turns`, `final_summary`, and `final_state_hash`. Every decision event stores phase, actor, action, hashes, and pre/post snapshots. `TurnRecord` groups all decisions sharing a player-turn ID and stores the normal-phase primary action plus payment, discard, purchase, reservation, token, noble, score, and snapshot summaries.

Turn-mode replay uses the record's full pre/post snapshots. Decision mode uses each decision's snapshots. This prevents a payment choice from replacing a purchase as the visible turn title.
