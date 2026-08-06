# Time semantics

The engine separates three clocks. A **decision** is one agent choice and increments `decision_id`. A **player turn** starts with a normal-phase action and includes any payment, discard, and noble choices; `player_turn_id` increments only after control passes or the game ends. A **round** is one complete seat cycle, so `round_id = player_turn_id // num_players`.

All stored IDs are zero-based. User interfaces display them as ID + 1. `acting_player` identifies the player responsible for an event or turn; `next_player` is populated only after a turn completes. Automatic market refill or single-choice resolution does not create a new player turn or decision.

Example: buying a card and then choosing payment produces decisions `(8, 0)` and `(8, 1)` for the same `player_turn_id=8`. Only the payment event completes that turn.
