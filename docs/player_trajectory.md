# Player-specific trajectories

The collector maintains a pending transition for each `(environment, player)`. It closes that transition only when the same player acts again or the episode ends. Other seats' decisions never enter that player's trajectory.

Consecutive sub-decisions in the same `player_turn_id` use discount 1.0. The next decision on a later player turn uses `gamma`. Official termination assigns constant-sum rank utility and bootstrap zero to every pending player. Safety truncation gives zero reward and permits critic bootstrap.
