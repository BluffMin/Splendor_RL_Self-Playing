# CTDE privileged critic

The actor receives `game.observation(player, omniscient=False)` and the action mask. The critic independently receives the egocentrically rotated omniscient observation. Hidden deck reservations can therefore affect value estimation but cannot enter actor logits. Evaluation and checkpoint watching load and execute the actor without requiring the critic.
