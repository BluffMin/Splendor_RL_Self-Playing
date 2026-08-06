# CTDE privileged critic

The actor receives the public egocentric observation and action mask. The v0.4.1 critic receives the same 475-value layout with privileged private-reservation payloads exposed. It does **not** encode the complete hidden deck order, so it is a privileged hidden-reservation critic rather than a full-state critic. Evaluation requires only the actor.

A future optional full-state critic may add an explicit hidden deck-sequence encoder; it is not part of v0.4.1.
