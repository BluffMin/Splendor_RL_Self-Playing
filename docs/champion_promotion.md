# Champion promotion

v0.5.1 requires paired Candidate-vs-Champion success plus hard and saturated anchor gates. By default, hard aggregate regression is limited to 0.02, each hard anchor to 0.05, and each saturated anchor to 0.04.

Each paired seed produces Candidate-P0 and Candidate-P1 games against the same frozen
Champion. A pair score averages the two 1/0.5/0 results. NumPy bootstrap resampling of
pair scores gives a deterministic confidence interval.

Default head-to-head requirements are mean score `>= 0.55` and lower 95% bound
strictly `> 0.50`. Identical actor hashes cannot promote. Candidate and Champion are
also compared with paired seats against Random, Greedy, Shortest, Noble, Blocking,
and available historical Champion anchors. Aggregate regression must be `<= 0.03`
and every anchor regression `<= 0.07` by default.

Success archives a new immutable Champion and updates the pointer without resetting
Candidate, critic, optimizer, schedules, or transitions. Failure leaves both policies
in place and training continues.
