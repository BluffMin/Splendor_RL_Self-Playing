# Prioritized fictitious self-play (PFSP)

For Candidate score `s`, with a Beta(1,1) prior for sparse records:

```text
difficulty = 4 s (1 - s)
weight = (epsilon + difficulty) ** alpha
probability_i = weight_i / sum(weight)
```

Defaults are `epsilon=0.05` and `alpha=1`. Every opponent therefore remains
selectable, including unseen and apparently very weak/strong policies, while scores
near 0.5 receive the most weight. The default historical set contains Hall-of-Fame
Champions other than the current Champion plus recent Candidate snapshots. Current
Champion has its own episode category and is not duplicated in historical PFSP.
