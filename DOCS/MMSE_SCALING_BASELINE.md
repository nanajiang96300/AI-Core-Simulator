# MMSE Baseline Scaling (Ascend 910B)

Cycles vs. antenna configuration for the baseline MMSE operator (batch=96, Newton–Schulz iterations=10).

Note: per-unit numbers are wall-clock cycles where at least one core of that unit type is active (union of intervals across cores), not a sum of per-core active cycles.


## Fixed K = 32 (varying M)

| M | K | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |
|---|---|--------------|------|--------|----------------|-------|
| 64 | 32 | 1179 | 4049 | 3023 | 1127 | 4187 |
| 128 | 32 | 1932 | 4779 | 3117 | 1538 | 4912 |
| 256 | 32 | 3721 | 6702 | 3666 | 2497 | 6954 |
| 512 | 32 | 5840 | 8736 | 4139 | 3416 | 9396 |
| 1024 | 32 | 11276 | 13457 | 4894 | 6953 | 14765 |

## Fixed M = 256 (varying K)

| M | K | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |
|---|---|--------------|------|--------|----------------|-------|
| 256 | 16 | 1724 | 4297 | 2021 | 1369 | 4430 |
| 256 | 32 | 3721 | 6702 | 3666 | 2497 | 6954 |
| 256 | 64 | 7384 | 10494 | 7979 | 5064 | 11484 |
| 256 | 128 | 17663 | 19529 | 17243 | 11442 | 21883 |