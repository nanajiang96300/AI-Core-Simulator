# Blocked Cholesky vs Blocked LDL (Core0)

- LDL Core0 max_end_cycle: `3600`
- Cholesky Core0 max_end_cycle: `13632`
- ratio (CHOL/LDL): `3.7867`

## Per-operation table (Core0)

| Operation | LDL cnt | LDL dur | CHOL cnt | CHOL dur | CHOL-LDL dur |
|---|---:|---:|---:|---:|---:|
| CHOL_BARRIER_FACTOR_STEP | 0 | 0 | 32 | 32 | 32 |
| CHOL_BARRIER_FWD_COL | 0 | 0 | 32 | 32 | 32 |
| CHOL_BARRIER_LOAD2GRAM | 0 | 0 | 4 | 4 | 4 |
| CHOL_BARRIER_REG2FACTOR | 0 | 0 | 4 | 4 | 4 |
| CHOL_BARRIER_SOLVE2STORE | 0 | 0 | 4 | 4 | 4 |
| CHOL_BWD_MAC_FULL | 0 | 0 | 4 | 128 | 128 |
| CHOL_FWD_DIAG_INV | 0 | 0 | 32 | 128 | 128 |
| CHOL_FWD_OFF_MAC | 0 | 0 | 112 | 3584 | 3584 |
| CHOL_FWD_OFF_UPD | 0 | 0 | 112 | 448 | 448 |
| CHOL_GRAM | 0 | 0 | 4 | 140 | 140 |
| CHOL_POTRF_DIAG_SQRT | 0 | 0 | 32 | 128 | 128 |
| CHOL_POTRF_DIAG_UPD | 0 | 0 | 112 | 3584 | 3584 |
| CHOL_REG | 0 | 0 | 4 | 4 | 4 |
| CHOL_RK_UPDATE | 0 | 0 | 336 | 10752 | 10752 |
| CHOL_TRSM_DIV | 0 | 0 | 112 | 448 | 448 |
| CHOL_TRSM_NUM_UPD | 0 | 0 | 224 | 7168 | 7168 |
| CubeWait | 54 | 884 | 674 | 95437 | 94553 |
| LDL_BARRIER_BLDL_STEP | 32 | 32 | 0 | 0 | -32 |
| LDL_BARRIER_BWD2STORE | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_BWD_COL | 32 | 32 | 0 | 0 | -32 |
| LDL_BARRIER_BWD_DIAG2OFF | 32 | 32 | 0 | 0 | -32 |
| LDL_BARRIER_GRAM2REG | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_LOAD2GRAM | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_REG2BLDL | 4 | 4 | 0 | 0 | -4 |
| LDL_BWD_DIAG_ACC | 28 | 28 | 0 | 0 | -28 |
| LDL_BWD_DIAG_MUL | 28 | 772 | 0 | 0 | -772 |
| LDL_BWD_OFF_ACC | 112 | 112 | 0 | 0 | -112 |
| LDL_BWD_OFF_MUL | 112 | 3460 | 0 | 0 | -3460 |
| LDL_D_DIAG_INV | 32 | 128 | 0 | 0 | -128 |
| LDL_D_INV_MUL | 32 | 32 | 0 | 0 | -32 |
| LDL_D_UPDATE | 32 | 1024 | 0 | 0 | -1024 |
| LDL_GRAM | 4 | 140 | 0 | 0 | -140 |
| LDL_L_UPDATE | 28 | 772 | 0 | 0 | -772 |
| LDL_REG | 4 | 4 | 0 | 0 | -4 |
| Load | 160 | 10497 | 160 | 10030 | -467 |
| Store | 32 | 1806 | 32 | 1866 | 60 |