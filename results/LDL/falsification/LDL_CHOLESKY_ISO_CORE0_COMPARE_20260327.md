# LDL vs Cholesky(strict-iso) Core0 对比

- LDL Core0 max_end_cycle: `4282`
- Cholesky Core0 max_end_cycle: `6276`
- cycle ratio (CHOL/LDL): `1.4657`

## 分操作耗时与事件数（Core0）

| Operation | LDL cnt | LDL dur | CHOL cnt | CHOL dur | CHOL-LDL dur |
|---|---:|---:|---:|---:|---:|
| CHOL_NB_BARRIER_LOAD2GRAM | 0 | 0 | 4 | 4 | 4 |
| CHOL_NB_BARRIER_REG2FACTOR | 0 | 0 | 4 | 4 | 4 |
| CHOL_NB_GRAM | 0 | 0 | 4 | 140 | 140 |
| CHOL_NB_ISO_BARRIER_FACTOR_STEP | 0 | 0 | 64 | 64 | 64 |
| CHOL_NB_ISO_BARRIER_FWD_COL | 0 | 0 | 64 | 64 | 64 |
| CHOL_NB_ISO_BARRIER_SOLVE2STORE | 0 | 0 | 4 | 4 | 4 |
| CHOL_NB_ISO_BWD_MAC_FULL | 0 | 0 | 4 | 128 | 128 |
| CHOL_NB_ISO_FWD_DIAG_INV | 0 | 0 | 64 | 256 | 256 |
| CHOL_NB_ISO_FWD_OFF_MAC | 0 | 0 | 480 | 480 | 480 |
| CHOL_NB_ISO_FWD_OFF_MUL | 0 | 0 | 480 | 480 | 480 |
| CHOL_NB_ISO_POTRF_DIAG_SQRT | 0 | 0 | 64 | 256 | 256 |
| CHOL_NB_ISO_POTRF_DIAG_UPD | 0 | 0 | 60 | 60 | 60 |
| CHOL_NB_ISO_RK_UPDATE | 0 | 0 | 480 | 480 | 480 |
| CHOL_NB_ISO_TRSM_DIAG_INV | 0 | 0 | 64 | 256 | 256 |
| CHOL_NB_ISO_TRSM_MUL | 0 | 0 | 480 | 480 | 480 |
| CHOL_NB_ISO_TRSM_NUM_UPD | 0 | 0 | 420 | 420 | 420 |
| CHOL_NB_REG | 0 | 0 | 4 | 4 | 4 |
| CubeWait | 4 | 129 | 8 | 253 | 124 |
| LDL_BARRIER_BLDL_STEP | 64 | 64 | 0 | 0 | -64 |
| LDL_BARRIER_BWD2STORE | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_BWD_COL | 64 | 64 | 0 | 0 | -64 |
| LDL_BARRIER_BWD_DIAG2OFF | 64 | 64 | 0 | 0 | -64 |
| LDL_BARRIER_GRAM2REG | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_LOAD2GRAM | 4 | 4 | 0 | 0 | -4 |
| LDL_BARRIER_REG2BLDL | 4 | 4 | 0 | 0 | -4 |
| LDL_BWD_DIAG_ACC | 60 | 60 | 0 | 0 | -60 |
| LDL_BWD_DIAG_MUL | 60 | 60 | 0 | 0 | -60 |
| LDL_BWD_OFF_ACC | 480 | 480 | 0 | 0 | -480 |
| LDL_BWD_OFF_MUL | 480 | 480 | 0 | 0 | -480 |
| LDL_D_DIAG_INV | 64 | 256 | 0 | 0 | -256 |
| LDL_D_INV_MUL | 64 | 64 | 0 | 0 | -64 |
| LDL_D_UPDATE | 64 | 64 | 0 | 0 | -64 |
| LDL_GRAM | 4 | 140 | 0 | 0 | -140 |
| LDL_L_UPDATE | 480 | 480 | 0 | 0 | -480 |
| LDL_REG | 4 | 4 | 0 | 0 | -4 |
| Load | 160 | 9310 | 160 | 9396 | 86 |
| Store | 32 | 1044 | 32 | 1044 | 0 |

## 结论（按 Core0 统计）

- Factor阶段时长: LDL `864` vs CHOL `1952` (ratio `2.259`)
- Solve阶段时长: LDL `1080` vs CHOL `1344` (ratio `1.244`)
- LDL 优势主要来自 factor 阶段：CHOL 仍包含 `TRSM_NUM_UPD` 和 `RK_UPDATE` 两类额外更新，即使 strict-iso 聚合后，事件和依赖链仍更长。
- `SQRT` 仅占 CHOL 总时长的小部分，不是主导差异。