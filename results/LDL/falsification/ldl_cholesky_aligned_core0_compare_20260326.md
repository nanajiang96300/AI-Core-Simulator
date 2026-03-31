# LDL vs Cholesky 完全对齐版（Core0）

## 1. 数据来源

- LDL aligned trace: `results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv`
- Cholesky inv*mul trace: `results/CHOL/falsification/cholesky_noblock_64x16_trace_invmul.csv`
- 统计范围：仅 `Core0_*` 事件

## 2. 总览对比

| 指标 | LDL-aligned | Cholesky-invmul | 比值(Chol/LDL) |
|---|---:|---:|---:|
| 总事件数 | 2164 | 7424 | 3.43x |
| 总事件时长 | 12779 | 18753 | 1.47x |
| max_end_cycle | 4282 | 15236 | 3.56x |

## 3. 单元统计

### 3.1 LDL-aligned

| 单元 | 事件数 | 时长 |
|---|---:|---:|
| `MTE2` | 160 | 9310 |
| `Vector` | 1964 | 2156 |
| `MTE3` | 32 | 1044 |
| `Cube` | 4 | 140 |
| `Wait` | 4 | 129 |

### 3.2 Cholesky-invmul

| 单元 | 事件数 | 时长 |
|---|---:|---:|
| `MTE2` | 160 | 9396 |
| `Vector` | 7216 | 7792 |
| `MTE3` | 32 | 1044 |
| `Cube` | 8 | 268 |
| `Wait` | 8 | 253 |

## 4. 分操作统计

### 4.1 LDL-aligned

| 操作 | 事件数 | 时长 |
|---|---:|---:|
| `Load` | 160 | 9310 |
| `Store` | 32 | 1044 |
| `LDL_L_UPDATE` | 480 | 480 |
| `LDL_BWD_OFF_MUL` | 480 | 480 |
| `LDL_BWD_OFF_ACC` | 480 | 480 |
| `LDL_D_DIAG_INV` | 64 | 256 |
| `LDL_GRAM` | 4 | 140 |
| `CubeWait` | 4 | 129 |
| `LDL_D_UPDATE` | 64 | 64 |
| `LDL_D_INV_MUL` | 64 | 64 |
| `LDL_BARRIER_BLDL_STEP` | 64 | 64 |
| `LDL_BARRIER_BWD_DIAG2OFF` | 64 | 64 |
| `LDL_BARRIER_BWD_COL` | 64 | 64 |
| `LDL_BWD_DIAG_MUL` | 60 | 60 |
| `LDL_BWD_DIAG_ACC` | 60 | 60 |
| `LDL_BARRIER_LOAD2GRAM` | 4 | 4 |
| `LDL_BARRIER_GRAM2REG` | 4 | 4 |
| `LDL_REG` | 4 | 4 |
| `LDL_BARRIER_REG2BLDL` | 4 | 4 |
| `LDL_BARRIER_BWD2STORE` | 4 | 4 |

### 4.2 Cholesky-invmul

| 操作 | 事件数 | 时长 |
|---|---:|---:|
| `Load` | 160 | 9396 |
| `CHOL_NB_RK_UPDATE` | 2720 | 2720 |
| `CHOL_NB_TRSM_NUM_UPD` | 2240 | 2240 |
| `Store` | 32 | 1044 |
| `CHOL_NB_TRSM_MUL` | 480 | 480 |
| `CHOL_NB_POTRF_DIAG_UPD` | 480 | 480 |
| `CHOL_NB_FWD_OFF_MAC` | 480 | 480 |
| `CHOL_NB_FWD_OFF_MUL` | 480 | 480 |
| `CHOL_NB_POTRF_DIAG_SQRT` | 64 | 256 |
| `CHOL_NB_TRSM_DIAG_INV` | 64 | 256 |
| `CHOL_NB_FWD_DIAG_INV` | 64 | 256 |
| `CubeWait` | 8 | 253 |
| `CHOL_NB_GRAM` | 4 | 140 |
| `CHOL_NB_BWD_MAC_FULL` | 4 | 128 |
| `CHOL_NB_BARRIER_FACTOR_STEP` | 64 | 64 |
| `CHOL_NB_BARRIER_FWD_COL` | 64 | 64 |
| `CHOL_NB_BARRIER_LOAD2GRAM` | 4 | 4 |
| `CHOL_NB_REG` | 4 | 4 |
| `CHOL_NB_BARRIER_REG2FACTOR` | 4 | 4 |
| `CHOL_NB_BARRIER_SOLVE2STORE` | 4 | 4 |

## 5. 结论（完全对齐后）

- LDL 经过对称 `inv+mul` 改造后，`D_INV` 路径从单步除法变为两步（`D_DIAG_INV + D_INV_MUL`），整体周期小幅上升。
- Cholesky 采用 `inv*mul` 后显著下降，但仍因 `RK_UPDATE`、`TRSM_NUM_UPD` 等高体量更新，整体事件与时长高于 LDL。
- 单核最终对比仍显示 Cholesky 更慢，这部分差异更接近算法/循环结构本身，而非单纯 DIV 建模差异。
