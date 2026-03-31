# LDL vs Cholesky（No-Block, Core0）操作统计与周期差异分析

## 1. 数据来源

- LDL trace: `results/LDL/falsification/ldl_noblock_64x16_trace_fix3.csv`
- Cholesky trace: `results/CHOL/falsification/cholesky_noblock_64x16_trace_fix.csv`
- 统计范围: 仅 `Core0_*` 事件

## 2. 总览对比（Core0）

| 指标 | LDL | Cholesky | 比值(Chol/LDL) |
|---|---:|---:|---:|
| 总事件数 | 2100 | 7360 | 3.50x |
| 总事件时长（事件持续周期和） | 12715 | 21377 | 1.68x |
| `max_end_cycle`（Core0 最晚结束） | 4154 | 17796 | 4.28x |

## 3. 单元级统计（Core0）

### 3.1 LDL

| 单元 | 事件数 | 时长 |
|---|---:|---:|
| `MTE2` | 160 | 9310 |
| `Vector` | 1900 | 2092 |
| `MTE3` | 32 | 1044 |
| `Cube` | 4 | 140 |
| `Wait` | 4 | 129 |

### 3.2 Cholesky

| 单元 | 事件数 | 时长 |
|---|---:|---:|
| `Vector` | 7152 | 10416 |
| `MTE2` | 160 | 9396 |
| `MTE3` | 32 | 1044 |
| `Cube` | 8 | 268 |
| `Wait` | 8 | 253 |

## 4. 分操作统计（Core0）

### 4.1 LDL

| 操作 | 事件数 | 时长 |
|---|---:|---:|
| `Load` | 160 | 9310 |
| `Store` | 32 | 1044 |
| `LDL_L_UPDATE` | 480 | 480 |
| `LDL_BWD_OFF_MUL` | 480 | 480 |
| `LDL_BWD_OFF_ACC` | 480 | 480 |
| `LDL_D_INV` | 64 | 256 |
| `LDL_GRAM` | 4 | 140 |
| `CubeWait` | 4 | 129 |
| `LDL_D_UPDATE` | 64 | 64 |
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

### 4.2 Cholesky

| 操作 | 事件数 | 时长 |
|---|---:|---:|
| `Load` | 160 | 9396 |
| `CHOL_NB_RK_UPDATE` | 2720 | 2720 |
| `CHOL_NB_TRSM_NUM_UPD` | 2240 | 2240 |
| `CHOL_NB_TRSM_DIV` | 480 | 1920 |
| `CHOL_NB_FWD_OFF_UPD` | 480 | 1920 |
| `Store` | 32 | 1044 |
| `CHOL_NB_POTRF_DIAG_UPD` | 480 | 480 |
| `CHOL_NB_FWD_OFF_MAC` | 480 | 480 |
| `CHOL_NB_POTRF_DIAG_SQRT` | 64 | 256 |
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

## 5. 为什么 LDL 比 Cholesky 快这么多

1. **事件数量本身显著更少（主因）**
   - LDL 总事件 `2100`，Cholesky `7360`（`3.5x`）。
   - Cholesky 在 `RK_UPDATE`、`TRSM_NUM_UPD` 上有大量细粒度更新（`2720 + 2240`），显著抬高总执行长度。

2. **Cholesky 的高延迟向量算子更多**
   - `CHOL_NB_TRSM_DIV` 和 `CHOL_NB_FWD_OFF_UPD` 各 `480` 次、总时长都 `1920`。
   - 这类操作平均每次约 `4` 周期，高于 LDL 大部分 `1` 周期级的 `MAC/ADD` 更新。

3. **LDL 在 no-block/fix3 下把大部分分解更新压成低开销向量路径**
   - LDL 分解核心 (`LDL_L_UPDATE`, `LDL_BWD_OFF_MUL`, `LDL_BWD_OFF_ACC`) 都是 `1` 周期级累计，且无大规模长尾等待。
   - `D_UPDATE` 已全部转到 Vector（`1536` 次在全局上），显著减少 Cube/Wait 干扰。

4. **内存项两者接近，不是差距主因**
   - `Load/Store` 时长接近（LDL: `9310/1044`, Chol: `9396/1044`）。
   - 主要差异来自计算路径（尤其是 Cholesky 的分解与前向更新阶段），不是 DRAM。

5. **结果体现到关键结束周期**
   - `max_end_cycle`: LDL `4154` vs Cholesky `17796`（`4.28x`）。
   - 与“Cholesky 操作数更多 + 单次操作平均更重”一致。

## 6. 结论

在相同 no-block 条件下，LDL 的分解/回代路径在当前建模中更“扁平”和低延迟，而 Cholesky 包含更密集的更新与高延迟向量除法步骤，因此 Cholesky 的总周期显著更高。

## 7. 对齐实验：Cholesky 改为 `inv*mul` 路径

### 7.1 实验改动

文件：`src/operations/CholeskyInvNoBlockOp.cc`

- 新增 `CHOL_NB_TRSM_DIAG_INV_j`：每列仅 1 次 `DIV` 计算对角倒数；
- 将 `CHOL_NB_TRSM_DIV_i_j` 改为 `CHOL_NB_TRSM_MUL_i_j`；
- 将 `CHOL_NB_FWD_OFF_UPD_i_c`（原 `DIV`）改为 `CHOL_NB_FWD_OFF_MUL_i_c`。

新 trace / 图：

- `results/CHOL/falsification/cholesky_noblock_64x16_trace_invmul.csv`
- `results/CHOL/falsification/cholesky_noblock_64x16_timeline_invmul.png`

### 7.2 Core0 总览（旧 Cholesky vs 新 Cholesky）

| 指标 | Cholesky-旧 | Cholesky-invmul | 变化 |
|---|---:|---:|---:|
| 总事件数 | 7360 | 7424 | +64 |
| 总事件时长 | 21377 | 18753 | -2624 |
| `max_end_cycle` | 17796 | 15236 | -2560 |

说明：事件数增加 64 是因为新增了每列一次的 `TRSM_DIAG_INV`；但大量高延迟逐元素 `DIV` 被 `MUL` 替代，因此总时长和结束周期下降明显。

### 7.3 关键操作对比（Core0）

| 操作 | 旧事件数/时长 | 新事件数/时长 |
|---|---:|---:|
| `CHOL_NB_TRSM_DIV` | 480 / 1920 | 0 / 0 |
| `CHOL_NB_FWD_OFF_UPD` | 480 / 1920 | 0 / 0 |
| `CHOL_NB_TRSM_MUL` | 0 / 0 | 480 / 480 |
| `CHOL_NB_FWD_OFF_MUL` | 0 / 0 | 480 / 480 |
| `CHOL_NB_TRSM_DIAG_INV` | 0 / 0 | 64 / 256 |

结论：把逐元素除法改成 “少量对角求逆 + 大量乘法” 后，Cholesky 的重操作开销显著下降。

### 7.4 与 LDL（fix3）重新对比（Core0）

| 指标 | LDL-fix3 | Cholesky-旧 | Cholesky-invmul |
|---|---:|---:|---:|
| 总事件数 | 2100 | 7360 | 7424 |
| 总事件时长 | 12715 | 21377 | 18753 |
| `max_end_cycle` | 4154 | 17796 | 15236 |

对齐实验后，Cholesky 相比 LDL 的差距缩小（尤其是时长与结束周期），但仍显著偏大，主因仍是：

- `CHOL_NB_RK_UPDATE` 与 `CHOL_NB_TRSM_NUM_UPD` 的事件体量远高于 LDL 对应路径；
- Cholesky 的前向/分解阶段仍有更密集的细粒度更新链。
