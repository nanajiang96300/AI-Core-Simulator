# LDL vs Cholesky 详细综合报告（含分块前后）

## 1) 结论先回答你的问题

你问的是：**“LDL 的矩阵运算更多但是反而慢一点，这个结论对吗？”**

结论：**不完全对，需要区分“运算次数”与“运算耗时/映射单元”。**

- 在 `no-block`（`LDL-aligned` vs `CHOL-iso`）里，**LDL 的矩阵更新事件数更少、总周期也更短**。
- 在 `block_size=2`（优化后 `CHOL-block-opt` vs `LDL-block`）里，**LDL 并不是“矩阵运算次数更多”**；
  但 LDL 的矩阵相关计算**落在 Cube 的高时延路径更多**，因此其矩阵计算总时长更高，最终全局周期反而略慢于优化后的 Cholesky。

---

## 2) 数据来源与口径

- no-block 对比：
  - `results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv`
  - `results/CHOL/falsification/cholesky_noblock_64x16_trace_iso.csv`
- block 对比：
  - `results/LDL/falsification/ldl_block_64x16_trace.csv`
  - `results/CHOL/falsification/cholesky_block_64x16_trace.csv`（优化前）
  - `results/CHOL/falsification/cholesky_block_64x16_trace_opt.csv`（优化后）
- no-block 演进汇总：
  - `results/LDL/falsification/cholesky_ldl_evolution_summary_20260327.csv`

说明：
- `max_end` 代表全局关键路径周期（越小越好）。
- `matrix_cnt / matrix_dur` 的统计口径如下：
  - LDL: `D_UPDATE/L_UPDATE/BWD_DIAG_MUL/BWD_OFF_MUL`
  - CHOL(no-block): `POTRF_DIAG_UPD/TRSM_NUM_UPD/TRSM_MUL/RK_UPDATE/FWD_OFF_MAC/FWD_OFF_MUL/BWD_MAC_FULL`
  - CHOL(block): `POTRF_DIAG_UPD/TRSM_NUM_UPD/RK_UPDATE/FWD_OFF_MAC/FWD_OFF_UPD/BWD_MAC_FULL`

---

## 3) no-block（分块前）结果

### 3.1 演进与对齐结果（全局）

| Case | events | total_dur | max_end |
|---|---:|---:|---:|
| CHOL-old | 307796 | 29467524 | 97792 |
| CHOL-invmul | 178173 | 414276 | 15244 |
| CHOL-iso | 70653 | 306756 | 6284 |
| LDL-aligned | 51933 | 273400 | 4290 |

关键结论：
- Cholesky 从 old→iso 的优化收益显著：`97792 -> 6284`（约 `15.56x`）。
- 但在 no-block 最终对齐后，LDL 仍更快：`4290 vs 6284`（CHOL/LDL=`1.465x`）。

### 3.2 no-block 矩阵相关统计（全局）

| Case | matrix_cnt | matrix_dur | max_end |
|---|---:|---:|---:|
| LDL-aligned | 26016 | 26016 | 4290 |
| CHOL-iso | 57696 | 60672 | 6284 |

关键结论：
- no-block 下，**CHOL 的矩阵相关事件数与时长都高于 LDL**，且周期更慢。
- 这与你之前的“LDL 矩阵运算更多”说法不一致（no-block 下不成立）。

---

## 4) block（分块后）结果

### 4.1 优化前后对比（全局）

| Case | events | total_dur | max_end |
|---|---:|---:|---:|
| CHOL-block-old | 48614 | 3185924 | 13697 |
| CHOL-block-opt | 32637 | 275652 | 3404 |
| LDL-block | 18497 | 424972 | 3704 |

关键结论：
- Cholesky 分块优化后：`13697 -> 3404`，约 `4.02x` 提升。
- 优化后 Cholesky 已略快于 LDL：`3404 vs 3704`（CHOL/LDL=`0.919`）。

### 4.2 block 矩阵相关统计（全局）

| Case | matrix_cnt | matrix_dur | max_end |
|---|---:|---:|---:|
| LDL-block | 4800 | 144672 | 3704 |
| CHOL-block-old | 21600 | 615936 | 13697 |
| CHOL-block-opt | 21600 | 32640 | 3404 |

关键结论：
- **按“次数”看**：block 下是 Cholesky 矩阵事件更多（`21600`）而非 LDL 更多（`4800`）。
- **按“耗时”看**：优化后 LDL 的矩阵总时长反而更高（`144672` vs `32640`），这是 LDL 在 block 对比中略慢的主要原因。

### 4.3 block 单元占用变化（CHOL 优化前后）

| Case | Cube_dur | Vector_dur | Wait_dur | max_end |
|---|---:|---:|---:|---:|
| CHOL-block-old | 608544 | 29568 | 2289266 | 13697 |
| CHOL-block-opt | 6432 | 48384 | 5927 | 3404 |

关键结论：
- 旧版 Cholesky 分块主要问题是：小块微更新大量落到 Cube，导致 Wait 激增。
- 优化后小块改走 Vector，`Cube_dur` 和 `Wait_dur` 均大幅下降，关键路径被显著缩短。

---

## 5) Core0 验证（与全局趋势一致）

| Case | core0_end | core0_events | core0_total_dur | core0_matrix_cnt | core0_matrix_dur |
|---|---:|---:|---:|---:|---:|
| LDL no-block | 4282 | 2164 | 12779 | 1084 | 1084 |
| CHOL no-block iso | 6276 | 2944 | 14273 | 2404 | 2528 |
| LDL block | 3600 | 770 | 19771 | 200 | 6028 |
| CHOL block opt | 3396 | 1360 | 12977 | 900 | 1360 |

关键结论：
- no-block：LDL Core0 更短（`4282 < 6276`）。
- block 优化后：CHOL Core0 更短（`3396 < 3600`）。
- 与全局 `max_end` 结论一致，说明结论稳定，不是单核偶然现象。

---

## 6) 对“为何会这样”的详细说明

1. no-block 阶段：
   - Cholesky 即便 strict-iso 后，因子阶段依赖链仍更长（`TRSM_NUM_UPD/RK_UPDATE` 等），
   - 导致其矩阵相关事件和时长都高于 LDL，最终周期慢于 LDL。

2. block 阶段（优化前）：
   - `2x2` 微更新错误映射到 Cube，造成大量 `CubeWait`，
   - 关键路径被等待拖长，出现 `13697` 的异常高周期。

3. block 阶段（优化后）：
   - `blk<=2` 的微更新改走 Vector（`MAC`）后，
   - Cholesky 的等待和 Cube 压力显著降低，整体周期降到 `3404`。
   - LDL 仍包含部分长时延 Cube 路径（如 `D_UPDATE`/`BWD_*_MUL` 的累计），因此在该配置下略慢于 CHOL-opt。

---

## 7) 最终判断（可直接引用）

- “LDL 的矩阵运算更多但是反而慢一点”这句话：
  - **按运算次数（events）统计：不正确**（block/no-block 均不是 LDL 更多）。
  - **按矩阵计算耗时（dur）理解：在 block 优化后是成立的**（LDL matrix_dur 更高且周期略慢）。

- 更准确表述应为：
  - **LDL 在 block 配置下并非算得更多，而是部分矩阵路径更“重”（单位时延更高/更依赖 Cube），因此慢一点。**
