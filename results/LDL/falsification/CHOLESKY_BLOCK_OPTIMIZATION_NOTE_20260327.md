# Cholesky Block(2x2) 优化说明（2026-03-27）

## 1) 问题确认

在 `results/CHOL/falsification/cholesky_block_64x16_trace.csv` 中，2x2 微更新大量落在 Cube：

- `CHOL_POTRF_DIAG_UPD`: `Cube=2688`
- `CHOL_TRSM_NUM_UPD`: `Cube=5376`
- `CHOL_RK_UPDATE`: `Cube=8064`
- `CHOL_FWD_OFF_MAC`: `Cube=2688`

导致整体：
- `Cube events=19008`
- `Wait events=16166`
- `max_end_cycle=13697`

## 2) 修复

修改 `src/operations/CholeskyInvOp.cc`：
- 新增 `pick_chol_step_mul_opcode(...)`
- 对 `blk<=2` 的微更新优先使用 `Opcode::MAC`（Vector）而非 `Opcode::GEMM_PRELOAD`（Cube）
- 保留 `GRAM`/`BWD_MAC_FULL` 等大矩阵阶段的 Cube 使用

## 3) 优化后结果

优化后 trace：`results/CHOL/falsification/cholesky_block_64x16_trace_opt.csv`

- `max_end_cycle`: `13697 -> 3404`（`4.02x` 提升）
- `event_count`: `48614 -> 32637`
- `total_duration`: `3185924 -> 275652`（`11.56x` 减少）
- `Cube events`: `19008 -> 192`
- `Wait events`: `16166 -> 189`

并且上述四类 2x2 微更新全部转为 Vector：
- `CHOL_POTRF_DIAG_UPD`: `Vector=2688`
- `CHOL_TRSM_NUM_UPD`: `Vector=5376`
- `CHOL_RK_UPDATE`: `Vector=8064`
- `CHOL_FWD_OFF_MAC`: `Vector=2688`

## 4) 与 LDL block 对比（同配置）

- LDL block `max_end_cycle=3704`
- Cholesky block(opt) `max_end_cycle=3404`
- 比值（CHOL/LDL）=`0.919`

Core0：
- LDL `3600`
- CHOL(opt) `3396`
- 比值（CHOL/LDL）=`0.943`

对应图/表：
- 图：`results/LDL/falsification/ldl_cholesky_block_core0_timeline_opt.png`
- 表：`results/LDL/falsification/LDL_CHOLESKY_BLOCK_CORE0_COMPARE_OPT_20260327.md`
