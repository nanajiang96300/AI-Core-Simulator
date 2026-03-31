# LDL vs Cholesky 时序对比（2026-03-09）

本目录用于对比同配置（24核，`64x16`, `batch=96`）下：
- 新算子：`ldl_test`
- 基线算子：`cholesky_test`

## 目录结构

- `ldl_new/`
  - `trace_c24.csv`：24核全量 trace
  - `trace_core01.csv`：从 24 核中筛选 `Core0/1` 的 trace
  - `timeline_c24.png`：24核全图
  - `timeline_core01.png`：Core0/1 视图

- `cholesky_baseline/`
  - `trace_c24.csv`：24核全量 trace
  - `trace_core01.csv`：从 24 核中筛选 `Core0/1` 的 trace
  - `timeline_c24.png`：24核全图
  - `timeline_core01.png`：Core0/1 视图

## 生成说明

- 运行时均开启 `ONNXIM_MAX_CORE_CYCLES=120000` 防止死锁导致无界运行。
- `core01` 图由 `trace_c24.csv` 过滤 `Unit` 前缀 `Core0_`/`Core1_` 生成。
