# Scalar 最小可用实现报告（ONNXim）

## 1. 目标与设计边界

本次实现目标是为模拟器增加**独立的标量执行单元（Scalar）**，使其与 `Vector/Cube/MTE` 并列，满足以下约束：

- 最小可用：先覆盖 Cholesky no-block 里最关键的标量语义。
- 低侵入：不重构调度框架，仅在现有 pipeline 模型上扩展一条 Scalar pipeline。
- 可观测：在 trace 中新增 `CoreX_Scalar` 事件，统计中新增 scalar 利用率。
- 可配置：复用/对齐现有 latency 配置（如 `scalar_*_latency`、`div_latency`）。

---

## 2. 总体方案

### 2.1 指令层

在全局 opcode 中新增标量指令：

- `SCALAR_ADD`
- `SCALAR_MUL`
- `SCALAR_DIV`
- `SCALAR_SQRT`

作用：把“标量语义”从向量指令中分离出来，为独立 pipeline 分流提供类型依据。

### 2.2 核心调度层

在 Core/SystolicWS 中增加 Scalar pipeline：

- 独立队列（类似 Vector/Cube 的执行队列）
- 独立完成回调（`finish_scalar_pipeline`）
- issue 阶段识别 scalar opcode，进入 Scalar pipeline
- cycle 阶段推进 scalar pipeline，并更新统计

### 2.3 计时模型

新增 `get_scalar_compute_cycles()`：

- `SCALAR_ADD` 使用 `scalar_add_latency`
- `SCALAR_MUL` 使用 `scalar_mul_latency`
- `SCALAR_SQRT` 使用 `scalar_sqrt_latency`
- `SCALAR_DIV` 使用 `div_latency`（与现有除法时延保持一致）

对应源码位置：`src/SystolicWS.cc` 的 `get_scalar_compute_cycles()`。

### 2.4 计时原理（公式化说明）

Scalar 指令在 issue 时会被打上：

- `start_cycle = 当前 core cycle`
- `finish_cycle = start_cycle + scalar_latency(opcode)`

其中：

- `scalar_latency(SCALAR_ADD) = scalar_add_latency`
- `scalar_latency(SCALAR_MUL) = scalar_mul_latency`
- `scalar_latency(SCALAR_SQRT) = scalar_sqrt_latency`
- `scalar_latency(SCALAR_DIV) = div_latency`

执行语义上，当前实现中的 Scalar pipeline 为单发射/单占用（同一时刻仅允许一条 Scalar 指令在执行），因此若有连续标量指令，后续指令会顺延到前一条完成后再发射。

### 2.5 Trace 与统计

- Trace 记录新增 `CoreX_Scalar`
- 统计项新增 scalar 相关占用/计数
- 打印输出可直接看到 Scalar 的使用量

---

## 3. 代码改动清单

- `src/Common.h`
  - 新增 scalar opcode 定义。

- `src/Core.h`
  - 新增 scalar pipeline 队列与状态字段。
  - 声明 `finish_scalar_pipeline`。

- `src/Core.cc`
  - 接入 scalar pipeline 的运行推进。
  - 增加 scalar 统计与打印。
  - 增加 `CoreX_Scalar` trace 事件写出。

- `src/SystolicWS.h`
  - 声明 `get_scalar_compute_cycles()`。

- `src/SystolicWS.cc`
  - 增加 scalar issue 判定与发射路径。
  - 在 `cycle()` 中推进 scalar pipeline。
  - 实现 `get_scalar_compute_cycles()`。

- `src/operations/CholeskyInvNoBlockOp.cc`
  - 将关键“标量性质”步骤映射为 scalar opcode：
    - `POTRF_DIAG_SQRT -> SCALAR_SQRT`
    - `TRSM_DIAG_INV -> SCALAR_DIV`
    - `FWD_DIAG_INV -> SCALAR_DIV`

---

## 4. 验证方法与结果

### 4.1 功能验证

使用 no-block Cholesky ISO trace：

- 输入 trace：`results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv`
- 验证点：存在 `Core0_Scalar` 事件（说明指令已进入独立标量通道）

### 4.2 可视化验证

使用脚本：`scripts/plot_cholesky_core0_timeline_with_scalar.py`

- 输出图：`results/CHOL/falsification/cholesky_noblock_64x16_timeline_iso_scalar.png`
- 结果：时间线含单独 `Scalar` 坐标轴。

### 4.3 标量计时示例（可直接用于报告口径）

下面给出与实现一致的示例推导。设某条指令在 `start_cycle = t` 被发射：

1. `SCALAR_ADD`
  - 周期：`finish = t + scalar_add_latency`
  - 例：若 `scalar_add_latency = 2`，则执行区间为 `[t, t+2)`，耗时 2 cycle。

2. `SCALAR_MUL`
  - 周期：`finish = t + scalar_mul_latency`
  - 例：若 `scalar_mul_latency = 3`，则执行区间为 `[t, t+3)`，耗时 3 cycle。

3. `SCALAR_SQRT`
  - 周期：`finish = t + scalar_sqrt_latency`
  - 例：若 `scalar_sqrt_latency = 8`，则执行区间为 `[t, t+8)`，耗时 8 cycle。

4. `SCALAR_DIV`
  - 周期：`finish = t + div_latency`
  - 例：若 `div_latency = 10`，则执行区间为 `[t, t+10)`，耗时 10 cycle。

串行示例（单 Scalar pipeline）：

- 指令序列：`SCALAR_ADD -> SCALAR_MUL -> SCALAR_DIV`
- 假设 `scalar_add_latency=2, scalar_mul_latency=3, div_latency=10`，且第一条在 `t=100` 发射：
  - `ADD`: `[100, 102)`
  - `MUL`: `[102, 105)`
  - `DIV`: `[105, 115)`
- 总耗时：`2 + 3 + 10 = 15` cycles。

说明：若配置文件中的这些 latency 全设为 `1`，则每条标量指令在模型中均占用 1 cycle；这也是部分默认配置文件中出现的行为。

---

## 5. 与“向量替代标量”的对比结论

- 向量替代标量：实现简单、可用于粗粒度趋势分析。
- 独立 Scalar 单元：
  - 更符合昇腾式“标量/向量/Cube 解耦”的执行语义。
  - 单元占比与瓶颈定位更准确。
  - 对后续微架构优化（指令分流、调度冲突建模）更友好。

因此，本次最小实现优先了**架构可解释性与可观测性**，同时保持改动面可控。

---

## 6. 已知限制与后续建议

### 当前限制

- 目前 scalar 映射主要覆盖 `CholeskyInvNoBlockOp` 的关键路径。
- 其他算子/变体（如 block/chain 版本）尚未系统性迁移。

### 建议下一步

1. 扩展映射到 `CholeskyInvOp` 与 `CholeskyInvChainOp`。
2. 增加“scalar-vector 竞争/互斥”细粒度策略（如果目标硬件需要）。
3. 在对比脚本中默认输出 Scalar 占比，统一报表口径。

---

## 7. 结论

本次已完成“最小可用 Scalar 单元”闭环：

- 指令定义、调度发射、计时模型、trace 观测、统计输出全部打通。
- 在 no-block Cholesky 场景验证有效。
- 为后续全算子一致化和更真实硬件建模提供了稳定基础。
