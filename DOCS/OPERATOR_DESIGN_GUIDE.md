# ONNXim 算子设计经验与伪并行排查指南

> 基于 Newton–Schulz 逆阵算子与 Ascend 910B (24 核) 仿真实践的总结

本文档用于记录在 ONNXim 中设计算子（`Operation`）时踩过的坑、解决“伪并行”现象的经验，以及关于 tiling / 批处理 / 可视化分析的一些通用建议，方便后续新算子复用和避坑。

相关参考实现：
- `src/operations/NewtonSchulzOp.cc`
- `src/models/NewtonSchulzModel.cc`
- `src/operations/LSEstimatorOp.cc`
- `visualizer_png.py`
- 配置：`configs/ascend_910b_quiet.json`

---

## 1. 背景：Newton–Schulz 算子与 910B 配置

目标场景：
- 算法：Newton–Schulz 矩阵求逆迭代，对 $N\times N$ 方阵 $A$ 进行多轮迭代，输出 $A^{-1}$ 近似。
- 硬件抽象：Ascend 910B 类 NPU，`configs/ascend_910b_quiet.json` 中建模为：
  - 24 个 `systolic_ws` core，每核 16×16 阵列；
  - 向量单元宽度 2048 bit；
  - SPAD / ACCUM_SPAD 约 256 KB；
  - 后端 HBM2 (Ramulator2) 提供高带宽存储。
- 框架：ONNXim 仿真器 `Simulator`，通过 `Model` + `Operation` + `Tile` 的层次描述算子执行；TraceLogger 输出 CSV，用 `visualizer_png.py` 画 pipeline timeline。

Newton–Schulz 实现的关键点：
- `NewtonSchulzOp` 支持 **batch 维度**，默认从输入 tensor 形状 `[B, N, N]` 或 attributes 中推断 `_batch_size`；
- 为每个 batch 生成一个 tile，并通过 `tile->core_id = b % num_cores` 轮询分配 24 核；
- 对每个 tile：
  - `MOVIN A`、`MOVIN X_init`、`MOVIN C` 到 SPAD；
  - 多轮迭代，每轮包含 `GEMM_PRELOAD (A*X)`、`PIPE_BARRIER`、`ADD (C - T)`、`PIPE_BARRIER`、`GEMM_PRELOAD (X*R)`；
  - 最后 `MOVOUT X_out` 写回 DRAM。

---

## 2. 伪并行现象：问题是什么？

在最初的 Newton–Schulz 实验中，pipeline timeline 存在明显“伪并行”特征：
- 24 个 core 都有活动，但时间轴上 **几乎像是串行链**：某个 core 长时间在算，其它 core 的 Load / Compute / Store 明显排在后面。
- 对比 `LSEstimatorOp`（LS 信道估计）的小例子：
  - LS 的 tile 较小，8 个 tile 平均分配到多个 core，Load / Compute / Store 在时间轴上高度重叠，展现出明显的并行性。

经过排查，主要问题不在硬件配置，而在 **Model 结构与 batch / tiling 设计**：

1. **Model-level batch 与 op-level batch 双重 batching**
   - 旧版 `NewtonSchulzModel` 做法：
     - 固定 `batch_size = 32`；
     - 外层循环 `for (b=0; b<batch_size; ++b)`：
       - 为每个 batch 构造一组 2D tensor（`A_b`, `X_init_b`, `C_b`, `X_out_b`）；
       - 为每个 batch 构造一个 `NewtonSchulzOp` 实例（每个 op 内部自己还有 `_batch_size` / tile 的概念）。
     - 最终在 model 中注册 **32 个独立 op**，形成一个包含 32 层的 executable layer 列表。
   - 同时，`NewtonSchulzOp` 自身又支持 batch 维度（`_batch_size`），对每个 batch 生成 tile，这就形成了 **model 层级 + op 内部的双重 batch**。

2. **Scheduler 的 layer 语义导致近串行**
   - ONNXim 的 scheduler （simple scheduler）行为是：
     - 一次只取一个 Operation 作为当前 `layer`；
     - 把这个 op 的所有 tiles 分发到各个 core 跑完；
     - 然后再切下一个 op / layer。
   - 在“双重 batching”的旧设计中：
     - 对于 32 个 op，scheduler 会 **依次** 把 32 个 layer 跑完；
     - 每个 layer 又在内部按 `_batch_size` 生成一批 tiles；
     - 在 timeline 上表现为：**多个 batch 先在某个 core 上跑完，再切到下一个 op 的 batch**，核间几乎串行，DRAM 也在为某个大 tile 集中服务。

3. **Tile 粒度过粗，导致 DRAM 排队严重**
   - Newton–Schulz 每个 tile 目前仍是完整的 $N\times N$ 矩阵：
     - 每个 tile 进入 core 就要搬完整的 A/X/C/X_out；
     - 迭代次数较多，每轮 GEMM / ADD 都对整块矩阵操作；
   - 这样的 tile 对 DRAM 来说是“大颗粒突发负载”，多核并发访问更容易在同一 channel 上排队，进一步放大“串行化”的感觉。

总结：
> **伪并行的根源不在硬件配置，而在 op / model 的结构设计：双重 batching + 过粗 tile + scheduler 的逐层调度语义，共同导致了核间接近串行的执行模式。**

---

## 3. 关键修正：单 op batched 结构

### 3.1 设计目标

在保持 Newton–Schulz **算法语义完全正确** 的前提下：
- 使用 `batch_size = 96` 的测试场景，让每个核大约处理 \(96 / 24 = 4\) 个 batch；
- 避免 model-level 与 op-level 的双重 batching；
- 让 scheduler 只看到一个 op / layer，把所有 batch tiles 一次性分发到 24 核。

### 3.2 `NewtonSchulzModel` 重构要点

新版本 `NewtonSchulzModel::initialize_model()` 的做法：

1. **只构建一个 Model-level op**
   - 读取 JSON 中的：
     - `batch_size`（例如 96）；
     - `attributes.iterations`（例如 10）。
   - 固定矩阵维度 `N = 32`（对应 `newton_schulz_32x32`）。
   - 构造 4 个 3D tensor：
     - 形状均为 `[B, N, N]`，即 `{batch_size, N, N}`；
     - 分别对应 `A`, `X_init`, `C`, `X_out`。

2. **batch 维留给 op 内部管理**
   - 在 `NewtonSchulzOp::infer_shapes_from_model()` 内，从输入 tensor dims 推断：
     - `_batch_size = B`；
     - `_matrix_shape = {N, N}`。
   - `initialize_tiles()` 对每个 batch 生成一个 tile，并按 RR 分配到 24 核：每核约 4 个 tile。

3. **保证 scheduler 只看到一个 layer**
   - model 只注册一个 op（`newton_schulz_32x32.NewtonSchulz`），并把它加入 `_executable_layer`；
   - scheduler 启动时：
     - 这个唯一的 layer 被一次性激活；
     - 所有 96 个 batch tiles 同时进入 24 核的 tile 队列，调度和执行才真正体现出多核并行。

实际效果（batch=96, 910B 配置）：
- log 中显示：`Dispatched 96 batches across 24 cores. Load Distribution: Core0:4, Core1:4, ...`；
- 各个 core 的 Cube 利用率约 86–89%，DRAM channel 平均带宽利用率约 12%；
- pipeline 图中 24 个 core 的 Load/Compute/Store 呈现出 **对称且高度重叠** 的形态，不再是明显的长链式串行。

---

## 4. Watchdog：避免仿真“空转不结束”

在复杂算子和新 tiling 方案开发时，经常会出现“utilization 几乎为 0，但仿真不退出”的情况。为防止每次手动 `Ctrl+C`，在 `Simulator::cycle()` 中增加了 watchdog：

- 环境变量 `ONNXIM_MAX_CORE_CYCLES`：
  - 若未设置或为 0，则 watchdog 关闭；
  - 若设置为正整数 `K`，当 `_core_cycles >= K` 时：
    - 打印错误信息："Simulation aborted: reached ONNXIM_MAX_CORE_CYCLES limit"；
    - 跳出 `while (running())` 循环；
    - 打印已有统计信息并正常收尾。

推荐开发期设置示例：

```bash
# 小规模快速实验
ONNXIM_MAX_CORE_CYCLES=100000 \
ONNXIM_TRACE_CSV=results/tmp.csv \
./build/bin/Simulator --config configs/ascend_910b_quiet.json \
  --models_list example/newton_schulz_test.json \
  --mode newton_schulz_test --log_level info
```

这一机制与算子设计没有直接关系，但在调 tiling / 指令生成时非常重要，可以快速发现死锁或进展停滞的问题。

---

## 5. Tiling 设计经验与建议

### 5.1 算法正确性 vs 架构友好

设计新算子时，通常有两种取向：

1. **算法优先（本次 Newton–Schulz 采用的方案）**
   - 保持数学语义严格等价：例如始终对完整矩阵做精确迭代求逆；
   - tiling 仅用于并行度与数据搬运优化，不改变计算公式。

2. **架构优先**
   - 允许引入近似或重构算法（如 block-wise 近似、低秩分解等），换取更好的局部性和并行性；
   - 适用于性能/能效关注高于数值精度要求的场景。

实际开发中建议：
- 明确当前实验是“精度基线”还是“体系结构优化”阶段；
- 在算法优先阶段，**不要匆忙做行块近似等改变语义的 tiling**，先把正确版本的 batch / 并行性理顺；
- 在有了可靠基线后，再尝试 block 算法或稀疏近似。

### 5.2 Tiling 粒度与多核调度

以 `LSEstimatorOp` 为对比：
- A 为 32×32, B/C 为 32×512；
- 沿行方向按 4 行切为 8 个 tile；
- 每个 tile：
  - Load 4 行 A 和全量 B；
  - 做一次 GEMM；
  - Store 对应 4 行 C；
- Round-Robin 分配到多个 core，使得在时间轴上：
  - 多个 core 的 Load / Compute / Store 高度重叠；
  - DRAM 请求较小且交错，带宽利用更平滑。

结合 Newton–Schulz 的经验，可以提炼以下 tiling 建议：

1. **避免过粗的 tile 粒度**
   - “一个 tile = 整个矩阵 + 多次迭代” 往往会：
     - 导致 DRAM 大块搬运，增加排队；
     - 让 scheduler 很难在 op 层面插入其它工作；
   - 更建议：
     - 先在 batch 维度上分散（如本次 96 batch, 每核 4 个 batch）；
     - 如需进一步优化，可考虑在矩阵空间维度（行/列）上再细分。

2. **tiling 维度要与并行资源对齐**
   - 一般希望：`tile_count ≫ core_count`，这样 scheduler 可以更灵活地填充空隙；
   - 对 24 核系统，96 个 batch/tile 是一个比较合理的起点（每核 4 个 tile），再往上加 batch 能进一步平滑 load。

3. **Tile 生命周期内要保证流水线完整**
   - 每个 tile 内部通常遵循：
     - `MOVIN` → `GEMM/Vector` → `MOVOUT`；
   - 建议在指令序列里使用 `last_inst=true`（或类似标记）来明确 tile 结束时机，避免 core 长时间持有“已完成但未释放”的 tile；
   - 对 Newton–Schulz 这种多轮迭代的算子，**尽量让每个 tile 的 Load / Compute / Store 在时间上尽可能靠近**，避免“一口气算完所有迭代再统一 Store”带来的尾部写出高峰。

4. **合理使用 PIPE_BARRIER**
   - `PIPE_BARRIER` 会在 Cube / Vector / MTE 之间插入同步点；
   - 仅在确实需要等待某个 pipeline 完成结果时使用，避免过多 barrier 造成流水线气泡；
   - 在 Newton–Schulz 中：
     - GEMM 完成后需要 barrier 才能进行 ADD；
     - ADD 完成后需要 barrier 才能进行下一次 GEMM；
     - 这些 barrier 都属于算法数据依赖必需的同步点。

5. **tiling 与 DRAM 布局的配合**
   - 为每个 batch 计算 DRAM `batch_offset` 是常见模式：
     - `offset = batch_id * matrix_size_bytes`；
     - A / X_init / X_out 用带 offset 的地址；C 作为共享常量可以不偏移；
   - 对 block tiling，需要保证：
     - 各 tile 所需的数据块在 DRAM 中尽量连续；
     - 不同 tile 尽量映射到不同 bank/row，减少 row conflict；
   - 可通过查看 Ramulator 输出中的 row hits/misses/conflicts 来评估布局质量。

---

## 6. Timeline 可视化与“伪因果”的正确理解

使用 `visualizer_png.py` 渲染 CSV 时，要注意：

- 每条横线是 **某个 core 上的某个单元(unit)**：
  - `CoreX_Cube`：矩阵乘阵列；
  - `CoreX_Vector`：向量单元；
  - `CoreX_MTE2`：Load；
  - `CoreX_MTE3`：Store；
- Timeline 显示的是 **该单元上所有 tile 的活动叠加**，而不是单个 tile 的线性生命周期。

例如，在 96 batch Newton–Schulz 中：
- Core1 的最后一条 Cube event：`NS_X: start=3010, end=3072`；
- Core1 的最后一批 Store：`Store: start=3073, end=3082`；
- 在图上可能会看到“前面已经有一段 Store 块，然后还有 Cube 在继续算”，那往往是：
  - 对**已完成的某些 tile** 做的 Store；
  - 随后 Cube 继续为**剩余 tile** 计算；
- 若从 CSV 按 tile id 追踪，可以验证：**单个 tile 的顺序始终是 Compute → Store，没有真正的“回滚依赖”。**

因此，在解读 timeline 时要特别注意：
- 不要把“unit-level 的叠加视图”误认为“per-tile 的串行视图”；
- 如果怀疑存在因果顺序错误，可以：
  - 过滤某个 core + 某个 tile id，看该 tile 的所有事件顺序；
  - 或者只看 `name` 字段中带某个 batch 标记的记录。

---

## 7. 未来可优化的方向

在当前“单 op + batch=96 + 粗粒度 tile”正确运行的基础上，可以进一步探索：

1. **更细粒度的空间 tiling（保持算法正确性前提下）**
   - 调研块状 Newton–Schulz / block-wise matrix inversion 算法；
   - 以 block 为基本单元做 tiling，让每个 tile 只处理矩阵的一部分行/列；
   - 通过额外的拼装 / Schur complement，保持最终结果仍是整个矩阵的精确逆或可控近似。

2. **更智能的多核调度**
   - 目前主要依靠 RR 分配 batch 到 core；
   - 可尝试：
     - 基于 tile 大小 / 估算执行时间的 load-balance 调度；
     - 在 DRAM 多通道场景下，根据地址分布调整 tile → core 映射，减少 bank 冲突。

3. **自动 tiling / 映射工具链**
   - 类似 `MyNewOperator` 中展示的 `plan_tiling()` / `initialize_tiles()` 模式；
   - 将 tiling 策略与具体算子实现解耦，便于在不改算子数学公式的情况下快速切换 tiling 方案。

4. **更系统的 profiling 与指标收集**
   - 目前关注的指标：
     - 每核 MatMul / Vector active cycles；
     - DRAM 带宽利用率；
     - 总 cycles 与 tile TPS；
   - 未来可以增加：
     - per-tile 的完成时间分布；
     - per-core 队列深度 / stall 次数；
     - 更清晰的前端/后端 overlap 统计。

---

## 8. 设计新算子时的检查清单（Checklist）

设计 / 实现一个新 `Operation` 时，建议按以下 checklist 自查：

1. **Batch 管理**
   - [ ] 明确 batch 维度由 **Model 管还是 op 管**，避免 model-level 与 op-level 双重 batching；
   - [ ] 如果是 C++ 专用测试 `Model`，更推荐：
     - Model 只构建一个 batched op；
     - 输入/输出 tensor 采用 3D/4D 形式（例如 `[B, N, N]`），batch 逻辑集中在 op 内。

2. **Tiling 设计**
   - [ ] 计算 tile 数量与核心数量的比例，保证 tile 数量显著多于 core 数（例如 `tile_count ≥ 4 × num_cores`）；
   - [ ] 为每个 tile 设计清晰的 Load / Compute / Store 流水线，并在最后一条 Store 指令处标记 tile 完成；
   - [ ] 评估每个 tile 的数据量是否与 SPAD / 带宽匹配，避免过大 tile 造成 DRAM 拥塞；
   - [ ] 根据硬件拓扑（core 数、DRAM 通道数）设计合理的 `tile->core_id` 映射（RR、分区映射等）。

3. **指令序列与同步**
   - [ ] 检查 `PIPE_BARRIER` 的位置是否与数据依赖严格对应，避免多余 barrier；
   - [ ] 确保 MOVIN / GEMM / Vector / MOVOUT 等指令的地址和大小与 tensor / tiling 一致；
   - [ ] 检查 batch_offset / block_offset 计算是否正确，特别是广播常量与 per-batch 数据的区分。

4. **仿真与可视化**
   - [ ] 使用小 batch（如 1 或 2）做功能验证，结合日志确认结果正确；
   - [ ] 打开 `ONNXIM_MAX_CORE_CYCLES` watchdog，在开发期避免死锁；
   - [ ] 用 `ONNXIM_TRACE_CSV` + `visualizer_png.py` 观察：
     - 核间并行度是否符合预期；
     - 是否存在明显的 DRAM 写出/读入尖峰；
     - 是否出现“所有 core 长时间 idle”但循环未结束的异常情况。

5. **性能与可扩展性**
   - [ ] 在小规模正确的前提下，逐步增大 batch / 矩阵维度，观察总 cycles 和带宽利用率走势；
   - [ ] 检查 tile TPS（tile per second）指标是否随 batch 增加保持稳定或提升；
   - [ ] 对比不同 tiling / 调度策略的 timeline，优先选择具有更好 overlap 和更平滑带宽利用的方案。

---

通过以上经验和检查清单，希望后续在 ONNXim 中设计算子时：
- 能尽量避免双重 batching / 伪并行等结构性问题；
- 在保持算法正确性的前提下，更系统地探索 tiling 与多核调度空间；
- 让新的算子在 910B 等目标配置上从一开始就具备可解释、可调优的执行行为。 
