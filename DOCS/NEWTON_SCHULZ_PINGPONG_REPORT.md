# Newton–Schulz 32×32 Ping‑Pong Double Buffering Analysis

本报告总结在 Ascend 910B 配置下，对 32×32 Newton–Schulz 矩阵求逆算子加入跨 batch 的 ping‑pong 双缓冲之后，为什么**系统总周期明显变长、计算不再紧密**。报告对比了加入双缓冲前后的时序图，并从流水线结构和瓶颈角度解释性能下滑的根本原因。

> 环境：
> - 硬件配置：`configs/ascend_910b_quiet.json`
> - 模型配置：`example/newton_schulz_opt_test.json`（batch_size=96, M=32, K=32, iterations=10）
> - Baseline：`NewtonSchulzOp` / `NewtonSchulzOptOp` 按 batch 切 tile（1 tile = 1 batch）
> - Ping‑pong 版本：`NewtonSchulzOptOp` per‑core super‑tile + 跨 batch 双缓冲

## 1. 实验对象与前后对比

### 1.1 Baseline：按 batch 切 tile 的多步 Newton–Schulz

- 算子：`NewtonSchulzOp`（以及结构上等价的 `NewtonSchulzOptOp` v4）。
- Tiling / 调度：
  - 96 个 batch，每个 batch 对应 1 个 tile，总共 96 tiles。
  - 24 个核心，round‑robin 分配（每核约 4 个 tile）。
  - 每个 tile 在单个 core 上按序执行：
    - `Load(A, X, C)` → barrier → `[T = A·X, R = C − T, X = X·R] × 10` → `Store(X_10)`。
  - 不在同一 tile 内做跨 batch 的 overlap，调度器可以跨 tile 全局交错执行。
- 性能（多步迭代、10 次 NS）：
  - 总周期：约 **3159 cycles**（baseline 32×32, 10 iterations）。
  - Systolic Array 利用率：约 **86–88%**。
  - HBM2 带宽利用：约 **12–14%**，明显非带宽瓶颈。
- 时序图引用：
  - CSV：`results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32.csv`
  - PNG：`results/newton_schulz/910b/pipeline_newton_schulz_910b_32x32.png`

这一版本的 Cube 行呈现为**相对紧凑的绿色长条**，说明 10 次 NS 迭代在各 core 上被紧密排布，空隙较少，整体是 compute‑bound 但 pipeline 充分。

### 1.2 Ping‑pong 双缓冲版本：per‑core super‑tile + 跨 batch overlap

- 算子：`NewtonSchulzOptOp` v5（带 ping‑pong 双缓冲）。
- 关键结构变化：
  1. **Per‑core super‑tile**：
     - 不再按 batch 建 tile，而是每个 core 只建 1 个 tile。
     - 该 tile 通过 `local_batches = { b | b % num_cores == core_id }` 管理自己负责的所有 batch（约 4 个）。
  2. **SRAM 布局（per‑batch A/X + 共享 C/R + ping‑pong T）**：
     - 对每个本地 batch `li` 分配独立的 `[A_li, X_li]` SPAD 区域，避免 MOVIN 目标地址复用导致的 “Destination allocated” 报错。
     - C（2I）只在每个 core 上加载一次，放在共享的 `addr_C`。
     - R（残差）使用共享 `addr_R`。
     - ACCUM 中为 T 分配 **两个 buffer**：`T_Ping` 与 `T_Pong`，用于真正意义上的 ping‑pong 累加器：
       - 迭代内部始终在当前 `cur_T` 上做 `T = A·X` 和 `X_new = X·R`。
       - 不同 batch 之间交替使用 `T_Ping` / `T_Pong`，使得 batch i 的 `Store(cur_T)` 可以和 batch i+1 的 `Compute(next_T)` 并行。
  3. **跨 batch 的 Load/Compute/Store overlap**：
     - Prologue：对 `local_batches[0]` 加载 `A_0, X_0` 和共享 `C`，然后 barrier（type=1）。
     - 主循环对每个本地 batch 索引 `li`：
       - 若存在 `li+1`：
         - 提前对 `A_{li+1}, X_{li+1}` 发起 MOVIN（MTE 背景加载），目标为该 batch 独立的 `[A_{li+1}, X_{li+1}]` 区。
       - 在当前 `cur_T`（Ping 或 Pong）上执行 10 次 NS 迭代：
         - `T = A_li · X_k` → barrier(type=2) → `R = C − T` → barrier(type=3) → `X_{k+1} = X_k · R`（写回 `cur_T`）。
       - 若存在 `li+1`：在进入下一个 batch 迭代前发一个 barrier(type=1)，确保 `A_{li+1}, X_{li+1}` 已加载完毕。
       - 对 batch `li` 的 `cur_T` 触发 MOVOUT，把结果写回 DRAM；由于下一 batch 将切换到另一块 T（Ping/Pong），`Store(i)` 可以在后台与 `Compute(i+1)` 重叠。
- 性能（10 次 NS，无改迭代数）：
  - `Layer ... finish at 7934`，`Simulation Finished at **7953 cycle**`（与之前 v5 ~7956 cycles 基本一致）。
  - Systolic Array 利用率：约 **62.37%**。
  - Vector 单元利用率：约 **3.07%**。
  - HBM2 带宽利用率：约 **3%**（216 reads, 96 writes）。
- 时序图引用：
  - CSV：`results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32_opt.csv`
  - PNG：`results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32_opt.png`

从新时序图可以看到：每个 core 的 CubeCore 行比 baseline 明显**更长、更稀疏**，绿色块之间有更多空档，而且 24 个 core 的带状分布整体被拉长，导致总周期从 ~3159 cycle 增长到 ~7953 cycle，几乎 **慢了 2.5×**。

## 2. 为什么 ping‑pong 双缓冲导致性能下滑？

表面上看，引入跨 batch ping‑pong 双缓冲应该能把 `Load(i+1)` 和 `Store(i)` 隐藏在 `Compute(i)` 后面，从而缩短总执行时间。但在这个算子和规模下，实际情况恰好相反：

> **根本原因：核算子本身是强 compute‑bound 的，Load/Store 并不是关键路径；而为实现跨 batch overlap 引入的 super‑tile、额外 barrier 和更长的依赖链，反而拉长了关键路径，使计算不再紧密。**

### 2.1 Per‑core super‑tile 导致跨 batch 串行化

- Baseline：
  - 每个 batch 是独立 tile，调度器可以在 24 核之间灵活交错不同 batch 的执行顺序。
  - 即使某个 tile 出现短暂 stall，其他 tile 仍可在同一 core 或别的 core 上继续推进，整体时间由“全局最优排布”决定。
- Ping‑pong 版本：
  - 每个 core 只存在 **1 个 super‑tile**，内部用 for‑loop 顺序处理 `local_batches`。
  - 这意味着：
    - 该 core 的所有 batch 在时间上被**硬性串行化**，中间插不进别的 tile。
    - 全球调度的自由度下降，整个 core 更像一个“长事务”，无法像 baseline 那样让短任务和长任务穿插。
- 结果：
  - 虽然我们在 super‑tile 内实现了 `Load(i+1) || Compute(i) || Store(i−1)` 这种交叠，但由于所有 batch 都被锁在单一 tile 内，**关键路径被 super‑tile 的长度主导**，总周期被显著拉长。

### 2.2 额外 barrier 和更复杂的依赖链稀释了计算密度

- 为保证跨 batch 的正确性，ping‑pong 版本在原有迭代内部 barrier（Cube→Vector, Vector→Cube）之外，又加入了：
  - Prologue 后的 `PIPE_BARRIER(type=1)`，确保第一个 batch 的 A/X/C 均加载完成；
  - 每个 batch 结束前的 `PIPE_BARRIER(type=1)`，保证下一 batch 的 A/X 已全部到位。
- 这些 barrier 会：
  - 在时间轴上插入额外的同步点，使得所有相关单元必须等待最慢的那条路径；
  - 放大不同单位间的轻微抖动，使 Cube 无法持续“紧凑地”发射 GEMM 指令。
- 反映在时序图上：
  - Baseline 的 Cube 行呈现为**较长的连续绿带**，间隙很小；
  - Ping‑pong 图中，绿带被更多的空白和短 block 分割，说明 barrier 带来的同步停顿显著增加，**计算密度下降**，最终体现在 Cube 利用率从 ~86–88% 降到 ~62.37%。

### 2.3 Load/Store 本身不是瓶颈，overlap 收益极小

- 两个版本的 HBM2 带宽利用率对比：
  - Baseline：约 **12–14%**，读取次数 288、写入 96（不同实现略有差异）。
  - Ping‑pong：约 **3%**，读取 216、写入 96。
- 关键点：
  - 即使在 baseline 中，HBM 带宽也远未打满，说明 kernel 主要是 **算子计算（GEMM）主导的 compute‑bound**，而非 memory‑bound。
  - 在这种场景下，“隐藏”Load/Store 的潜在节约在总时间中占比极小；
  - 反而为实现 overlap 引入 super‑tile + 多级 barrier 的结构开销要大得多，得不偿失。
- 换句话说：

  $$ T_{total} \approx T_{compute} + T_{sync} + \text{(少量 load/store)} $$

  baseline 下 $T_{sync}$ 较小，而 ping‑pong 版本为了追求更小的 $T_{load/store}$，却把 $T_{sync}$ 和整体依赖长度大幅增加，最终 $T_{total}$ 显著上升。

### 2.4 Per‑batch A/X 槽位增加了 SRAM 足迹但没有减少 FLOPs

- 为了解决“Destination allocated” 的 SPAD 冲突，ping‑pong 版本为每个本地 batch 分配了独立的 `[A_li, X_li]` 槽位：
  - 这在资源上是可行的（SRAM 容量足够），但并没有减少任何计算量；
  - 每个 batch 仍然执行完整的 10 次 NS 迭代，FLOPs 完全一致。
- 额外开销：
  - 更多的地址计算与更大的 SPAD 占用；
  - 在 trace 中表现为更多的 MOVIN 指令和更复杂的地址映射逻辑，但这些并不会被 MTE→Cube overlap 所完全掩盖。

综合来看：ping‑pong 双缓冲在**逻辑结构上是正确且“更聪明”的加载/存储策略**，但在当前 32×32、10 次 NS、batch=96 的设定下，

- Load/Store 原本就不是关键路径；
- 新增的 super‑tile 和 barrier 使关键路径被一个更长且更难被调度器拆开的“长事务”主导；
- 计算密度下降，Cube 利用率从 ~86–88% 掉到 ~62.37%，总周期从 ~3159 cycle 上升到 ~7953 cycle。

## 3. 对论文撰写的启示

从体系结构/映射设计角度，这个对比可以在论文中支持如下结论：

1. **双缓冲并非总是收益**：
   - 在 compute‑bound kernel 上，贸然增加跨 batch ping‑pong 流水线，反而可能增加同步和调度复杂度，拉长关键路径。
2. **关键在于识别真实瓶颈**：
   - 若 baseline 已经是 compute‑bound，优先级应放在：
     - 提高阵列利用率（减少 barrier、减少 super‑tile 串行化）、
     - 或降低迭代次数 / 改善初值使收敛加快，
     而不是先做复杂的 Load/Store overlap。
3. **时序图是重要证据**：
   - 建议在论文中并排展示：
     - Baseline：`pipeline_newton_schulz_910b_32x32.png`；
     - Ping‑pong：`profiling_log_newton_schulz_910b_32x32_opt.png`；
   - 并标注关键指标：总周期、Cube 利用率、HBM 带宽利用率。

## 4. 未来可能的优化方向

若希望在保持 10 次迭代前提下进一步优化，可以考虑：

- 回到 per‑batch tiling（避免 super‑tile 串行化），只进行**更局部、更轻量的 overlap**：例如仅在单个 batch 内 overlap 局部的 Load 和 GEMM，而不是跨 batch；
- 减少不必要的 barrier，精简为“足够保证数据依赖”的最小集合，观察 Cube 行是否重新变得紧凑；
- 在算法层面，通过更好的初值或预条件，使得在 8 次迭代就能达到与 10 次迭代相近的精度，从根本上减少 compute 成本。

以上分析基于当前的 profiling CSV 与时序图，后续如有新的变体（例如改回 per‑batch tile 但保留局部 ping‑pong），可以继续在本文件中追加对比章节。