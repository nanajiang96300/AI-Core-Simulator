# ONNXim Newton–Schulz 与 MMSE 实验总结

本文档总结了本轮对话中围绕 ONNXim 框架所做的主要工作，包括：

- 如何在模拟器中创建与实现一个新算子；
- 如何从 CSV trace 中提取算子性能指标；
- Newton–Schulz 和 MMSE 算子的优化、缩放分析与可视化方法；
- 如何用脚本自动化 sweep / 分析；
- 实践中踩过的坑与经验教训。

---

## 1. 在 ONNXim 中创建 / 修改算子

### 1.1 算子类结构

- 所有算子都继承自 `Operation`，典型样例：
  - `NewtonSchulzOp`、`NewtonSchulzOptOp`
  - `MMSEOp`
- 关键成员函数：
  - `parse_attributes()`  
    从 `_attributes` 中解析 JSON / ONNX 属性，例如：
    - `iterations`（Newton–Schulz 迭代次数）
    - `batch_size`
    - 其他算子相关参数。
  - `infer_shapes_from_model()`  
    从 `_model` 与输入 `Tensor` 的 dims 推断：
    - `_matrix_shape`（矩阵维度）
    - `_batch_size`（当输入是 `[Batch, N, K]` 三维时，从 `dims[0]` 覆盖）。
  - `initialize_tiles(MappingTable&)`  
    决定生成多少 `Tile`（通常为 `batch_size` 个），以及如何映射到物理核心（负载均衡策略）。
  - `initialize_instructions(Tile* tile, Mapping mapping)`  
    为每个 tile 生成一串 `Instruction`：
    - `MOVIN` / `MOVOUT`
    - `GEMM_PRELOAD` / `GEMM`
    - 向量指令（`ADD` 等）
    - 人为 Barrier（`PIPE_BARRIER`）等。

### 1.2 批处理和多核映射（以 `NewtonSchulzOp` 为例）

- 在 `parse_attributes()` 中支持：
  - `iterations`（默认 10）
  - `batch_size`（默认宏 `DEFAULT_BATCH_SIZE=96`）
- 在 `infer_shapes_from_model()` 中：
  - 若输入 tensor 维度为 `[Batch, N, K]`：
    - `_batch_size = dims[0];`
    - `_matrix_shape = {dims[1], dims[2]};`
- 在 `initialize_tiles()` 中为每个 batch 生成 1 个 tile，并用 round-robin 分配核心：
  ```cpp
  for (uint32_t b = 0; b < _batch_size; ++b) {
      uint32_t assigned_core = b % _config.num_cores;
      tile->batch = b;
      tile->core_id = static_cast<int>(assigned_core);
      ...
  }
  ```
- 通过日志输出负载分布（便于检查）：
  ```cpp
  spdlog::info("NewtonSchulzOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  ```

### 1.3 指令编程模式与内存寻址

- 基础类型与宏：
  - 定义在 `Common.h`：
    - `typedef uint64_t addr_type;`
    - `typedef uint64_t cycle_type;`
    - `#define SPAD_BASE 0x10000000`
    - `#define ACCUM_SPAD_BASE 0x20000000`
- `Instruction` 结构体（`Common.h`）关键字段：
  - `opcode`：`Opcode::MOVIN`, `MOVOUT`, `GEMM_PRELOAD`, `ADD`, `PIPE_BARRIER` 等。
  - `dest_addr`, `src_addrs`, `size`, `compute_size`。
  - `tile_m`, `tile_k`, `tile_n` 用于延迟模型。
  - `src_from_accum`：是否从 ACCUM SRAM 读取。
  - `operand_id`, `base_addr` 等元信息。
- Newton–Schulz 每轮迭代典型指令序列：
  1. `GEMM_PRELOAD("NS_T", dest=addr_T, src={A, X_src})` → `T = A·X`
  2. `PIPE_BARRIER("NS_BARRIER_CUBE2VEC")`
  3. `ADD("NS_R", dest=addr_R, src={C, T})` → `R = C - T`（`src_from_accum = true`）
  4. `PIPE_BARRIER("NS_BARRIER_VEC2CUBE")`
  5. `GEMM_PRELOAD("NS_X", dest=addr_T, src={X_src, R})` → `X_{k+1} = X_k·R`

### 1.4 关键数据路径 Bug 与修复

- 问题：后续迭代仍然从初始 `X_0` 读，而没有从 ACCUM 中读 `X_k`。
- 修复方案：在 `initialize_instructions()` 中区分首轮和后续轮次：
  ```cpp
  addr_type x_src_for_AX = (iter == 0) ? addr_X : addr_T;
  addr_type x_src_for_XR = (iter == 0) ? addr_X : addr_T;
  bool use_accum_for_x = (iter > 0);

  // GEMM_PRELOAD for T = A * X_k
  .src_addrs = {addr_A, x_src_for_AX},
  .src_from_accum = use_accum_for_x;

  // GEMM_PRELOAD for X_{k+1} = X_k * R
  .src_addrs = {x_src_for_XR, addr_R},
  .src_from_accum = use_accum_for_x;
  ```
- 这样保证：
  - 第 0 次迭代用 DRAM 读入的 `X_0`（SPAD）。
  - 后续迭代始终用 ACCUM 中更新后的 `X_k`。

### 1.5 指令执行与延迟模型所在位置

- `Core.cc`：
  - 从每个 tile 的 `instructions` 取出当前 `Instruction`，并根据 `opcode` 放入：
    - `_ld_inst_queue`（`MOVIN`）
    - `_st_inst_queue`（`MOVOUT` / `MOVOUT_POOL`）
    - `_ex_inst_queue`（包括 GEMM / 向量 / PIPE_BARRIER 等）。
- `SystolicWS.cc`：
  - 矩阵指令（Cube）：
    - 若 `inst->opcode` 是 `GEMM` 或 `GEMM_PRELOAD`，则进入 `_compute_pipeline`：
      - `start_cycle` 由前一条 GEMM 的结束时间和 core 高度等因素决定。
      - `finish_cycle = start_cycle + get_inst_compute_cycles(inst)`.
  - 向量指令（Vector）：
    - 在 `get_vector_compute_cycles()` 中根据不同 opcode 使用不同 latency：
      - 例如：`case Opcode::ADD: return vec_op_iter * add_latency;`
      - `case Opcode::PIPE_BARRIER: return 1;`（人为 barrier，以 1 cycle 占用 Vector pipeline，使 trace 可见）。

---

## 2. 从 CSV 中读取算子性能

### 2.1 生成 Trace 的方法

- 运行模拟器前设置：
  ```bash
  export ONNXIM_MAX_CORE_CYCLES=800000
  export ONNXIM_TRACE_CSV="results/newton_schulz/910b/profiling_log_newton_910b_batch96.csv"
  ```
- 运行示例（以 Newton–Schulz 为例）：
  ```bash
  cd /project/ONNXim
  ./build/bin/Simulator \
    --config configs/ascend_910b_quiet.json \
    --model example/newton_schulz_32x32.json \
    --mode newton_schulz_test
  ```

### 2.2 CSV 格式

- 统一列名：
  - `name`：阶段名称，例如 `Load`, `NS_T`, `NS_R`, `NS_X`, `Store`；或 `MMSE_HtH`, `MMSE_INV_X`, `MMSE_WH` 等。
  - `unit`：如 `Core0_CUBE`, `Core3_VEC`, `Core5_MTE2`。
  - `start_cycle`, `end_cycle`：在该 unit 上的起止周期。

### 2.3 分类与 union-of-intervals 聚合

- 在 `scripts/analyze_newton_schulz_scaling.py` 与 `scripts/analyze_mmse_scaling.py` 中的通用套路：
  1. 用 pandas 读取 CSV。
  2. 根据 `name` 或 `unit` 关键字把事件映射到四类：
     - MOVIN：`name == "Load"` 或 `unit` 中含 `MTE2`。
     - MOVOUT：`name == "Store"` 或 `unit` 中含 `MTE3`。
     - Cube：`unit` 中含 `CUBE` 或 `opcode == GEMM/GEMM_PRELOAD` 对应的事件。
     - Vector：`unit` 中含 `VEC` 或 各种 `NS_*` / `MMSE_*` 向量阶段。
  3. 针对每一类：
     - 收集所有 core 上该类事件的 `[start_cycle, end_cycle)` 区间。
     - 做区间并集（union-of-intervals），得到「至少有一个核心在该类上忙碌」的总时间。
  4. 计算：
     - MOVIN / MOVOUT / Cube / Vector 的活跃周期。
     - 所有事件的并集，得到 Total 周期。

### 2.4 时间线可视化

- 单算子 per-core pipeline：
  - `visualizer_png.py`：从 CSV 生成按 core 分行的 Gantt 图。
- 两个配置的 overlay：
  - `scripts/plot_mmse_timeline_overlay.py`：
    - 输入两份 summary CSV（如 256×32 和 512×32）。
    - 按 `Name` 聚合，跨 core 做 union-of-intervals。
    - 每个阶段一行，蓝色代表配置 A，橙色代表配置 B。

---

## 3. Newton–Schulz 算子：缩放、优化与结论

### 3.1 基线缩放结果

- 尺寸：16×16, 32×32, 64×64, 128×128（batch=96，iterations=10）。
- 通过 `analyze_newton_schulz_scaling.py` 生成：
  - `DOCS/NEWTON_SCHULZ_SCALING_BASELINE.md`（表 + 复杂度分析）。
  - `newton_schulz_910b_baseline_scaling_cycles.png`。

- 表中典型结果（Total）：
  - 16²：2263 cycles
  - 32²：3159
  - 64²：7040
  - 128²：16369

### 3.2 理论复杂度与实测趋势

- 单矩阵、每轮迭代：
  - Cube：2 次 GEMM，复杂度 $\Theta(N^3)$。
  - Vector：一次 $R = C - T$，复杂度 $\Theta(N^2)$。
  - MOVIN/MOVOUT：读写数据量 $\Theta(N^2)$。
- 所以主项为 $\Theta(T N^3)$，Vector 与 Load/Store 是低阶项。
- 实测中：
  - $16^2 \to 32^2$：Total ×1.40
  - $32^2 \to 64^2$：×2.23
  - $64^2 \to 128^2$：×2.33
- 原始 FLOP 的 $N^3$ 会给出「翻倍尺寸 → ×8」的增长，但在 910B 类架构中：
  - 大量并行 PE + 多 core。
  - 在 batch 维度和 tile 维度复用权重和激活。
- 因此 **wall-clock 周期的增长远低于纯 FLOP 的 $N^3$**，呈现「亚立方」趋势，但：
  - Cube 行随 $N$ 增长快于 MOVIN / MOVOUT / Vector。
  - 大尺寸下 Total 基本由 Cube 主导。

### 3.3 ping–pong / super-tile 优化实验

- 在 `NewtonSchulzOptOp` 中尝试：
  - 在同一 core 上对多个 tile 做 double-buffering。
  - 通过 barrier + 地址切换隐藏 load/store 与 vector 阶段。
- 结果：
  - 基线 NS 已经是 compute-bound 算子（Cube 很满）。
  - 新的优化增加了 barrier 和控制开销，总周期 **反而增加**。
- 结论：
  - 在 compute-bound kernel 上，过多的 pipelining / double buffering 可能适得其反。

---

## 4. MMSE 算子：结构、缩放与版本一致性

### 4.1 MMSE 算子流水线

- 参数：
  - `matrix_m` = M（天线数 / BS 维度）。
  - `matrix_k` = K（用户数）。
  - `batch_size` = 96。
  - `iterations` = 10（Newton–Schulz 逆的迭代次数）。
- 单 batch 算法序列：
  1. 读入 $H, X_0, C, Y$ → `Load`。
  2. Cube：`MMSE_HtH` 计算 $G = H^\top H$。
  3. Vector：`MMSE_G_PLUS_SIGMA` 计算 $\tilde G = G + \sigma^2 I$。
  4. Newton–Schulz 逆：`MMSE_INV_T/R/X` 在 32×32 块上迭代近似 $\tilde G^{-1}$。
  5. Cube：`MMSE_WH` 计算 $W = G^{-1} H^\top$。
  6. Cube：`MMSE_WY` 计算 $\hat X = W Y$。
  7. `Store` 结果。

### 4.2 MMSE 缩放分析脚本

- `scripts/analyze_mmse_scaling.py`：
  - 固定 K=32：M ∈ {64, 128, 256, 512, 1024}。
  - 固定 M=256：K ∈ {16, 32, 64, 128}。
  - 对每个 (M, K)：
    - 加载对应 CSV。
    - 分类为 MOVIN / Cube / Vector / MOVOUT。
    - 做 union-of-intervals 聚合。
  - 输出：
    - `DOCS/MMSE_SCALING_BASELINE.md`。
    - `mmse_910b_scaling_fixed_k32_cycles.png`。
    - `mmse_910b_scaling_fixed_m256_cycles.png`。

### 4.3 阶段命名不一致的坑

- 早期 256×32 的 run 使用旧版本 MMSEOp：
  - `name` 集合为：`MMSE_NS_*`, `MMSE_APPLY_*`, 多个 `MMSE_BARRIER_*`。
- 512×32 run 使用新版 MMSEOp：
  - `name` 集合为：`MMSE_HtH`, `MMSE_G_PLUS_SIGMA`, `MMSE_INV_*`, `MMSE_WH`, `MMSE_WY`, 若干 `MMSE_BARRIER_*`。
- 直接做 overlay 时出现问题：
  - 除了 `Load` / `Store` 外，几乎没有完全同名阶段。
  - 结果是每一行要么只有蓝色、要么只有橙色，很难直观比较。
- 解决办法：
  - 用当前 binary 和实现 **重新跑 256×32**：
    - 得到新的 `profiling_log_mmse_910b_256x32.csv`，其 `Name` 集合与 512×32 一致。
    - 新 256×32 Total 为 6954 cycles，512×32 为 9396：
      - 之前「512×32 总周期 < 256×32」的反直觉现象被消除。
  - 重新运行 `scripts/analyze_mmse_scaling.py`：
    - 更新所有包含 256×32 的表格与曲线。

### 4.4 Overlay 清理与对比

- `scripts/plot_mmse_timeline_overlay.py` 的改进：
  - 过滤掉所有 `MMSE_BARRIER_*` 及细碎向量阶段。
  - 仅保留主阶段：
    - `MMSE_HtH`, `MMSE_G_PLUS_SIGMA`,
    - `MMSE_INV_T`, `MMSE_INV_R`, `MMSE_INV_X`,
    - `MMSE_WH`, `MMSE_WY`,
    - 加上 `Load`, `Store`。
- 生成的 `pipeline_mmse_910b_256x32_vs_512x32_overlay.png`：
  - 每一行都是同一语义阶段，蓝色（256×32）和橙色（512×32）在同一行上比较。
  - 可以更清楚地看出：随着 M 从 256→512，哪些阶段变长/变短，以及 Load/Store 是否被更好地隐藏。

---

## 5. 自动化迭代与实验流程

### 5.1 配置与模型 JSON

- 典型位置：`example/`：
  - Newton–Schulz：
    - `newton_schulz_16x16.json`, `newton_schulz_32x32.json`, `newton_schulz_64x64.json`, `newton_schulz_128x128.json`。
  - MMSE：
    - 固定 K=32：`mmse_64x32.json`, `mmse_128x32.json`, `mmse_256x32.json`, `mmse_512x32.json`, `mmse_1024x32.json`。
    - 固定 M=256：`mmse_256x16.json`, `mmse_256x32.json`, `mmse_256x64.json`, `mmse_256x128.json`。
    - 基线测试：`mmse_test.json`。
- 硬件配置：
  - 统一使用 `configs/ascend_910b_quiet.json`。

### 5.2 批量实验脚本

- Newton–Schulz：
  - `scripts/analyze_newton_schulz_scaling.py`：
    - 内部约定好不同尺寸对应的 trace CSV 路径。
    - 一次性完成：
      - 数据统计 → Markdown 表格 → 缩放曲线。
- MMSE：
  - `scripts/analyze_mmse_scaling.py`：
    - 读取所有 (M, K) 组合对应的 CSV。
    - 输出两组缩放表格 + 两张曲线。
- Timeline / Overlay：
  - `visualizer_png.py`：生成 per-core pipeline PNG。
  - `plot_mmse_timeline_overlay.py`：生成两个配置之间的 overlay。

### 5.3 建议的新算子开发工作流

1. 在 `src/operations` 中以现有算子（`NewtonSchulzOp.cc` / `MMSEOp.cc`）为模板创建新 `.cc` 与 `.h`。
2. 实现：
   - `parse_attributes`
   - `infer_shapes_from_model`
   - `initialize_tiles`
   - `initialize_instructions`
3. 编写一个小尺寸、小 batch 的模型 JSON（例如 16×16, batch=2），先验证功能与数值正确性。
4. 开启 `ONNXIM_TRACE_CSV`，跑一次仿真，检查：
   - tile 负载均衡日志。
   - per-core pipeline PNG。
5. 编写 `scripts/analyze_<op_name>_scaling.py`：
   - 扫描一系列尺寸/参数组合。
   - 输出 Markdown 表与缩放曲线。
6. 如有 baseline vs 优化版本：
   - 增加 overlay 脚本，对比时间线与阶段重叠情况。

---

## 6. 踩过的坑与经验总结

### 6.1 阶段命名与实现版本不一致

- MMSE 早期和后期版本使用了不同的 `Name` 集合：
  - 旧：`MMSE_NS_*`, `MMSE_APPLY_*`。
  - 新：`MMSE_HtH`, `MMSE_G_PLUS_SIGMA`, `MMSE_INV_*`, `MMSE_WH`, `MMSE_WY`。
- 后果：
  - overlay 图上除了 Load/Store 外，很难做“同一阶段”的一行一对比。
  - 不同 run 的 Total 与 per-phase 行被混在一起，分析容易出错。
- 经验：
  - 变更算子内部结构或阶段命名时，要同步更新：
    - 分析脚本中的 phase 白名单 / 分类逻辑。
    - 最好重新跑一遍关键 baseline，避免新旧实现混用。

### 6.2 向量 / barrier 噪音

- profiling CSV 中存在大量 `MMSE_BARRIER_*` 和极短向量事件：
  - 在 timeline 上表现为密集的垂直细线，干扰观察。
- 经验：
  - 在 overlay / 高层次汇总图中要先过滤掉这些噪音阶段。
  - 真正需要排查调度细节时，再看完整 per-core pipeline PNG。

### 6.3 Newton–Schulz 多轮迭代的数据路径错误

- 初始版本中，没有正确从 ACCUM 中读取迭代更新后的 `X_k`。
- 修复后：
  - 第 0 次迭代从 SPAD 中的 `X_0` 读。
  - 后续迭代从 ACCUM 读 `X_k`，并用 `src_from_accum=true` 标记。
- 经验：
  - 多轮迭代算子一定要非常清晰地指定：
    - 状态写到哪一层存储。
    - 下一轮从哪一层读取。
  - 否则数值等价性和性能统计都会偏离。

### 6.4 “优化”不一定更快

- NS 的 ping–pong / super-tile 实验：
  - 理想：隐藏 load/store 与 vector。
  - 实际：增加 barrier / 控制指令，使得已经 compute-bound 的 kernel 更慢。
- 经验：
  - 优化前先判断算子是 compute-bound 还是 memory-bound：
    - compute-bound：主要瓶颈在 Cube。
    - memory-bound：主要瓶颈在 DRAM / MTE。
  - 对 compute-bound 算子，增加 pipeline 深度/双缓冲可能只是增加调度复杂度与同步成本。

### 6.5 环境变量与路径问题

- 容易出现的问题：
  - 忘记更新 `ONNXIM_TRACE_CSV`，误读旧 CSV。
  - 使用旧 binary 配合新脚本，导致阶段名不一致。
- 经验：
  - 分析脚本中打印当前使用的 CSV 路径与 run 参数。
  - 对关键 baseline 保留“运行命令 + config JSON”记录，确保可重现。

---

**后续对话建议入口**

如果下一轮对话希望：

- 设计一个新型矩阵求逆 / 预条件算子；
- 或者在 NS / MMSE 的基础上尝试新的调度（如分块、分级缓存策略）；
- 或者在现有脚本上扩展更多自动化 sweep / 报告生成；

可以直接以上述内容为默认上下文，我可以从：

1. C++ 实现新算子；
2. JSON / config 建模；
3. trace 生成与分析脚本编写；
4. 到最终 Markdown 报告与图表，

整条链路帮你接着往下做。
