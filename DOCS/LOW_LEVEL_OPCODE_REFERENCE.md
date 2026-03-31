# Asim 底层可调用算子（Opcode）总览

本文档基于当前源码实现梳理底层 `Instruction::opcode`（定义于 `src/Common.h`），用于回答“类似 `GEMM` 的底层算子有哪些、怎么用”。

## 1) 算子清单表

| Opcode | 类别 | 主要执行单元/路径 | 典型用途 | 时延/行为模型（当前实现） | 实现状态 |
|---|---|---|---|---|---|
| `MOVIN` | 数据搬运 | Core `LD` 队列 + MTE/DRAM | 从 DRAM 读入 SPAD/ACCUM | 按 `dram_req_size` 分包发起读请求，完成后填充 SRAM | ✅ 已实现 |
| `MOVOUT` | 数据搬运 | Core `ST` 队列 + MTE/DRAM | 从 SPAD/ACCUM 回写 DRAM | 按 `dram_req_size` 分包发起写请求，等待写完成计数归零 | ✅ 已实现 |
| `MOVOUT_POOL` | 数据搬运 | Core `ST` 队列 + MTE/DRAM | 池化输出回写 | 与 `MOVOUT` 共用同一路径 | ✅ 已实现 |
| `GEMM_PRELOAD` | Cube 计算 | SystolicWS Compute Pipeline | 预装载权重并做矩阵乘 | Ascend Cube 模型：`base + fill_drain + blocks_m*blocks_n*blocks_k`；否则经典阵列公式 | ✅ 已实现 |
| `GEMM` | Cube 计算 | SystolicWS Compute Pipeline | 常规矩阵乘（QK、Conv/Gemm 内核等） | 同 `GEMM_PRELOAD`，但无“首发预装载补偿”逻辑 | ✅ 已实现 |
| `GEMM_WRITE` | Cube 计算（保留） | 未进入当前执行分支 | 预期用于 GEMM 写回语义 | 当前未见执行分支和时延模型 | ⚠️ 枚举存在，未实际落地 |
| `COMP` | Vector 计算 | SystolicWS Vector Pipeline | 比较/判定类操作 | `vec_op_iter * 1` | ✅ 已实现 |
| `IM2COL` | Vector 计算（预处理） | 被当作 Vector op 下发 | 卷积前张量重排 | 当前 Vector `switch` 无专门 `case`，会走“not configured operation”并返回 0 周期 | ⚠️ 已被调用，但模型未完善 |
| `SOFTMAX` | Vector 复合 | SystolicWS Vector Pipeline | 注意力 Softmax | `2*add_tree + (add+exp+mul)`（按迭代计） | ✅ 已实现 |
| `LAYERNORM` | Vector 复合 | SystolicWS Vector Pipeline | LayerNorm 归一化 | 包含 `add_tree` + `scalar_sqrt_latency` + 向量算子组合 | ✅ 已实现 |
| `ADD` | Vector 标量化并行 | SystolicWS Vector Pipeline | 向量加法、累加更新 | `vec_op_iter * add_latency` | ✅ 已实现 |
| `MUL` | Vector 标量化并行 | SystolicWS Vector Pipeline | 向量乘法、缩放 | `vec_op_iter * mul_latency` | ✅ 已实现 |
| `MAC` | Vector 标量化并行 | SystolicWS Vector Pipeline | 乘加融合 | `vec_op_iter * mac_latency` | ✅ 已实现 |
| `DIV` | Vector 标量化并行 | SystolicWS Vector Pipeline | 倒数/归一化/三角解等 | `vec_op_iter * div_latency` | ✅ 已实现 |
| `ADDTREE` | Vector 归约 | SystolicWS Vector Pipeline | 求和树归约 | `add_tree_iter * add_tree_latency * tile_m` | ✅ 已实现 |
| `EXP` | Vector 非线性 | SystolicWS Vector Pipeline | Softmax/指数计算 | `vec_op_iter * exp_latency` | ✅ 已实现 |
| `SQRT` | Vector 非线性 | SystolicWS Vector Pipeline | Cholesky 对角开方、归一化 | `vec_op_iter * scalar_sqrt_latency` | ✅ 已实现（近期新增） |
| `GELU` | Vector 非线性 | SystolicWS Vector Pipeline | 激活函数 GELU | `vec_op_iter * gelu_latency` | ✅ 已实现 |
| `SWISH` | Vector 非线性 | SystolicWS Vector Pipeline | 激活函数 Swish | 当前与 `GELU` 共用同一时延模型（源码含 TODO） | ⚠️ 可运行但为近似模型 |
| `BAR` | 同步（保留） | Tile/调度语义层 | 历史/保留 barrier 语义 | 当前执行路径主要使用 `PIPE_BARRIER`，`BAR` 未见活跃分支 | ⚠️ 枚举保留 |
| `PIPE_BARRIER` | 同步 | Vector Pipeline（1-cycle 占位） | 显式建模 MTE/Cube/Vector 之间阶段屏障 | 固定返回 1 周期，用于 trace 可视化和依赖分段 | ✅ 已实现 |

## 2) 关键说明

- `Opcode` 的“定义全集”来自 `src/Common.h`，但“是否真正可用”取决于是否在执行路径里有对应处理逻辑（`Core.cc` + `SystolicWS.cc`）。
- 在当前主路径中：
  - `MOVIN/MOVOUT/MOVOUT_POOL` 由 Core 的内存队列处理；
  - `GEMM/GEMM_PRELOAD` 进入 Cube pipeline；
  - 其余大多进入 Vector pipeline，并由 `get_vector_compute_cycles` 给出时延。
- `SystolicOS` 当前 `cycle()` 仍是 `assert(0)`，因此生产路径实际以 `SystolicWS` 为主。

## 3) 与“类似 GEMM 的底层算子”对应关系

若你关注 NPU 核心算子（Cube + Vector），可以按下列最常用子集理解：

- **Cube 核心**：`GEMM_PRELOAD`, `GEMM`
- **Vector 基础算子**：`ADD`, `MUL`, `MAC`, `DIV`, `EXP`, `SQRT`, `ADDTREE`, `COMP`
- **Vector 复合算子**：`SOFTMAX`, `LAYERNORM`, `GELU`, `SWISH`
- **搬运与同步**：`MOVIN`, `MOVOUT`, `MOVOUT_POOL`, `PIPE_BARRIER`

---

如需，我可以继续基于这份表再生成一版“**按具体模型/算子（MMSE、LDL、Cholesky、Attention）统计实际使用到的 Opcode 覆盖矩阵**”，方便你做论文或报告附录。
