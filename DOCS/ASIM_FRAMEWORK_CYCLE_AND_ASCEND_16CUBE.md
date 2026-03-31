# Asim 仿真工程框架、周期计算原理与 Ascend `16×16×16` 矩阵计算方法

本文档面向工程实现，系统整理 Asim（ONNXim 工程化版本）的：

1. 整体框架与执行链路；
2. 周期计算原理（Core/NoC/DRAM 多时钟推进 + 指令级延迟模型）；
3. 矩阵乘法（PE 阵列）周期计算方法；
4. 改造后对齐 Ascend 风格 `16×16×16` 的矩阵计算组织方式。

---

## 1. 工程框架总览

### 1.1 分层结构

Asim 采用“模型图 → Tile → 指令 → 硬件流水线”的层次化仿真：

- **模型层（Model）**：负责构建计算图与张量（`src/models/*`）。
- **算子层（Operation）**：把算子展开为 tile 与指令序列（`src/operations/*`）。
- **调度层（Scheduler）**：把可执行 tile 分配到多核。
- **核心层（Core/SystolicWS）**：执行 `MOVIN / GEMM / Vector / MOVOUT`。
- **互连与存储层（ICNT + DRAM）**：处理 memory request/response。
- **统计与可视化层（TraceLogger + CSV）**：输出指令级时间戳与周期统计。

关键入口：

- `src/main.cc`：模式分发（`ldl_test`、`mmse_test`、`matmul_test` 等）。
- `src/Simulator.cc`：仿真主循环与多时钟域推进。
- `src/Core.cc`、`src/SystolicWS.cc`：核心执行与周期模型。

### 1.2 数据与控制流（从模型到周期）

1. `main` 读取 `config + models_list`，构建 `Simulator`。
2. `Model::initialize_model()` 创建 `Operation` 与 Tensor。
3. `Operation::initialize_tiles()` 生成 tile，设置 `core_id` 与 batch。
4. `Operation::initialize_instructions()` 生成指令序列：
   - 访存：`MOVIN`、`MOVOUT`
   - 计算：`GEMM_PRELOAD/GEMM`、`ADD/MUL/DIV/...`
   - 同步：`PIPE_BARRIER`
5. `Scheduler` 持续发 tile 到 core；core 每周期推进队列与流水线。
6. `TraceLogger` 记录每条指令的 `start_cycle/end_cycle`，导出 CSV。

---

## 2. 周期计算原理（全局）

### 2.1 多时钟域离散事件推进

`Simulator::set_cycle_mask()` 维护三个时钟域：

- Core：`_core_time += _core_period`
- DRAM：`_dram_time += _dram_period`
- ICNT：`_icnt_time += _icnt_period`

每次选当前最小时间戳，可能同时触发多个域（同相位）。这意味着：

- 周期统计不是简单“单时钟串行”仿真；
- Core、NoC、DRAM 的推进速率由配置频率共同决定。

### 2.2 运行终止条件

`Simulator::running()` 为真当且仅当以下任意组件仍忙：

- 模型队列非空；
- 任一 Core 仍有 tile/队列/流水线未清空；
- ICNT 或 DRAM 仍在运行；
- Scheduler 仍有 tile 待发。

开发保护：`ONNXIM_MAX_CORE_CYCLES` 可防止死循环。

---

## 3. 周期计算原理（Core 级）

### 3.1 Core 内部队列与流水线

每个 `Core` 维护：

- `_ld_inst_queue`：`MOVIN`
- `_ex_inst_queue`：计算类指令
- `_st_inst_queue`：`MOVOUT`
- `_compute_pipeline`：Cube（GEMM）流水
- `_vector_pipeline`：Vector 流水

每个 `cycle()` 的主要动作：

1. 完成到期的 compute/vector 指令（写回 SPAD/ACCUM）；
2. 尝试发射 load/execute/store；
3. 处理 SRAM/ACCUM 双缓冲与 tile 完成；
4. 更新统计量（active/bubble/idle/memory idle）。

### 3.2 指令发射与依赖判定

`Core::can_issue_compute()` 会检查 `src_addrs` 对应数据是否在 SPAD/ACCUM 命中；
不命中则不能发射，从而形成真实的数据相关 stall。

同步指令 `PIPE_BARRIER` 在当前实现作为 Vector 管线 1-cycle 占位，主要作用是：

- 在 trace 上显式显示阶段边界；
- 保证跨单元（MTE/Cube/Vector）的可视顺序与依赖分段。

### 3.3 Tile 与双缓冲

`Core::issue()` 使用双缓冲策略：

- `spad_id` 在两个 buffer 间切换；
- `accum_spad_id` 在需要时切换；
- 支持 tile 级流水重叠，减少覆盖冲突。

---

## 4. 矩阵乘法（PE 阵列）周期模型

### 4.1 经典 WS 阵列公式（未开启 Ascend Cube 模型）

当 `enable_ascend_cube_model=false` 时，`SystolicWS::get_inst_compute_cycles()` 使用：

$$
C_{gemm}=H+W-2+\max(\text{compute\_size},4)
$$

其中：

- $H$：阵列高（`core_height`）
- $W$：阵列宽（`core_width`）
- `compute_size`：该指令有效计算长度（由算子侧填写）

该模型可理解为：阵列填充/排空延迟 + 主体计算延迟。

### 4.2 Ascend 风格 Cube 模型（开启后）

当 `enable_ascend_cube_model=true` 时，使用 cube 分块模型：

设：

- cube 基本块：$(c_m,c_n,c_k)$，默认 $(16,16,16)$；
- 指令 tile：$(t_m,t_n,t_k)$；
- 基础延迟：$L_b$（`cube_base_latency`）。

则：

$$
b_m=\left\lceil\frac{t_m}{c_m}\right\rceil,\quad
b_n=\left\lceil\frac{t_n}{c_n}\right\rceil,\quad
b_k=\left\lceil\frac{t_k}{c_k}\right\rceil
$$

$$
C_{cube}=L_b+(c_m+c_n-2)+\max(b_m\cdot b_n\cdot b_k,1)
$$

在 `16×16×16` 且 `L_b=1` 时：

- 若单条指令就是 `16×16×16`：
  $$C_{cube}=1+(16+16-2)+1=32$$
- 若单条指令为 `32×32×32`：
  $$b_m=b_n=b_k=2\Rightarrow C_{cube}=1+30+8=39$$

> 说明：这是**单条 GEMM 指令**的 compute 周期，不含 load/store、队列等待、barrier、NoC/DRAM 排队。

### 4.3 发射侧附加时序（GEMM_PRELOAD）

除 `get_inst_compute_cycles()` 外，`SystolicWS::cycle()` 在发射 `GEMM_PRELOAD` 时还建模了起始偏移逻辑：

- 当 compute pipeline 为空，首条 `GEMM_PRELOAD` 的 `start_cycle` 会加上预装载相关补偿（与 `core_height` 相关）；
- 当 pipeline 非空，下一条 GEMM 的可启动时间受前条指令与 offset 约束。

因此“指令级总耗时”应理解为：

$$
T_{inst}=\text{issue\_wait}+C_{cube}\quad(\text{或 }C_{gemm})
$$

---

## 5. PE 阵列矩阵乘法周期估算方法

设矩阵乘：

$$
C_{M\times N}=A_{M\times K}\cdot B_{K\times N}
$$

### 5.1 指令级估算（按 `16×16×16` 显式切片）

若算子侧显式以 `16×16×16` 切片发射，则 GEMM 指令数：

$$
N_{inst}=\left\lceil\frac{M}{16}\right\rceil
\left\lceil\frac{N}{16}\right\rceil
\left\lceil\frac{K}{16}\right\rceil
$$

理想纯计算周期近似：

$$
C_{pure}\approx N_{inst}\cdot 32
$$

工程总周期更准确写法：

$$
C_{total}\approx C_{movin}+C_{cube\_issue\_overlap}+C_{vector}+C_{movout}+C_{stall}
$$

其中 `stall` 包括 SRAM miss、依赖等待、NoC/DRAM 拥塞与 barrier 泡泡。

### 5.2 与“单条大 tile”建模的关系

Asim 同时支持两种方式：

1. **大 tile 单指令**：`tile_m=M,tile_n=N,tile_k=K`，由 `C_cube` 内部的 `b_m,b_n,b_k` 体现分块成本；
2. **显式 16^3 多指令**：算子侧自己三重循环发多条 `GEMM_PRELOAD`。

两者都可用，但用途不同：

- 方式 1 更紧凑，适合快速建模；
- 方式 2 更贴近“硬件块级执行轨迹”，更利于插 barrier、看碎片化与依赖链。

---

## 6. 改造后 Ascend `16×16×16` 矩阵计算方法

本工程当前“对齐 Ascend 风格”的核心在两层：

### 6.1 配置层：全局开启 Cube 模型

`configs/ascend_910b_quiet.json` 中：

```json
"ascend_cube_model": {
  "enabled": true,
  "cube_m": 16,
  "cube_n": 16,
  "cube_k": 16,
  "cube_base_latency": 1
}
```

`Common.cc` 会把该全局配置下发到每个 core（也可按 core 覆盖）。

### 6.2 算子层：按 cube 维度切 tile 发指令

在 `src/operations/NewtonSchulzOptOp.cc` 中，已实现 `emit_tiled_gemm`：

- 读取 `cube_m/cube_n/cube_k`；
- 三重循环 `(m0,n0,k0)` 切片；
- 每片发一条 `GEMM_PRELOAD`，并设置：
  - `tile_m = min(cube_m, N-m0)`
  - `tile_n = min(cube_n, K-n0)`
  - `tile_k = min(cube_k, K-k0)`

这就是与 Ascend `16×16×16` 对齐的“块化矩阵乘指令组织方式”。

### 6.3 LDL 等算子中的适配策略

在 `src/operations/LDLDecompOp.cc` 中，存在按 core cube 参数选择 pack 粒度的逻辑：

- 若启用 cube 模型，`cube_dim_target` 会受 `cube_m/n/k` 共同约束；
- 据此推导 `cube_pack_blocks`，把小块更新尽量拼成更合适的 cube 任务；
- 再用 `PIPE_BARRIER` 对阶段切分，保证 RAW 依赖正确。

这类改造的目标是：

- 让指令粒度更接近硬件执行块；
- 在保证正确性的前提下减少碎片化与过细 k-循环。

---

## 7. `16×16×16` 的实际计算示例

### 7.1 示例 A：单核 `32×32×32` GEMM，显式 `16^3` 切片

- 切片数：
  $$N_{inst}=\lceil 32/16\rceil^3=8$$
- 每条 `16^3` 指令 compute 周期约 32。

理想仅计算项（不含等待）近似：

$$
C_{pure}\approx 8\times 32=256
$$

实际 trace 中通常更高，因为存在：

- `GEMM_PRELOAD` 发射间隔与管线占位；
- `MOVIN/MOVOUT`、barrier；
- 共享内存与互连竞争。

### 7.2 示例 B：单条 `tile=32×32×32` 指令（非显式切片）

由 cube 公式：

$$
C_{cube}=1+30+8=39
$$

该值是“抽象单指令的周期模型”，不是完整算子总周期。

---

## 8. 工程落地建议（面向后续算子）

1. **先保正确性再调粒度**：先保证依赖链闭合（必要 barrier），再做分块/融合。
2. **优先对齐 `16×16×16`**：矩阵核算子默认按 cube 维度切片，尾块用 `min(...)` 处理。
3. **区分“模型延迟”和“系统总周期”**：`get_inst_compute_cycles` 只是一部分，必须结合 NoC/DRAM/队列 stall。
4. **用 trace 驱动优化**：开启 `ONNXIM_TRACE_CSV`，对 `Cube/Vector/MTE2/MTE3` 做分段归因。
5. **避免过度 barrier**：只在真实 RAW/阶段依赖处插入，减少 1-cycle 占位累积气泡。

---

## 9. 关键代码索引

- 全局推进与多时钟：`src/Simulator.cc`
- Core 队列/双缓冲/依赖检查：`src/Core.cc`, `src/Core.h`
- Cube/Vector 周期模型：`src/SystolicWS.cc`
- 配置解析（含 `ascend_cube_model`）：`src/Common.cc`, `src/SimulationConfig.h`
- 通用 GEMM 指令生成：`src/operations/GemmWS.cc`
- Ascend `16^3` 切片示例：`src/operations/NewtonSchulzOptOp.cc`
- block + barrier 示例：`src/operations/LDLDecompOp.cc`

---

## 10. MOVIN / MOVOUT / PIPE_BARRIER 原理与周期推导（补充）

本节补充三类“非 GEMM 主算子”指令的执行机理与周期来源，避免把它们简单等同为固定常数。

### 10.1 MOVIN（DRAM -> SPAD/ACCUM）

#### 1) 执行原理（代码路径）

1. `Core::cycle()` 把 `MOVIN` 放入 `_ld_inst_queue`。
2. `Core::handle_ld_inst_queue()`：
  - 先 `prefetch()` 在 SPAD/ACCUM 预留目标地址；
  - 对 `src_addrs` 中每个地址创建一个 `MemoryAccess`（`write=false, request=true`）；
  - 每个 request 的 `size = dram_req_size`，并记录 `start_cycle`。
3. `Simulator::cycle()` 在 ICNT 时钟域把 request 从 core 推到互连，再到 DRAM。
4. DRAM 完成后回包（`request=false`），再经 ICNT 返回 core。
5. `Core::push_memory_response()` 收到回包后填充 SPAD，并记录 trace 事件：
  - unit: `CoreX_MTE2`
  - name: `Load`
  - 区间：`[start_cycle, 当前_core_cycle]`

#### 2) 周期怎么得出

MOVIN 没有单独的 `get_inst_compute_cycles()` 常数；其周期来自**端到端存储系统延迟**：

$$
T_{movin}=T_{queue}+T_{icnt(req)}+T_{dram}+T_{icnt(rsp)}+T_{core\_consume}
$$

若数据总字节数为 $B$，请求粒度为 $G=\text{dram\_req\_size}$，请求数：

$$
N_{req}=\left\lceil\frac{B}{G}\right\rceil
$$

工程上可写为“首包延迟 + 吞吐项”：

$$
T_{movin}\approx T_{first} + (N_{req}-1)\cdot T_{interval}
$$

其中：

- `SimpleInterconnect` 下，同一 `(src,dst)` 路径近似每周期出一个包（加固定 `icnt_latency`）；
- `SimpleDram` 下，请求完成时间由 `max(_cycles + dram_latency, _last_finish_cycle)` 推进，体现串行服务特征；
- `Ramulator/Ramulator2` 下，`T_{dram}` 由行命中/冲突、bank 并行度和队列拥塞决定，不是固定常数。

### 10.2 MOVOUT（SPAD/ACCUM -> DRAM）

#### 1) 执行原理（代码路径）

1. `Core::cycle()` 把 `MOVOUT/MOVOUT_POOL` 放入 `_st_inst_queue`。
2. `Core::handle_st_inst_queue()`：
  - 检查源地址在 SPAD/ACCUM `check_hit()`；
  - 为每个写回块创建 `MemoryAccess`（`write=true, request=true`）；
  - `_waiting_write_reqs++`，用于跟踪未完成写回数量。
3. 经过 ICNT -> DRAM -> ICNT 返回写响应后，`Core::push_memory_response()`：
  - `response->write == true` 时执行 `_waiting_write_reqs--`；
  - 记录 trace 事件：unit `CoreX_MTE3`，name `Store`。

#### 2) 周期怎么得出

与 MOVIN 一样，MOVOUT 主体周期由写回路径决定，不是固定“计算周期”：

$$
T_{movout}=T_{queue}+T_{icnt(req)}+T_{dram(write)}+T_{icnt(rsp)}+T_{core\_consume}
$$

请求数同样近似为 $\lceil B/G\rceil$。

注意：

- `MOVOUT` 指令本身可较早从 `_st_inst_queue` 弹出；
- 但 core 的 `running()` 仍受 `_waiting_write_reqs != 0` 约束；
- 因此“tile 看似发完”与“系统真正完成写回”之间可能有显著尾延迟。

### 10.3 PIPE_BARRIER（阶段屏障）

#### 1) 执行原理（代码路径）

`PIPE_BARRIER` 走 Vector 执行路径，在 `SystolicWS::get_vector_compute_cycles()` 中明确返回：

$$
C_{barrier}=1
$$

它不发内存请求、不做算术计算，主要用于：

- 把跨单元依赖（MTE/Cube/Vector）显式化；
- 在 trace 中标出阶段边界，便于归因分析。

#### 2) 周期怎么得出

单条 barrier 的**执行占用**是 1 周期，但端到端观测到的延迟为：

$$
T_{barrier}=T_{wait\_issue}+1
$$

其中 `T_wait_issue` 取决于：

- 前序指令是否完成（依赖/队列）；
- Vector pipeline 是否空闲（当前实现一次只执行一条 vector 指令）；
- 与 GEMM/MOVIN/MOVOUT 的同步编排位置。

因此 barrier 的“性能影响”通常来自**串行化效应**而非这 1 个周期本身。

### 10.4 实践中如何读这三类周期

建议同时看两套口径：

1. **指令模型口径**：
  - `PIPE_BARRIER = 1 cycle`
  - MOVIN/MOVOUT 无固定 compute 周期
2. **系统观测口径（推荐）**：
  - 直接用 trace 的 `Load/Store` 区间（MTE2/MTE3）做并集统计；
  - 再与 `Cube/Vector` 区间叠加，得到真实瓶颈归因。

换言之：

- `PIPE_BARRIER` 是“可见同步点”；
- `MOVIN/MOVOUT` 是“受互连与内存系统支配的传输过程”；
- 三者都会影响总周期，但影响机理不同。

---

## 11. 一句话结论

Asim 当前的周期模型可概括为：**“以 tile 指令流为中心、以 Core/NoC/DRAM 多时钟协同推进为基础、以 Ascend `16×16×16` Cube 分块公式约束 GEMM 周期，并通过 trace 做端到端归因。”**

---

## 12. Asim 优缺点与“适合做什么仿真”的详细分析

本节从“是否能高保真拟合昇腾 NPU”的角度，给出 Asim 的能力边界。

### 12.1 主要优势（当前工程已经具备）

1. **算子级可编程性强，迭代效率高**  
  可在 `src/operations/*` 直接改指令生成逻辑（`GEMM_PRELOAD/ADD/PIPE_BARRIER/MOVIN/MOVOUT`），适合快速验证算法-算子协同优化思路。

2. **证据链完整，便于归因**  
  `TraceLogger` 可导出 `Cube/Vector/MTE2/MTE3` 的起止周期，支持“瓶颈定位 -> 修改 -> 对比复现”的闭环。

3. **具备多核+NoC+DRAM联合建模骨架**  
  `Simulator` 在 Core/ICNT/DRAM 三时钟域推进，优于只做核内算子延迟估算的纯解析模型。

4. **支持 Ascend 风格 Cube 分块参数化**  
  通过 `ascend_cube_model(cube_m,n,k,base_latency)` 可统一控制 `16×16×16` 分块模型，并可按 core 覆盖。

5. **工程组织清晰，适合研究型开发**  
  `Model -> Operation -> Tile -> Instruction` 分层明确，便于把论文算法映射成可执行实验工件。

### 12.2 主要缺陷（影响昇腾高保真拟合）

1. **OS 数据流未落地**  
  `SystolicOS::cycle()` 当前 `assert(0)`，意味着有效执行路径基本是 `SystolicWS`，覆盖面不足。

2. **互连/存储背压模型偏理想**  
  `SimpleInterconnect::is_full()` 与 `SimpleDram::is_full()` 当前恒为 `false`，容易低估拥塞、队列阻塞和尾延迟。

3. **向量侧并行能力建模偏保守**  
  `SystolicWS::can_issue_compute()` 对向量指令采用“vector pipeline 非空则不发射”，等效单发射近似，可能低估向量吞吐。

4. **Barrier 语义过于简化**  
  `PIPE_BARRIER` 固定 1 周期，更接近“可视化同步点”而非硬件真实同步代价。

5. **片上存储冲突细节不足**  
  `Sram` 当前主要做容量与 valid 管理，未细化 bank/port 冲突时序，难覆盖细粒度访存冲突主导场景。

6. **指令语义与昇腾 ISA 非一一对应**  
  `Opcode` 是工程抽象层，适合趋势分析，不等价于厂商硬件真实微指令与运行时协议。

### 12.3 适合做的仿真任务（推荐）

1. **算子设计空间探索（DSE）**  
  例如分块粒度、指令重排、barrier 布局、依赖链修复等“相对收益”比较。

2. **阶段级性能归因**  
  量化 `Load/Compute/Store` 占比，定位是 `Cube` 受限、`Vector` 受限还是 `MTE`/DRAM 受限。

3. **算法-算子协同验证**  
  对 Newton–Schulz、LDL、MMSE 等矩阵类算子，验证“数学改造是否转化为周期收益”。

4. **多核负载分配与调度策略比较**  
  用于比较不同 tile 划分、batch 映射策略下的核心利用率与尾核效应。

### 12.4 不适合或需谨慎解释的任务

1. **对标昇腾实机绝对周期/绝对吞吐**  
  当前更适合“相对变化趋势”，不宜直接声称与实机严格等价。

2. **强依赖运行时细节的系统级结论**  
  涉及复杂流控、优先级、抢占、多流并发策略时，需额外补模或外部校准。

3. **对片上冲突极敏感的微结构结论**  
  若性能主要由 bank/port 冲突决定，现模型可能高估或低估某些路径。

4. **需要 ISA 级行为一致性的验证**  
  若目标是“指令语义逐条等价”，需引入更底层 ISA/Runtime 协议建模。

### 12.5 结论性判断

Asim 目前最适合作为：**昇腾风格 AI Core 的“研究型周期级原型平台”**，用于相对趋势与瓶颈归因。  
若目标升级为“工程交付级高保真对标平台”，需按下一节路线图系统补齐关键模型。

---

## 13. 昇腾适配性边界与改造路线图（P0/P1/P2）

本节给出可执行任务清单，按“收益/风险比”从高到低排序。

### 13.1 P0（高优先级：先做，直接影响结论可信度）

#### P0-1：补齐互连与 DRAM 背压模型

- **任务**：为 `SimpleInterconnect` / `SimpleDram` 增加有限队列深度、`is_full()` 真实判定、拥塞统计。  
- **目标收益**：减少“无限吞吐”假设导致的乐观偏差。  
- **落地点**：`src/Interconnect.cc`, `src/Interconnect.h`, `src/Dram.cc`, `src/Dram.h`。  
- **验收标准**：在高并发 `MOVIN/MOVOUT` 场景下出现可解释的 backpressure；trace 中 MTE 尾延迟与队列占用相关。

#### P0-2：把 `PIPE_BARRIER` 从常数占位升级为依赖屏障模型

- **任务**：区分 barrier 类型（如 MTE->CUBE、CUBE->VECTOR、VECTOR->CUBE），引入“最小开销 + 等待前序完成”的模型。  
- **目标收益**：避免低估同步成本，提高阶段切换拟合度。  
- **落地点**：`src/SystolicWS.cc`（`get_vector_compute_cycles`）、相关算子 barrier 注入点。  
- **验收标准**：barrier 开销随前序繁忙度变化；同一算子不同 barrier 布局可在周期上拉开合理差距。

#### P0-3：文档化“适用边界”并固化评测口径

- **任务**：在报告模板中强制区分“相对收益”与“绝对对标”结论标签。  
- **目标收益**：避免过度解读仿真结果，提升对外报告可信度。  
- **落地点**：`DOCS/*` 报告模板、实验脚本说明。  
- **验收标准**：每份实验报告含适用边界声明与校准状态字段。

### 13.2 P1（中优先级：提升昇腾风格拟合精度）

#### P1-1：增强 Vector 并行发射能力模型

- **任务**：把当前“单向量指令串行”扩展为可配置并行度（如 lane 数/双发射窗口）。  
- **目标收益**：减少向量侧性能低估，提高混合算子拟合度。  
- **落地点**：`src/SystolicWS.cc`（`can_issue_compute`、vector pipeline 结构）。  
- **验收标准**：向量密集算子的吞吐对并行度参数敏感，且趋势与预期一致。

#### P1-2：补充片上存储冲突时序

- **任务**：在 `Sram` 中加入 bank/port 冲突模型与访问仲裁延迟。  
- **目标收益**：提升访存主导型算子的可解释性。  
- **落地点**：`src/Sram.cc`, `src/Sram.h`。  
- **验收标准**：不同访问模式（顺序/跨bank冲突）出现可区分周期差异。

#### P1-3：统一 GEMM_PRELOAD 发射偏移与 Cube 子块映射口径

- **任务**：梳理首条 preload 偏移与后续 issue offset 的物理含义，提供可配置参数。  
- **目标收益**：减少大 tile 与显式 `16^3` 切片间的模型不一致。  
- **落地点**：`src/SystolicWS.cc`, `src/operations/*`。  
- **验收标准**：同规模矩阵在两种建模方式下差异可由参数解释且稳定。

### 13.3 P2（低优先级：长期完善，面向工程对标）

#### P2-1：实现/启用 `SystolicOS` 完整路径

- **任务**：完成 `SystolicOS::cycle()` 与对应统计，支持 OS/WS 对比。  
- **目标收益**：提升体系结构研究覆盖面。  
- **落地点**：`src/SystolicOS.cc`, `src/SystolicOS.h`。  
- **验收标准**：OS 模式可运行并输出完整 trace/stats，无断言退出。

#### P2-2：细化精度与指令语义映射

- **任务**：引入更细粒度 dtype/累加路径配置，逐步逼近昇腾算子语义。  
- **目标收益**：提升绝对周期与数值路径解释能力。  
- **落地点**：`src/SimulationConfig.h`, `src/Common.h`, `src/operations/*`。  
- **验收标准**：不同精度路径对延迟/带宽/指令序列影响可复现实验。

#### P2-3：建立“仿真-实测”校准闭环

- **任务**：构建基准算子集合（MatMul/LayerNorm/Softmax/MMSE），维护校准参数表。  
- **目标收益**：从“趋势可信”升级到“量级可对齐”。  
- **落地点**：`tests/`, `results/`, `DOCS/`。  
- **验收标准**：关键基准在给定配置下达到可接受误差带（例如阶段级误差阈值）。

### 13.4 建议执行顺序（两阶段）

1. **阶段 A（2~4 周）**：完成 P0，先解决可信度问题。  
2. **阶段 B（4~8 周）**：推进 P1，显著提升昇腾风格拟合能力；P2 按项目资源分批并行。

### 13.5 路线图落地检查表（可直接打勾）

- [ ] 已实现 ICNT/DRAM 队列上限与 `is_full()` 背压。  
- [ ] 已实现 barrier 类型化延迟模型并完成回归测试。  
- [ ] 已建立统一报告口径（相对收益 vs 绝对对标）。  
- [ ] 已支持可配置向量并行发射能力。  
- [ ] 已加入 SRAM bank/port 冲突建模。  
- [ ] 已完成至少 1 轮仿真-实测校准并输出参数表。
