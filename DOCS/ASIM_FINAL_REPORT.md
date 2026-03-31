# Asim 综合技术报告（整合版）

报告内容顺序为：

1. 仿真工程框架
2. 周期与矩阵计算原理（先搬运与同步、后 GEMM）
3. 底层可调用算子（Opcode）总览
4. 仿真工具对比与选型依据
5. LDL vs Cholesky 实验报告

---

## 1. 仿真工程框架

### 1.1 分层架构

Asim（ONNXim 工程化版本）采用“模型图 -> Tile -> 指令 -> 硬件流水线”的分层体系：

- **模型层（Model）**：构建算子图与张量，负责任务定义（`src/models/*`）。
- **算子层（Operation）**：将算子展开为 tile 和指令序列（`src/operations/*`）。
- **调度层（Scheduler）**：把可执行 tile 分发到多核，维护 layer 级生命周期（`src/scheduler/*`）。
- **核心层（Core/SystolicWS）**：执行 `MOVIN/GEMM/Vector/MOVOUT`，维护队列与流水线。
- **互连与存储层（ICNT + DRAM）**：处理 request/response 传输与访存完成。
- **统计层（TraceLogger）**：输出指令级 trace（CSV）与各类统计指标。

用 Cholesky 分解做一个“每层在干什么”的对应示例：

- **模型层（Model）**：创建 `H`、`A=H^HH+\lambda I`、`A_inv` 等张量与算子依赖。
- **算子层（Operation）**：把 Cholesky 路径拆成 `POTRF/TRSM/RK_UPDATE/SOLVE` 指令序列。
- **调度层（Scheduler）**：把不同 batch / tile 派发到 24 个 core，维护层完成状态。
- **核心层（Core/SystolicWS）**：执行 `MOVIN -> GEMM_PRELOAD -> DIV/ADD/... -> MOVOUT`。
- **互连与存储层**：承载每条 `MOVIN/MOVOUT` 触发的 memory request 往返。
- **统计层**：输出 `Cube/Vector/MTE2/MTE3` 事件，为“哪个阶段最慢”提供证据。

关键入口：

- `src/main.cc`：模式分发（如 `ldl_test`、`cholesky_test`、`mmse_test`、`matmul_test`）。
- `src/Simulator.cc`：全局仿真主循环与多时钟域推进。
- `src/Core.cc`, `src/SystolicWS.cc`：核心执行逻辑与周期模型。

### 1.2 以 910B 配置为例：配置文件定义了什么

以 `configs/ascend_910b_quiet.json` 为例，配置中给出了仿真必须的硬件参数：

- **核心规模**：`num_cores = 24`
- **核心频率**：`core_freq = 1200`（用于 cycle -> time 换算）
- **Cube 模型**：
  - `cube_m = cube_n = cube_k = 16`
  - `cube_base_latency = 1`
- **每核阵列与向量参数**（`core_i`）：
  - `core_width = 16`, `core_height = 16`
  - `vector_process_bit = 2048`
  - `add/mul/div/exp/...` 等向量时延
- **片上存储**：`spad_size`, `accum_spad_size`, `sram_width`
- **DRAM 侧**：`dram_type`, `dram_freq`, `dram_channels`, `dram_req_size`
- **互连侧**：`icnt_type`, `icnt_freq`, `icnt_latency`, `icnt_injection_ports_per_core`

这些参数共同决定：

1. 指令单条时延模型（特别是 GEMM / Vector）；
2. 搬运请求颗粒度（`dram_req_size`）；
3. 总周期由 Core/ICNT/DRAM 三域耦合后的结果。

### 1.3 端到端执行链路

1. 读取 `config + models_list`，初始化 `Simulator` 与 `OperationFactory`。
2. `Model::initialize_model()` 构建 `Tensor` 与 `Operation`。
3. `Operation::initialize_tiles()` 生成 tile（包含 `core_id/batch` 等信息）。
4. `Operation::initialize_instructions()` 下发底层指令：
   - 搬运：`MOVIN/MOVOUT`
   - 计算：`GEMM_PRELOAD/GEMM/ADD/MUL/...`
   - 同步：`PIPE_BARRIER`
5. Scheduler 向 core 持续投喂 tile；core 周期推进各队列和流水线。
6. Trace 输出 `start_cycle/end_cycle`，用于后续分阶段归因与可视化。

## 2. 周期与矩阵计算原理

### 2.1 全局周期推进：多时钟域离散事件

`Simulator::set_cycle_mask()` 同时维护 Core/ICNT/DRAM 三个时钟域：

- Core 时间推进：`_core_time += _core_period`
- ICNT 时间推进：`_icnt_time += _icnt_period`
- DRAM 时间推进：`_dram_time += _dram_period`

每轮选最小时间戳推进，可能同一轮触发多个域。因此总周期是“多域耦合结果”，不是单核纯串行累计。

### 2.2 Core 级周期：队列 + 两条流水线

每个 core 维护：

- `_ld_inst_queue`（load）
- `_ex_inst_queue`（execute）
- `_st_inst_queue`（store）
- `_compute_pipeline`（Cube）
- `_vector_pipeline`（Vector）

每个 `cycle()` 主要动作：

1. 退休已到期的 compute/vector 指令；
2. 尝试发射新指令（受数据依赖和资源约束）；
3. 处理 load/store 请求与响应；
4. 更新 active/bubble/idle 统计。

### 2.3 MOVIN 原理与周期来源

#### 执行机理

- `MOVIN` 进入 `_ld_inst_queue`。
- `Core::handle_ld_inst_queue()` 对每个 `src_addr` 生成 `MemoryAccess` 请求（按 `dram_req_size` 分包）。
- 请求经 ICNT 到 DRAM，完成后响应返回 core。
- 响应消费时填充 SPAD/ACCUM，并记录 `CoreX_MTE2:Load` 事件。

#### 周期模型

`MOVIN` 没有固定 compute 周期，属于端到端传输延迟：

$$
T_{movin}=T_{queue}+T_{icnt(req)}+T_{dram}+T_{icnt(rsp)}+T_{consume}
$$

符号说明：

- $T_{queue}$：`MOVIN` 在 core 侧队列等待可发射的时间（受前序依赖、端口占用影响）。
- $T_{icnt(req)}$：读请求从 core 注入 ICNT 并到达 DRAM 侧的传输时间。
- $T_{dram}$：DRAM 侧服务时间（排队 + 访问 + 返回准备）。
- $T_{icnt(rsp)}$：读响应从 DRAM 返回 core 的互连传输时间。
- $T_{consume}$：core 消费响应并写入 SPAD/ACCUM 的时间。

设总传输字节为 $B$、请求粒度 $G=\text{dram\_req\_size}$：

$$
N_{req}=\left\lceil\frac{B}{G}\right\rceil
$$

其中：$B$ 为本次搬运总字节数，$G$ 为单次内存请求粒度（`dram_req_size`），$N_{req}$ 为请求条数。

近似可写为：

$$
T_{movin}\approx T_{first}+(N_{req}-1)\cdot T_{interval}
$$

其中：$T_{first}$ 为首包完成延迟（通常包含首包排队与首跳传输），$T_{interval}$ 为稳态下相邻请求完成的平均间隔。

#### 2.3.1 数值示例：搬运 `256×16` 的 fp16 矩阵

设矩阵尺寸 `256×16`，精度 `fp16`（2 Byte），`dram_req_size=64 Byte`：

$$
B = 256\times16\times2 = 8192\ \text{Byte},\quad
N_{req}=\left\lceil\frac{8192}{64}\right\rceil=128
$$

因此该 `MOVIN` 至少对应 128 个请求间隔（理想下限）；实际值还需叠加首包和拥塞。

### 2.4 MOVOUT 原理与周期来源

#### 执行机理

- `MOVOUT` 进入 `_st_inst_queue`。
- 命中 SPAD/ACCUM 后生成写请求，`_waiting_write_reqs++`。
- 写响应返回后 `_waiting_write_reqs--`，并记录 `CoreX_MTE3:Store` 事件。

#### 周期模型

$$
T_{movout}=T_{queue}+T_{icnt(req)}+T_{dram(write)}+T_{icnt(rsp)}+T_{consume}
$$

符号说明：

- $T_{queue}$：`MOVOUT` 在 store 队列等待发射时间。
- $T_{icnt(req)}$：写请求通过 ICNT 到 DRAM 的时间。
- $T_{dram(write)}$：DRAM 写入服务时间（排队 + 写执行）。
- $T_{icnt(rsp)}$：写完成响应返回 core 的时间。
- $T_{consume}$：core 侧完成写回记账与状态回收时间（如 `_waiting_write_reqs` 归零路径）。

同样由 $N_{req}=\lceil B/G\rceil$ 决定吞吐项；注意 tile 指令弹出后仍可能被 `_waiting_write_reqs` 拖尾。

#### 2.4.1 数值示例：同一矩阵的写回周期

对同样 `256×16` fp16 矩阵：

$$
N_{req}=128
$$

理想对称链路下至少 128 个请求间隔；写回拥塞较重时 `T_{movout}` 常大于 `T_{movin}`。

### 2.5 PIPE_BARRIER 原理与周期来源

#### 执行机理

`PIPE_BARRIER` 作为同步指令走 Vector 路径，用于显式分段 MTE/Cube/Vector 依赖。

#### 周期模型

在当前实现中，barrier 自身执行占用固定为 1 周期：

$$
C_{barrier}=1
$$

其中：$C_{barrier}$ 为 barrier 指令本体在 vector 流水线的执行占用周期。

端到端观测：

$$
T_{barrier}=T_{wait\_issue}+1
$$

其中：$T_{wait\_issue}$ 表示 barrier 因前序依赖尚未满足而无法发射的等待时间。

真实性能影响主要来自等待前序依赖，而非这 1 周期本身。

### 2.6 Ascend `16×16×16` 矩阵计算方法

#### 2.6.1 配置层：开启 Ascend cube 模型

典型配置（`configs/ascend_910b_quiet.json`）：

```json
"ascend_cube_model": {
  "enabled": true,
  "cube_m": 16,
  "cube_n": 16,
  "cube_k": 16,
  "cube_base_latency": 1
}
```

`Common.cc` 会把全局 cube 参数下发到各 core（并支持 per-core override）。

#### 2.6.2 算子层：按 cube 维度切片发射

在 `src/operations/NewtonSchulzOptOp.cc` 的 tiled GEMM 逻辑中，按 `(m0,n0,k0)` 三重循环发 `GEMM_PRELOAD`，并设置：

- `tile_m = min(cube_m, N-m0)`
- `tile_n = min(cube_n, K-n0)`
- `tile_k = min(cube_k, K-k0)`

这就是 Ascend 风格 `16×16×16` 的块化矩阵计算组织方法。

#### 2.6.3 切片示例：`32×32×32`

当 `N=K=32` 且 `cube_m=n=k=16` 时，`m/n/k` 三个维度各切 2 份，总计 `2×2×2=8` 条 `GEMM_PRELOAD`。

### 2.7 GEMM 指令周期模型（放在搬运之后）

这一节回答两个问题：

1. **PE 阵列模型为什么是这个公式**；
2. **Ascend `16×16×16` 分块为什么按 blocks 乘积计工作量**。

先给直观图景：

- 在 WS（Weight-Stationary）PE 阵列中，权重尽量驻留在阵列/片上，输入和部分和沿阵列传播；
- 传播是“波前”方式，因此会出现**填充（fill）**与**排空（drain）**开销；
- 真正的乘加工作量由需要处理的有效块数决定，因此会出现“块数乘积”项。

这也是后续公式里 `H+W-2`（填充排空）与 `b_m b_n b_k`（块工作量）的物理来源。

#### 2.7.1 经典 WS 模型（未启用 Ascend cube）

$$
C_{gemm}=H+W-2+\max(\text{compute\_size},4)
$$

其中：

- $H/W$：PE 阵列高/宽；
- $H+W-2$：二维波前从首个 PE 传播到最后一个 PE 的填充+排空开销；
- `compute_size`：该实现中的主工作量近似项（反映有效计算深度）；
- `max(\cdot,4)`：最小执行占用下限，避免过小 tile 被估成不合理的极低周期。

为什么这样算：该模型把 GEMM 周期拆成“固定结构开销 + 与工作量相关开销”，适合快速估算与架构趋势对比。

#### 2.7.2 Ascend cube 分块模型（启用后）

设 cube 大小为 $(c_m,c_n,c_k)$，指令 tile 为 $(t_m,t_n,t_k)$：

$$
b_m=\left\lceil\frac{t_m}{c_m}\right\rceil,\
b_n=\left\lceil\frac{t_n}{c_n}\right\rceil,\
b_k=\left\lceil\frac{t_k}{c_k}\right\rceil
$$

$$
C_{cube}=L_b+(c_m+c_n-2)+\max(b_m\cdot b_n\cdot b_k,1)
$$

其中：

- $c_m,c_n,c_k$：cube 基本计算块尺寸（Ascend 风格常取 `16,16,16`）；
- $t_m,t_n,t_k$：当前 GEMM 指令 tile 尺寸；
- $b_m,b_n,b_k$：tile 在三维上需要切成多少个 cube 子块；
- $L_b$：cube 基础启动开销（`cube_base_latency`）；
- $(c_m+c_n-2)$：二维阵列路径上的 fill/drain 结构项；
- $b_m b_n b_k$：需要执行的 cube 子任务总数（近似主工作量）。

为什么这样算：

- $m,n$ 维决定输出平面要覆盖多少个 `16×16` 子块；
- $k$ 维决定每个输出子块需要累加多少段部分和；
- 因而总子任务数自然是 $b_m\times b_n\times b_k$；
- 尾块（非 16 整除）通过上取整与 `min(...)` 切片自动处理，不会丢算。

此外，`GEMM_PRELOAD` 发射时还包含 pipeline 启动偏移与相邻指令发射间隔影响。

#### 2.7.3 GEMM 修改前后原理对比（重点）

本项目 `GEMM` 改造的核心，是从“经典 PE 阵列近似模型”过渡到“Ascend `16×16×16` 分块模型”。

**修改前（经典 PE 阵列 / WS 近似）**

- 周期由阵列形状与单一 `compute_size` 主导：

$$
C_{old}=H+W-2+\max(\text{compute\_size},4)
$$

- 特点：实现简单、可快速估算；但不显式区分 `m/n/k` 三维切块数量。

**修改后（Ascend cube 分块）**

- 周期由 `(tile_m,tile_n,tile_k)` 与 `(cube_m,cube_n,cube_k)` 的分块关系决定：

$$
C_{new}=L_b+(c_m+c_n-2)+\max\left(\left\lceil\frac{t_m}{c_m}\right\rceil\left\lceil\frac{t_n}{c_n}\right\rceil\left\lceil\frac{t_k}{c_k}\right\rceil,1\right)
$$

符号与 2.7.2 一致；该式可理解为“结构固定项 + 三维分块工作量项”。

- 特点：直接表达 `16×16×16` 微块执行数量，更贴近 Ascend 风格 GEMM 组织方式。

**同一 tile 的对比示例（便于直观理解）**

设 `core_height=core_width=16`，`cube_m/n/k=16`，`L_b=1`，目标 tile 为 `32×32×32`：

- 旧模型（取 `compute_size=32`）：

$$
C_{old}=16+16-2+32=62
$$

- 新模型：

$$
b_m=b_n=b_k=2\Rightarrow C_{new}=1+30+8=39
$$

这里可以看出：新模型把“工作量”映射到 cube 子块数量，解释性更强。  
但总时延仍需叠加发射等待、访存、barrier 与流水线偏移，不能只看单条公式值。

#### 2.7.4 为什么 Ascend GEMM 在 Asim 中必须改

从工程与建模一致性的角度，必须从旧 WS 近似切到 Ascend cube 分块，原因有四类：

1. **硬件语义不一致会造成周期解释偏差**  
  旧式 $C_{old}=H+W-2+\max(\text{compute\_size},4)$ 只保留“阵列形状 + 单一工作量”，
  但 Ascend GEMM 的真实调度语义是 $(m,n,k)$ 三维分块执行。若模型不显式保留三维块数，
  就无法解释“同等 FLOPs、不同形状 tile”在硬件上的周期差异。

2. **旧模型对 `k` 维累加深度不敏感**  
  Ascend 的 CUBE 计算中，$k$ 维代表每个输出块的累加段数；
  旧模型把其折叠进 `compute_size` 后，会弱化 `split-k` 或长 `k` 场景的差异，
  导致对关键路径压力判断失真。

3. **无法准确表达尾块与非整除 tile 开销**  
  实际运行中大量 tile 不是 16 的整数倍，Ascend 路径用 `min(...)` + 上取整处理尾块；
  若周期模型不按 $\lceil t_m/c_m\rceil,\lceil t_n/c_n\rceil,\lceil t_k/c_k\rceil$ 计，
  会系统性低估或高估边界块时延。

4. **会削弱“算法优化结论”的可信度**  
  本项目要比较 `LDL vs Cholesky` 的并行收益，本质依赖“指令组织 + 分块粒度”差异。
  如果 GEMM 周期模型仍停留在旧抽象，结论可能只反映“公式口径变化”，
  而不是 Ascend 风格执行流真实收益。

因此，改造目标不是“让数值更小/更大”，而是让模型对齐硬件执行语义：

- **可解释**：能从块数直接追溯周期；
- **可比较**：不同 tile 形状比较公平；
- **可归因**：瓶颈能落到 `m/n/k` 哪一维与哪类指令组织。

#### 2.7.5 Ascend GEMM 在 Asim 中应该怎么改（实现路径）

建议按“配置 -> 指令生成 -> 执行周期 -> 统计验证”四步改造：

1. **配置层（Config）**  
  在配置中显式给出：
  - `cube_m/cube_n/cube_k`（默认 `16/16/16`）
  - `cube_base_latency`  
  并支持全局参数下发到每个 core（允许 per-core override）。

2. **算子层（Operation 指令生成）**  
  生成 `GEMM_PRELOAD/GEMM` 时，保留 tile 三维信息 `tile_m/tile_n/tile_k`，
  并按 `(m0,n0,k0)` 三重循环切块；尾块由 `min(...)` 自动裁剪。

3. **执行层（SystolicWS 周期计算）**  
  在执行时采用：

$$
b_m=\left\lceil\frac{t_m}{c_m}\right\rceil,\ 
b_n=\left\lceil\frac{t_n}{c_n}\right\rceil,\ 
b_k=\left\lceil\frac{t_k}{c_k}\right\rceil
$$

$$
C_{cube}=L_b+(c_m+c_n-2)+\max(b_m\cdot b_n\cdot b_k,1)
$$

  同时保留发射偏移、流水线占用和相邻指令间隔等时序项，避免只用“裸计算周期”。

4. **统计层（Trace 与结果校验）**  
  用 `trace` 验证三件事：
  - `Cube avg_dur` 是否与分块规模变化趋势一致；
  - `Cube events` 是否与指令条数（分块数）同向变化；
  - 总 `span` 的变化是否能由 `Cube + MTE + barrier` 联合解释。

#### 2.7.6 改造后应满足的正确性与边界

- **正确性目标**：
  1) 退化到整齐块时，与手算块数一致；
  2) 非整除尾块时，不丢算、不重复算；
  3) `ascend_cube_model.enabled=false` 时，仍可回退旧模型。

- **边界说明**：
  当前公式仍是“周期近似模型”，不是 RTL 级逐拍仿真；
  对 bank conflict、细粒度端口争用、特殊指令融合等细节未完全展开，
  但已足以支撑算法路径对比与瓶颈归因。

- **实践建议**：
  在报告中同时给出“单条 tile 估算（口径A）+ 显式多指令切片（口径B）”，
  分别用于快速估算与条目级归因，避免口径混用。

#### 2.7.7 数值示例：`256×32 × 32×256` 的 GEMM 周期

以下采用当前 Ascend 配置常见参数：`cube_m=n=k=16`、`L_b=1`。

**口径A：单条“大 tile”指令估算**（直接用分块公式）

- $t_m=256,\ t_n=256,\ t_k=32$
- $b_m=\lceil 256/16\rceil=16$
- $b_n=\lceil 256/16\rceil=16$
- $b_k=\lceil 32/16\rceil=2$
- $b_m b_n b_k = 512$

$$
C_{cube}=1+(16+16-2)+512=543
$$

即：按“大 tile 抽象”时，单条 GEMM 指令计算周期约 **543 cycles**（不含发射等待与访存）。

**口径B：显式 `16×16×16` 多指令切片**

- 指令条数：

$$
N_{inst}=\left\lceil\frac{256}{16}\right\rceil\left\lceil\frac{256}{16}\right\rceil\left\lceil\frac{32}{16}\right\rceil=16\times16\times2=512
$$

- 单条 `16^3` 纯计算近似 `32 cycles` 时：

$$
C_{pure}\approx 512\times 32=16384
$$

两种口径分别用于不同层次：口径A适合快速抽象，口径B更接近指令条目级开销分析。

#### 2.7.8 数值示例：`64×32 × 32×64`

对 $C_{64\times 64}=A_{64\times 32}B_{32\times 64}$，按 `16×16×16` 切片：

$$
N_{inst}=\left\lceil\frac{64}{16}\right\rceil\left\lceil\frac{64}{16}\right\rceil\left\lceil\frac{32}{16}\right\rceil=4\times4\times2=32
$$

若单条 `16^3` 纯计算近似 `32 cycles`：

$$
C_{pure}\approx 32\times 32 = 1024
$$

#### 2.7.9 LDL 场景下的 GEMM 粒度适配

`LDLDecompOp` 中按 `cube_m/n/k` 推导 pack 粒度（如 `2x2 -> 16x16` 拼接），并以 `PIPE_BARRIER` 保证关键阶段依赖，从而减少碎片化与过细 k 循环开销。

以 8 组独立 `2x2` 子块更新为例：

- **未拼接**：频繁发射微小 GEMM，队列与调度开销高；
- **拼接后**：先 pack 成 `16x16` 块，再以更大粒度执行 `GEMM_PRELOAD`，减少指令条目与气泡。

该策略在 LDL 实验中体现为 Cube 事件数显著下降（见第 5 节量化表）。

---

## 3. 底层可调用算子（Opcode）总览

### 3.1 算子清单（完整版）

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
| `SQRT` | Vector 非线性 | SystolicWS Vector Pipeline | Cholesky 对角开方、归一化 | `vec_op_iter * scalar_sqrt_latency` | ✅ 已实现 |
| `GELU` | Vector 非线性 | SystolicWS Vector Pipeline | 激活函数 GELU | `vec_op_iter * gelu_latency` | ✅ 已实现 |
| `SWISH` | Vector 非线性 | SystolicWS Vector Pipeline | 激活函数 Swish | 当前与 `GELU` 共用同一时延模型（源码含 TODO） | ⚠️ 可运行但为近似模型 |
| `BAR` | 同步（保留） | Tile/调度语义层 | 历史/保留 barrier 语义 | 当前执行路径主要使用 `PIPE_BARRIER`，`BAR` 未见活跃分支 | ⚠️ 枚举保留 |
| `PIPE_BARRIER` | 同步 | Vector Pipeline（1-cycle 占位） | 显式建模 MTE/Cube/Vector 之间阶段屏障 | 固定返回 1 周期，用于 trace 可视化和依赖分段 | ✅ 已实现 |

### 3.2 关键说明

- `Opcode` 的“定义全集”来自 `src/Common.h`，但“是否真正可用”取决于执行路径里是否有对应处理逻辑（`Core.cc` + `SystolicWS.cc`）。
- 当前主路径中：
  - `MOVIN/MOVOUT/MOVOUT_POOL` 由 Core 内存队列处理；
  - `GEMM/GEMM_PRELOAD` 进入 Cube pipeline；
  - 其余多数进入 Vector pipeline，并由 `get_vector_compute_cycles()` 给出时延。
- 当前生产路径以 `SystolicWS` 为主；`SystolicOS::cycle()` 仍未落地。

### 3.3 与“类似 GEMM 的底层算子”对应关系

- **Cube 核心**：`GEMM_PRELOAD`, `GEMM`
- **Vector 基础算子**：`ADD`, `MUL`, `MAC`, `DIV`, `EXP`, `SQRT`, `ADDTREE`, `COMP`
- **Vector 复合算子**：`SOFTMAX`, `LAYERNORM`, `GELU`, `SWISH`
- **搬运与同步**：`MOVIN`, `MOVOUT`, `MOVOUT_POOL`, `PIPE_BARRIER`

---

## 4. 仿真工具对比与选型依据

### 4.1 对比维度

围绕本项目真实需求，选型关注：

1. 是否支持算子执行流改造（指令级）
2. 是否可定位微架构瓶颈（依赖、碎片化、同步）
3. 是否可输出可复盘证据（trace/周期/时序图）
4. 是否支持计算-存储-互联联合建模

### 4.2 工具横向对比

| 工具 | 主要定位 | 核心优势 | 与本项目目标的适配边界 |
|---|---|---|---|
| **Asim (ONNXim扩展)** | 多核 NPU 周期级仿真 | 指令级 trace、可改 C++ 算子、可视化链路完整 | 标准公开 benchmark 对照少于 Timeloop/SCALE-Sim |
| **SCALE-Sim v3** | 阵列层/层级分析 | 吞吐与带宽快扫高效 | 细粒度指令依赖表达不足 |
| **Timeloop (+Accelergy)** | 映射搜索与性能/能耗分析 | 映射空间探索强、复现生态好 | 不是指令执行引擎，难做指令链排障 |
| **MAESTRO** | 数据流 cost model | 分析快、复用解释性强 | 非完整周期执行仿真 |
| **STONNE** | 可重构 DNN 加速器周期仿真 | dense/sparse 可重构抽象丰富 | 与 Ascend 风格单元同构性较弱 |
| **Astra-sim** | 分布式训练系统仿真 | 系统级计算-通信建模强 | 不聚焦单 NPU 核内执行流 |
| **Ramulator2** | DRAM 周期仿真 | 内存系统建模成熟 | 不含 NPU 计算核语义 |

### 4.3 本项目选型结论

对于本项目的核心任务（`LDL vs Cholesky` 指令流优化与周期归因），Asim 匹配度最高，原因：

- 支持在算子源码层直接改指令链并立即验证；
- 可形成“定位 -> 修正 -> 量化”闭环证据链；
- 在同一框架内覆盖 Core/NoC/DRAM，避免单核模型偏差。
---

## 5. LDL vs Cholesky 实验报告

### 5.1 研究目标

在相同硬件配置与相同问题规模下，对比两类矩阵求逆实现路径的时序效率：

- 新算子：`LDL` 路径（含 `block_size=2` 的 `2x2 -> 16x16` 批拼接优化）
- 基线算子：`Cholesky` 路径

目标是回答：在当前 NPU 风格指令级仿真模型中，`LDL` 相对 `Cholesky` 是否带来显著时延收益，以及收益来自哪些实现层面的差异。

### 5.2 对比口径

- 硬件配置：`configs/ascend_910b_quiet.json`（24 核）
- 维度配置：`M=64, U=16, batch=96`
- 新算子：`ldl_test`
- 基线算子：`cholesky_test`
- 数据来源：
  - `ldl_new/trace_c24.csv`
  - `cholesky_baseline/trace_c24.csv`

两条实验路径保持：

- 相同 `M/U/batch`；
- 相同核心数量与时钟配置；
- 相同 trace 导出方式（`ONNXIM_TRACE_CSV`）；
- 相同最大周期保护（`ONNXIM_MAX_CORE_CYCLES`）。

### 5.3 算法与实现映射（保留精华版）

目标矩阵：

$$
A = H^H H + \lambda I
$$

#### 5.3.1 Cholesky 路径

$$
A = LL^H,\quad A^{-1} = (L^H)^{-1}L^{-1}
$$

实现映射上，主要由以下阶段构成：

- `POTRF_GEMM`
- `TRSM`
- `RK_UPDATE`
- `SOLVE_*`

在当前基线实现中，`RK_UPDATE` 小块更新链是事件密度与关键路径拉长的主要来源。

#### 5.3.2 LDL 路径

$$
A = LDL^H,\quad A^{-1}=L^{-H}D^{-1}L^{-1}
$$

实现映射上，主要对应：

- `D_UPDATE`
- `D_INV`
- `L_UPDATE`
- `BWD_*`

与 Cholesky 相比，LDL 在当前建模中避免显式开方密集链，并通过 `L_UPDATE` 拼接优化减少微小 GEMM 条目。

#### 5.3.3 LDL 相对 Cholesky 的并行性优势（论文关注点）

从任务依赖图看，Cholesky 的关键瓶颈之一是对角更新与后续更新链条存在更强的顺序依赖；而 LDL 将“对角缩放（$D$）”与“下三角更新（$L$）”分离后，更容易形成可并行的批处理更新窗口。

在 Asim 模型中的体现主要有三层：

- **算子依赖层（Operation DAG）**：`D_UPDATE/D_INV/L_UPDATE` 可在更多 tile 上形成并行就绪，减少长串行链。
- **调度层（Scheduler）**：更多 ready tile 可同时投喂 24 个 core，提高多核并行填充度。
- **指令层（Core trace）**：`GEMM_PRELOAD` 更偏向大粒度批处理，微小更新条目减少，队列气泡与发射间隙下降。

对应到观测指标：

- `Cube events` 显著下降但 `Cube avg_dur` 基本不变，说明优化主要来自“指令组织与并行暴露”，而非单条指令神奇变快；
- `span` 大幅下降，符合“关键路径被缩短 + 并行覆盖提升”的并行性收益特征。

#### 5.3.4 2x2 拼接优化可行性（结论保留）

将多组独立 `2x2` 子问题按块对角拼接为更大块后，其乘法结果在块级上与逐块独立计算等价；在本配置下取 `m=8`，可把 8 个 `2x2` 子问题拼成 `16x16`，与 `16x16x16` cube 粒度匹配，从而减少小粒度发射与调度气泡。

### 5.4 主要结论

- LDL 相比 Cholesky 在本配置下实现了 **71.94%** 的总周期跨度降低（约 70%+）。
- 提升来源：
  1) LDL 路径减少了 Cholesky 中高成本 `RK_UPDATE` 链；
  2) `block_size=2` 的 `L_UPDATE` 批拼接提升了 Cube 发射颗粒度；
  3) Cube 指令条数显著下降，调度/发射开销同步下降。

### 5.5 量化结果（24 核全量）

| 指标 | LDL New | Cholesky Baseline | 变化 |
|---|---:|---:|---:|
| 总事件数 | 16,416 | 32,448 | -49.41% |
| 总周期跨度（span） | 3,844 | 13,697 | **-71.94%** |
| Cube 事件数 | 4,896 | 19,008 | **-74.24%** |
| Vector 事件数 | 6,912 | 8,832 | -21.74% |
| MTE2 事件数 | 3,840 | 3,840 | 0 |
| MTE3 事件数 | 768 | 768 | 0 |
| Cube 平均持续周期 | 32.06 | 32.02 | +0.04 |
| Vector 平均持续周期 | 1.33 | 3.35 | -2.01 |

提升比例：

$$
\frac{13697 - 3844}{13697} \approx 71.94\%
$$

### 5.6 Core0/1 视图补充

- `ldl_new/trace_core01.csv`：`events=1368`, `span=3755`
- `cholesky_baseline/trace_core01.csv`：`events=2704`, `span=13632`

Core0/1 局部视图与全量趋势一致，LDL 关键路径更短。

### 5.7 实验方法说明

#### 5.7.1 执行步骤

1. 编译 `Simulator`。
2. 在同一 24 核配置下分别运行：
   - `ldl_test`
   - `cholesky_test`
3. 统一设置：
   - `ONNXIM_TRACE_CSV=...` 导出 trace；
   - `ONNXIM_MAX_CORE_CYCLES=120000` 防止异常死锁。
4. 用 `visualizer_png.py` 绘制：
   - `timeline_c24.png`（全核）
   - `timeline_core01.png`（Core0/1）

#### 5.7.2 指标定义

- `span = end_cycle.max - start_cycle.min`（关键指标）
- `events`：事件总数
- `cube_events/vector_events`：按执行单元分类事件数
- `avg_dur`：同类事件平均持续周期

#### 5.7.3 公平性与边界

- 比较对象是**当前仿真建模下的执行时序效率**，不直接等价数值精度对比。
- 结论适用于当前 `M=64, U=16, batch=96, 24核` 配置；更大规模需复测。

### 5.8 两核流水线与分公式周期统计（完整版）

#### 5.8.1 两核流水线图

- Cholesky（Core0/Core1）：`cholesky_baseline/timeline_core01.png`
- BlockLDL（Core0/Core1）：`ldl_new/timeline_core01.png`

#### 5.8.2 Cholesky 分步骤周期表

Trace：`cholesky_baseline/trace_c24.csv`，`span=13697` cycles。

| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X$ 与外存交换 | 258546 | 0 | 0 | 258546 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| POTRF对角开方 | $L_{jj}=\operatorname{chol}(\tilde A_{jj})$ | 0 | 0 | 3072 | 3072 | 768 |
| TRSM列缩放 | $L_{ij}=(\cdots)L_{jj}^{-H}$ | 0 | 0 | 10752 | 10752 | 2688 |
| Schur/RK更新 | $A_{ik}\leftarrow A_{ik}-L_{ij}L_{kj}^H$ | 0 | 258048 | 0 | 258048 | 8064 |
| 前向对角求逆 | $Y_{jj}=1/L_{jj}$ | 0 | 0 | 3072 | 3072 | 768 |
| 前向累乘 | $Y_{ij}\mathrel{-}=L_{ik}Y_{kj}$ | 0 | 86016 | 0 | 86016 | 2688 |
| 前向缩放 | $Y_{ij}\leftarrow Y_{ij}/L_{ii}$ | 0 | 0 | 10752 | 10752 | 2688 |
| 后向组装 | $A^{-1}=Y^H Y$ | 0 | 3072 | 0 | 3072 | 96 |
| 同步屏障(Barrier) | 列/阶段依赖同步 | 0 | 0 | 1824 | 1824 | 1824 |

#### 5.8.3 BlockLDL 分步骤周期表

Trace：`ldl_new/trace_c24.csv`，`span=3844` cycles。

| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X$ 与外存交换 | 246520 | 0 | 0 | 246520 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| D块更新 | $D_{jj}=A_{jj}-\sum L_{jk}D_{kk}L_{jk}^H$ | 0 | 24576 | 0 | 24576 | 768 |
| D块求逆 | $D_{jj}^{-1}$ | 0 | 0 | 3072 | 3072 | 768 |
| L块更新 | $L_{ij}=(A_{ij}-\sum L_{ik}D_{kk}L_{jk}^H)D_{jj}^{-1}$ | 0 | 21504 | 0 | 21504 | 672 |
| 回代对角初始化 | $X_{jj}=D_{jj}^{-1}$ | 0 | 21504 | 672 | 22176 | 1344 |
| 回代（乘+加） | $tmp\mathrel{+}=L^H X\;\text{并累加到}\;X$ | 0 | 86016 | 2688 | 88704 | 5376 |
| 同步屏障(Barrier) | 列/阶段依赖同步 | 0 | 0 | 2688 | 2688 | 2688 |

注：当前实现中无独立 `LDL_BWD_OFF_SCALE` 指令；回代路径采用 `GEMM_PRELOAD + ADD` 融合执行，因此以“回代（乘+加）”合并展示。

#### 5.8.4 统计口径说明：为何 LDL 的搬运与 Cube 看起来更“平衡”

你这个观察是对的，`ldl_new/trace_c24.csv` 在该实验口径下确实比前面 DeepUnfold 报告更“平衡”。主要原因有两类：

1) **工作负载不同（核心原因）**
- 本节 LDL/Cholesky 对比来自 `M=64, U=16, batch=96, 24核` 路径，且 LDL 内含较多块更新/回代 GEMM，Cube 工作量占比高。
- 对应全量统计（24核）里，LDL 约为：`MTE=246520`, `Cube=156960`, `Vector=9216`，即 MTE 约 59.7%，Cube 约 38.0%。

2) **统计口径不同（次要但常见）**
- 本表是“全核全时域累计 `sum(dur)`”。
- 若看局部时间窗或单核 timeline，连续的 Cube 段在视觉上会更显长；大量碎片化 MTE 段会被低估。

因此，“这里看起来平衡”与“另一个报告中 MTE 占比极高”并不冲突，本质是**任务规模与算子结构不同**，并叠加了展示口径差异。

### 5.9 LDL 优化点（对应当前实现）

1. **算法路径更轻量**：减少 Cholesky 中高密度更新链开销。
2. **2x2 批拼接提升 Cube 利用**：显著降低微小 GEMM 条数。
3. **依赖与稳定性完善**：关键阶段补充保守屏障策略。

### 5.10 结果文件索引

- LDL：
  - `results/LDL/compare_ldl_vs_cholesky_20260309/ldl_new/timeline_c24.png`
  - `results/LDL/compare_ldl_vs_cholesky_20260309/ldl_new/timeline_core01.png`
- Cholesky：
  - `results/LDL/compare_ldl_vs_cholesky_20260309/cholesky_baseline/timeline_c24.png`
  - `results/LDL/compare_ldl_vs_cholesky_20260309/cholesky_baseline/timeline_core01.png`

### 5.11 深度展开（Deep-Unfolding NS）与 Cholesky/LDL 对比（同环境）

为验证“深度展开参数化 NS”在当前 MMSE 场景下的可用性，本次新增了 Python 统一评测：

- 三种方法：`Cholesky-MMSE`、`Block-LDL`、`Deep-Unfolding NS`；
- 同一环境：`nr=64, nt=16, n_sc=32, batch=24, trials=3, 16QAM`；
- 同一信道估计：`LS`，`pilot_len=16`；
- 同一 SNR 网格：`0/5/10/15/20 dB`；
- 同一数值仿真口径：`fp16 + reciprocal approx`。

Deep-Unfolding NS 采用：

$$
X_{k+1}=X_k(\beta_k I-A X_k)
$$

在本实验中使用 `phase1` 低成本集成：

- 自适应迭代：`nt=16 -> 1` 次迭代；
- `\beta_{avg}=3.33`；
- `maxeig` 归一化 + 后缩放校正（抵消固定点尺度偏置）。

关键结果（来自 `results/LDL/deep_unfold_compare_main/deep_unfold_vs_cholesky_ldl_metrics.csv`）：

| SNR(dB) | BER-Cholesky | BER-LDL | BER-DeepUnfold | SE-Cholesky | SE-LDL | SE-DeepUnfold |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.2569 | 0.2579 | 0.2782 | 31.94 | 32.01 | 31.32 |
| 5  | 0.1066 | 0.1080 | 0.1606 | 54.70 | 54.77 | 45.86 |
| 10 | 0.0106 | 0.0109 | 0.0868 | 80.05 | 80.12 | 55.01 |
| 15 | 3.39e-05 | 5.43e-05 | 0.0634 | 106.23 | 106.26 | 59.33 |
| 20 | 0.0 | 0.0 | 0.0561 | 132.70 | 132.63 | 60.85 |

结论（当前配置下）：

1. `Cholesky` 与 `LDL` 曲线基本重合；
2. `Deep-Unfolding NS (phase1, 1 iter)` 在高 SNR 区域仍有明显 BER 尾部；
3. 说明该新方法在“零额外周期偏好”下可作为低成本近似，但要达到与 Cholesky/LDL 接近的检测精度，仍需提升迭代层数或切到 phase2 动态参数。

#### 5.11.1 LDL 与深度展开（周期口径）对比（aligned: `M=256,K=32,batch=96`）

来源：`results/compare_aligned/four_method_cycle_summary.csv` + `four_method_wallclock_vs_work.csv`。

| 指标 | LDL | DeepUnfold | DeepUnfold-Opt |
|---|---:|---:|---:|
| Wall 周期 | 13,953 | 6,921 | **6,370** |
| Work 周期和（`sum(dur)`） | 12,487,766 | 30,398,792 | 29,321,742 |
| MTE 总周期 | 11,912,438 | 30,267,944 | 29,195,790 |
| Cube 周期 | 550,848 | 117,216 | 117,216 |
| Vector 周期 | 24,480 | 13,632 | 8,736 |
| MTE 占比 | 95.39% | 99.57% | 99.57% |

补充（Scheme B，纳入 `CubeWait` 等待周期，来源：`results/LDL/*_trace_schemeB.csv`）：

| 指标 | LDL | DeepUnfold | DeepUnfold-Opt | Cholesky |
|---|---:|---:|---:|---:|
| Cube计算周期 | 550,848 | 117,216 | 117,216 | 4,562,112 |
| Cube等待周期（CubeWait） | 120,167 | 85,937 | 173,016 | 24,679,818 |
| Cube计算占比（`计算/(计算+等待)`） | **82.0918%** | 57.6984% | 40.3870% | 15.6013% |
| Cube计算事件占比（`events`） | 74.3855% | 50.1354% | 50.0966% | 50.7075% |

结论：
- 端到端延迟（Wall）上，DeepUnfold 系列快于 LDL；
- 但累计工作量上，DeepUnfold 系列显著高于 LDL，主要由 MTE 主导；
- 若纳入 `CubeWait`，LDL 的 Cube 有效计算占比最高（82.09%），DeepUnfold 次之，Cholesky 最低；
- 说明其优势主要来自并行重叠，而非总工作量更小。

口径说明：`CubeWait` 统计规则为 `name == "CubeWait"` 或 `unit` 后缀为 `_Wait`，计算规则为 `unit` 后缀为 `_Cube` 且 `name != "CubeWait"`。

#### 5.11.2 深度展开优化前后对比（DeepUnfold vs DeepUnfold-Opt）

| 指标 | 优化前 DeepUnfold | 优化后 DeepUnfold-Opt | 变化 |
|---|---:|---:|---:|
| Wall 周期 | 6,921 | **6,370** | **-7.96%** |
| Work 周期和（`sum(dur)`） | 30,398,792 | 29,321,742 | -3.54% |
| MTE 总周期 | 30,267,944 | 29,195,790 | -3.54% |
| Cube 周期 | 117,216 | 117,216 | 0.00% |
| Vector 周期 | 13,632 | 8,736 | -35.92% |

结论：优化收益主要来自 `Vector` 侧指令合并与依赖链收敛，Cube 工作量基本不变，最终体现为 Wall 周期下降。

新增结果文件索引：

- 指标表：`results/LDL/deep_unfold_compare_main/deep_unfold_vs_cholesky_ldl_metrics.csv`
- BER 图：`results/LDL/deep_unfold_compare_main/ber_vs_snr_cholesky_ldl_deep_unfold.png`
- SE 图：`results/LDL/deep_unfold_compare_main/se_vs_snr_cholesky_ldl_deep_unfold.png`
- 同图（SE+BER）：`results/LDL/deep_unfold_compare_main/se_ber_overlay_cholesky_ldl_deep_unfold.png`
- 文字报告：`results/LDL/deep_unfold_compare_main/deep_unfold_vs_cholesky_ldl_report.md`
