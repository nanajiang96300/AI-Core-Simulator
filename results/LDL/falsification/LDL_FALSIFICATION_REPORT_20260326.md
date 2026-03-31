# LDL 算子证伪实验报告（周期公式 + 数值正确性）

## 1. 目标与可证伪命题

本实验目标：验证 `LDLDecompOp` 的实现是否同时满足**周期模型可推导**与**算法结果正确**。

为确保“可证伪”，定义以下命题：

- **H1（事件计数命题）**：trace 中各类事件数必须与代码循环推导完全一致；
- **H2（事件周期命题）**：事件 duration 分布必须与 `SystolicWS` 周期公式一致；
- **H3（数值命题）**：LDL 求逆路径在质量评估中与基线一致（BER/SE 不劣化，重构误差接近 0）。

任一命题不满足，则判定该实现/模型被证伪。

---

## 2. 实验配置与输入

### 2.1 仿真配置

- 配置文件：`configs/ascend_910b_quiet.json`
- 模式：`ldl_test`
- 模型描述：`example/ldl_test.json`

`example/ldl_test.json` 关键参数：

- `batch_size = 96`
- `matrix_m = 64`
- `matrix_k = 16`（即 `U=16`）
- `block_size = 2`
- `bwd_steps = 1`

### 2.2 Trace 导出

- 环境变量：`ONNXIM_TRACE_CSV`
- 输出：`results/LDL/falsification/ldl_64x16_trace_new_run.csv`

---

## 3. 周期计算流程（严格推导）

以下推导直接对应 `src/operations/LDLDecompOp.cc` 与 `src/SystolicWS.cc`。

### 3.1 基本变量

记：

$$
U=16,\quad blk=2,\quad B=U/blk=8,\quad s=bwd\_steps=1
$$

由于启用 Ascend Cube 模型，`cube_m=cube_n=cube_k=16`，`base=1`。

单条 Cube 事件周期：

$$
C_{cube}=1+(16+16-2)+\max(\lceil tile_m/16\rceil\lceil tile_n/16\rceil\lceil tile_k/16\rceil,1)
$$

单条 Vector 事件周期（`Core::calculate_vector_op_iterations`）：

- `ADD`: `vec_iter * add_latency = 1 * 1 = 1`
- `DIV`: `vec_iter * div_latency = 1 * 4 = 4`
- `PIPE_BARRIER`: 固定 `1`

（`compute_size=4` 时，`vector_process_bit=2048`，每周期处理元素数 `2048/8=256`，故 `vec_iter=1`。）

### 3.2 事件数推导（每 batch）

对应 `LDLDecompOp::initialize_instructions`：

1. 前处理阶段
   - `LDL_GRAM`（GEMM_PRELOAD）: `1`
   - `LDL_REG`（ADD）: `1`

2. BLDL 阶段（`j=0..B-1`）
   - `LDL_D_UPDATE_j`（GEMM_PRELOAD）: `B=8`
   - `LDL_D_INV_j`（DIV）: `B=8`
   - `LDL_L_UPDATE_i_j`（GEMM_PRELOAD，带 pack）:
     $$
     \sum_{j=0}^{B-1}\left\lceil\frac{B-j-1}{P}\right\rceil
     $$
     其中 `P=cube_pack_blocks=8`，故每个 `j<7` 为 1，`j=7` 为 0，总计 `7`。

3. backward 阶段
   - 对角乘加：
     - `BWD_DIAG_MUL`: `s(B-1)=7`
     - `BWD_DIAG_ACC`: `s(B-1)=7`
   - 非对角乘加：
     - `BWD_OFF_MUL`: 
       $$
       s\sum_{j=0}^{B-1}j=s\frac{B(B-1)}{2}=28
       $$
     - `BWD_OFF_ACC`: 同上 `28`

4. barrier 事件（PIPE_BARRIER）
   - 固定三处：`LOAD2GRAM`, `GRAM2REG`, `REG2BLDL` => `3`
   - 每列 BLDL step barrier：`B=8`
   - 每列 backward `DIAG2OFF`：`B=8`
   - 每列 backward `COL`：`B=8`
   - 尾部 `BWD2STORE`：`1`
   - 合计：`3+8+8+8+1=28`

### 3.3 事件数推导（全 batch=96）

- `GEMM_PRELOAD_total = (1 + 8 + 7 + 7 + 28) * 96 = 4896`
- `ADD_total = (1 + 7 + 28) * 96 = 3456`
- `DIV_total = 8 * 96 = 768`
- `BARRIER_total = 28 * 96 = 2688`

### 3.4 duration 分布推导

#### Cube 事件

- `LDL_GRAM`: `tile_m=16,tile_n=16,tile_k=64`
  $$
  blocks=(1,1,4),\; steps=4,\; C=1+30+4=35
  $$
  数量 `96`。

- 其余 `GEMM_PRELOAD`：`tile_m,tile_n,tile_k \le 16`，故 `steps=1`
  $$
  C=1+30+1=32
  $$
  数量 `4896-96=4800`。

所以理论 Cube 分布：

$$
\{32:4800,\;35:96\}
$$

#### Vector 事件

- `ADD` + `PIPE_BARRIER` 均为 `1` 周期：
  `3456 + 2688 = 6144`
- `DIV` 为 `4` 周期：`768`

所以理论 Vector 分布：

$$
\{1:6144,\;4:768\}
$$

---

## 4. 实测结果与证伪判定

数据来源：

- `results/LDL/falsification/ldl_falsification_cycle_report.json`
- `results/LDL/falsification/ldl_falsification_cycle_report.md`

### 4.1 计数对照

| 指标 | 理论 | 实测 | 判定 |
|---|---:|---:|---|
| GEMM_PRELOAD_total | 4896 | 4896 | PASS |
| ADD_total | 3456 | 3456 | PASS |
| DIV_total | 768 | 768 | PASS |
| BARRIER_total | 2688 | 2688 | PASS |

### 4.2 duration 分布对照

- Cube 实测：`{32: 4800, 35: 96}`
- Vector 实测：`{1: 6144, 4: 768}`

与理论完全一致，判定 `PASS`。

### 4.3 周期命题结论

- H1（事件计数）通过；
- H2（事件周期）通过。

故“周期模型可推导并与实现一致”未被证伪。

---

## 5. 内存调度推导与验证（新增）

本节回答“内存调度是否也可推导”。结论：

- **请求数量可严格推导并可证伪**；
- **精确内存时延不可闭式推导**（`ramulator2` 动态仲裁），但可给出边界与流量一致性检验。

### 5.1 调度路径（代码对应）

1. `Core::handle_ld_inst_queue` / `Core::handle_st_inst_queue`：
  - 对每个 `src_addrs` 元素生成一条 `MemoryAccess`，入 `_request_queue`；
2. `Simulator::cycle`：
  - Core -> ICNT：每个 core 的每个注入端口每个 ICNT 周期最多推进 1 条请求；
  - ICNT -> DRAM：若目标 channel 可接收则推进；
  - DRAM -> ICNT -> Core：响应回传并在 `push_memory_response` 中记 `Load/Store` trace；
3. `DramRamulator2`：
  - 采用动态调度，延迟受行缓冲命中/冲突与队列状态影响。

### 5.2 请求数量公式（本实验）

配置：`precision=2`，`dram_req_size=64`，故

$$
elems\_per\_access=\frac{64}{2}=32
$$

`LDLDecompOp` 中仅有两类 `MOVIN` 与一类 `MOVOUT`：

1. `emit_movin(h_base, rows=64, cols=16)`
2. `emit_movin(reg_base, rows=16, cols=16)`
3. 输出 `MOVOUT(rows=16, cols=16)`

虽然 `cols < elems_per_access`，循环会按“每行一次”采样地址，但 `make_address` 最终经 `align_address(addr)=addr-(addr\bmod 64)` 对齐到 64B cacheline。

每行数据字节数：

$$
row\_bytes=16\times2=32\text{ B}
$$

故相邻两行映射到同一 64B 线，唯一地址数减半：

- H 载入：`64 -> 32` 请求/批次
- Reg 载入：`16 -> 8` 请求/批次
- Out 写回：`16 -> 8` 请求/批次

于是每批次：

$$
N_{load/batch}=32+8=40,
\quad
N_{store/batch}=8
$$

总批次 `96`：

$$
N_{load}=40\times96=3840,
\quad
N_{store}=8\times96=768
$$

### 5.3 实测对照（trace）

从 `results/LDL/falsification/ldl_64x16_trace_new_run.csv` 统计：

- `Load` 事件数：`3840`
- `Store` 事件数：`768`

与推导完全一致，说明“请求生成与调度链”未被证伪。

### 5.4 时延边界说明（为何不做闭式精确值）

在 `icnt_type=simple`、`icnt_injection_ports_per_core=8` 下，单 core 每个 ICNT 周期最多推进 `8` 条注入；Simple-ICNT 额外基础延迟为 `icnt_latency=1`。但 DRAM 为 `ramulator2`，其服务次序依赖运行时 bank/row 状态，故不能给出单条请求固定闭式延迟。

因此本报告采用“**请求数严格闭合 + 响应事件闭合 + 时延分布观测**”作为内存调度可证伪标准。

---

## 6. 数值正确性验证（算法层）

数据来源：

- `results/LDL/falsification/quality_check/ldl_quality_report.md`
- `results/LDL/falsification/quality_check/ldl_quality_metrics.csv`

关键结果（本次快速 sanity-check 配置）：

- `recon_error = 1.1740632013438952e-16`
- `max BER gap = 0.000000`
- `max SE gap = 0.000000`
- 自动判定：`PASS`

说明在该实验设置下，LDL 路径与基线在指标上对齐，且重构误差接近数值精度极限。

### 5.1 数值命题结论

- H3（数值命题）通过。

---

## 7. 最终结论

本次 LDL 证伪实验中：

1. **周期推导链闭合**：事件数与 duration 分布从代码推导到 trace 全量吻合；
2. **算法结果正确**：质量评估指标与基线一致，重构误差极小；
3. **内存调度可推导且闭合**：Load/Store 请求数推导与 trace 完全一致；
4. **可证伪而未被推翻**：H1/H2/H3 均通过。

因此，在当前配置与输入规模下，`LDLDecompOp` 的实现与周期模型均具备可靠性证据。

---

## 8. 复现实验命令

```bash
cd /project/Asim

# 1) 运行 LDL 并导出 trace
LD_LIBRARY_PATH=/project/Asim/build/lib:$LD_LIBRARY_PATH \
ONNXIM_TRACE_CSV=/project/Asim/results/LDL/falsification/ldl_64x16_trace_new_run.csv \
./build/bin/Simulator \
  --config ./configs/ascend_910b_quiet.json \
  --models_list ./example/ldl_test.json \
  --mode ldl_test

# 2) 数值正确性评估（快速版）
python3 scripts/evaluate_ldl_quality.py \
  --nr 64 --nt 16 --n-sc 32 --batch 8 --trials 3 \
  --block-size 2 --snr-db 10 \
  --num-format fp64 --reciprocal-mode exact \
  --out-dir results/LDL/falsification/quality_check
```
