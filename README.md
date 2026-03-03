# Asim: AI-Core Simulator

> 原项目 `ONNXim` 的工程化延续版本。当前仓库已用于 AI Core（含 Ascend 风格 Cube/Vector/MTE）与通信相关算子的周期级仿真。

---

## 1. 项目定位

`Asim`（AI-Core Simulator）是一个多核 NPU 周期级仿真器，支持两类工作流：

1. **ONNX 模型仿真**（原 ONNXim 主能力）
2. **C++ 自定义算子/模型仿真**（本项目重点，已用于 Newton–Schulz、MMSE、Series-Inverse、MatMul 等）

与仅做算子功能模拟不同，Asim 会显式建模：
- Core 级执行（Cube / Vector / MTE）
- 片上 SRAM/SPAD/ACCUM
- NoC（Simple/Booksim2）
- DRAM（Simple/Ramulator1/Ramulator2）

因此可以输出周期、利用率、访存行为和指令级 trace。

---

## 2. 仓库结构（与你当前工作流相关）

```text
src/
  main.cc                         # 模式分发入口（default/language/*_test）
  Simulator.cc                    # 主循环 + Trace CSV 导出
  operations/
    GemmWS.cc                     # 核心 GEMM tile/指令生成
    NewtonSchulzOp.cc             # 基线牛顿迭代算子
    NewtonSchulzOptOp.cc          # 优化版牛顿迭代算子
  models/
    NewtonSchulzModel.cc
    NewtonSchulzOptModel.cc
    MMSEModel.cc
    SeriesInverseModel.cc
    MatmulModel.cc                # 新增 MatMul 测试模型（256x32x256）

configs/
  ascend_910b_quiet.json          # 常用 910B 风格配置（24 核）

example/
  newton_schulz_opt_test.json
  matmul_256x32x256_test.json

visualizer_png.py                 # CSV -> 时序图 PNG
DOCS/                             # 算子开发和实验日志文档
```

---

## 3. 环境与编译

## 3.1 依赖

- Linux（推荐 Ubuntu）
- `python3`
- `cmake`
- `gcc/g++`
- `conan`（若你的本地流程依赖）

Python 依赖（至少用于可视化）：

```bash
python3 -m pip install -r requirements.txt
```

## 3.2 初始化与构建

```bash
git clone https://github.com/PSAL-POSTECH/ONNXim.git Asim
cd Asim
git submodule update --init --recursive

cmake -S . -B build
cmake --build build -j$(nproc)
```

可执行文件：`build/bin/Simulator`

---

## 4. 运行模式总览

`src/main.cc` 当前支持以下 `--mode`：

- `default`：ONNX 路径
- `language`：语言模型 custom trace 路径
- `ls_test`
- `newton_schulz_test`
- `newton_schulz_opt_test`
- `mmse_test`
- `series_inverse_test`
- `matmul_test`

通用参数：

```bash
./build/bin/Simulator \
  --config <config_json> \
  --models_list <models_list_json> \
  --mode <mode> \
  --log_level info
```

---

## 5. Ascend 风格 Cube 配置（16x16x16）

在 `configs/ascend_910b_quiet.json`（或 `ascend_910b.json`）中：

```json
"ascend_cube_model": {
  "enabled": true,
  "cube_m": 16,
  "cube_n": 16,
  "cube_k": 16,
  "cube_base_latency": 1
}
```

含义：
- `cube_m/n/k`：Cube 基本块大小
- `enabled=true`：`SystolicWS::get_inst_compute_cycles()` 采用 Cube 分块延迟模型

---

## 6. 新算子如何构建（重点）

可参考：
- `DOCS/ADD_NEW_OPERATOR.md`
- `DOCS/NEWTON_SCHULZ_OP_GUIDE.md`

## 6.1 开发路径 A：ONNX 节点接入

1. 在 `src/operations/` 新增算子类（继承 `Operation`）
2. 实现：
   - 构造函数（读取 attributes / shape）
   - `initialize_tiles(...)`
   - `initialize_instructions(...)`
3. 在 `OperationFactory` 注册 `op_type -> C++ class`
4. 确认 CMake 能编译到新源文件

## 6.2 开发路径 B：纯 C++ 模型接入

1. 在 `src/models/` 新增模型（如 `MatmulModel`）
2. 在模型内创建 Tensor + Operation 图
3. 在 `src/main.cc` 新增 mode 分支
4. 在 `example/*.json` 提供模型配置

当前仓库案例：
- `NewtonSchulzModel/NewtonSchulzOptModel`
- `MMSEModel`
- `SeriesInverseModel`
- `MatmulModel`

---

## 7. 构建算子的基础 API（GEMM / VECTOR / 内存搬运）

本节列的是你在 `Operation::initialize_instructions(...)` 里最常直接使用的基础能力。

### 7.1 地址与张量基础 API（所有单元通用）

- `get_operand_addr(operand_id)`：获取输入/输出 tensor 的 DRAM 基地址
- `make_address(index, dims)`：把逻辑索引映射成线性地址偏移
- `add_input(...)` / `add_output(...)`：维护算子输入输出关系
- `check_executable()`：检查输入依赖是否就绪

---

### 7.2 GEMM（Cube 单元）基础 API

#### A) 复用现成 GEMM 算子

- 类：`GemmWS`（`src/operations/GemmWS.cc`）
- 入口：`initialize_tiles(...)` + `initialize_instructions(...)`
- 指令核心：`Opcode::GEMM_PRELOAD`（可配 `tile_m/tile_k/tile_n`）

#### B) 在自定义算子里手工发 GEMM 指令

常用写法（示意）：

```cpp
tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
  .opcode = Opcode::GEMM_PRELOAD,
  .dest_addr = accum_addr,
  .src_addrs = std::vector<addr_type>{act_addr, weight_addr},
  .tile_m = m_blk,
  .tile_k = k_blk,
  .tile_n = n_blk,
}));
```

用途：
- 执行 `A x B` 的矩阵乘子块
- 在 Ascend cube 配置下与 `cube_m/n/k` 一起构成 `16x16x16` 等分块策略

---

### 7.3 VECTOR（向量单元）基础 API

通过 `Instruction.opcode` 发向量计算指令，常用包括：

- `Opcode::ADD`
- `Opcode::MUL`
- `Opcode::MAC`
- `Opcode::DIV`
- `Opcode::EXP`
- `Opcode::ADDTREE`
- `Opcode::LAYERNORM`
- `Opcode::SOFTMAX`

示意（逐元素加法）：

```cpp
tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
  .opcode = Opcode::ADD,
  .dest_addr = vec_dst,
  .src_addrs = std::vector<addr_type>{vec_a, vec_b},
  .compute_size = elem_cnt,
}));
```

用途：
- Newton 中 `R = C - T`、`X = X + R` 等逐元素操作
- MMSE/归一化中的标量或向量后处理

---

### 7.4 内存搬运（MTE2/MTE3）基础 API

用于 DRAM <-> SPAD/ACCUM 的数据搬运：

- `Opcode::MOVIN`：从 DRAM 搬入到 SPAD/ACCUM（对应 MTE2）
- `Opcode::MOVOUT`：从 SPAD/ACCUM 写回 DRAM（对应 MTE3）

示意：

```cpp
tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
  .opcode = Opcode::MOVIN,
  .dest_addr = spad_addr,
  .src_addrs = dram_addr_list,
  .size = static_cast<uint32_t>(dram_addr_list.size()),
}));

tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
  .opcode = Opcode::MOVOUT,
  .dest_addr = spad_addr,
  .src_addrs = dram_out_addr_list,
  .size = static_cast<uint32_t>(dram_out_addr_list.size()),
}));
```

同步相关：
- `Opcode::PIPE_BARRIER`：用于 MTE/Cube/Vector 之间的阶段同步

---

### 7.5 推荐的算子构建顺序（模板）

1. `MOVIN`：把 A/B/常量搬到 SPAD
2. `GEMM_PRELOAD`：做主矩阵乘
3. `ADD/MUL/...`：做向量后处理
4. `MOVOUT`：结果写回 DRAM

这个流程与 `NewtonSchulzOp` / `NewtonSchulzOptOp` / `GemmWS` 的实现风格一致。

---

## 8. Tiling 策略怎么做

## 8.1 两个层次要区分

1. **任务层 tiling（tile 分配到 core）**
   - 例如 Newton：按 batch 切 tile，然后 round-robin 分核
2. **指令层 tiling（单 tile 内 GEMM 子块）**
   - 例如 `16x16x16` Cube 块（M/N/K 三维分块）

图上“像一整块”通常是因为：
- 单核串行执行 + 事件密集，视觉连成条带；
- 不代表内部没有 16³ 分块。

## 8.2 16x16x16 的直观计算

以 `MatMul: 256x32 * 32x256` 为例：

$$
\#chunks = \frac{256}{16} \times \frac{256}{16} \times \frac{32}{16} = 16 \times 16 \times 2 = 512
$$

实际 trace 中可看到 `GEMM` 事件数量就是 512（与公式一致）。

## 8.3 映射来源

- 手工：`models/<model>/<model>.mapping`
- 自动：若未提供 mapping，使用默认 Gemmini 风格映射

---

## 9. 导出 CSV 与绘制时序图

## 9.1 导出指令级 trace CSV

通过环境变量：

```bash
export ONNXIM_TRACE_CSV=results/matmul_256x32x256.csv
```

可选安全阈值（防死循环/卡死）：

```bash
export ONNXIM_MAX_CORE_CYCLES=300000
```

运行示例：

```bash
./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/matmul_256x32x256_test.json \
  --mode matmul_test \
  --log_level info
```

## 9.2 绘制时序图 PNG

当前 `visualizer_png.py` 参数为 `-i/-o`：

```bash
python3 visualizer_png.py \
  -i results/matmul_256x32x256.csv \
  -o results/matmul_256x32x256.png
```

CSV 列要求（脚本可识别别名）：
- `unit`
- `name`
- `start_cycle`（或 `startcycle`）
- `end_cycle`（或 `endcycle`）

---

## 10. 已验证示例

## 10.1 Newton–Schulz（优化版）

```bash
export ONNXIM_TRACE_CSV=results/newton_schulz/newton_opt_96b_32x32.csv

./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/newton_schulz_opt_test.json \
  --mode newton_schulz_opt_test \
  --log_level info

python3 visualizer_png.py \
  -i results/newton_schulz/newton_opt_96b_32x32.csv \
  -o results/newton_schulz/newton_opt_96b_32x32.png
```

## 10.2 MatMul（单核观察友好）

```bash
export ONNXIM_TRACE_CSV=results/matmul_256x32x256.csv

./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/matmul_256x32x256_test.json \
  --mode matmul_test \
  --log_level info

python3 visualizer_png.py \
  -i results/matmul_256x32x256.csv \
  -o results/matmul_256x32x256.png
```

---

## 11. 常见问题与解决办法（实战总结）

## 11.1 新增模型后链接报错：`undefined reference to vtable`

**现象**：新增 `*.cc` 后链接失败。  
**原因**：构建系统未重新配置，目标文件未纳入。  
**解决**：

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

## 11.2 运行时 `Floating point exception`

**典型原因之一**：`Operation` 属性构造路径中 `target_core` 未正确写入成员，导致后续以错误 core 索引访问配置。  
**修复点**：`src/operations/Operation.cc`

```cpp
// 错误
target_core = target_core;

// 正确
this->target_core = target_core;
```

## 11.3 MatMul 形状对不上 / 行列混乱

`GemmWS` 期望权重维度语义与输入/输出一致，MatMul 模型中要特别注意 `weight_shape` 排布。当前 256x32x256 示例采用：

```text
input_shape  = [batch, M, K]
weight_shape = [K, N]
output_shape = [batch, M, N]
```

## 11.4 图看起来“一整块”，怀疑没有按 16³ 分块

先看统计，不只看图感受。以 256x32x256 为例，`GEMM` 事件数应为 512（16×16×2）。若一致，说明已按 16³ 在指令级切分。

## 11.5 可视化参数不兼容

不同版本脚本参数可能不同。当前仓库版本是 `-i/-o` 风格；若你使用 `--merge-adjacent` 一类参数，请先确认本地脚本版本是否支持。

## 11.6 双缓冲（ping-pong）反而变慢

在 32x32 Newton–Schulz 场景中，常见原因是：
- kernel 本身 compute-bound，memory overlap 收益有限；
- super-tile 串行化 + barrier 增多，拉长关键路径。  

详见：
- `DOCS/NEWTON_SCHULZ_OPT_LOG.md`
- `DOCS/NEWTON_SCHULZ_PINGPONG_REPORT.md`

---

## 12. 文档导航

- 新算子模板：`DOCS/ADD_NEW_OPERATOR.md`
- 牛顿算子流程：`DOCS/NEWTON_SCHULZ_OP_GUIDE.md`
- 牛顿优化迭代日志：`DOCS/NEWTON_SCHULZ_OPT_LOG.md`
- ping-pong 分析：`DOCS/NEWTON_SCHULZ_PINGPONG_REPORT.md`
- MMSE/缩放实验：`DOCS/MMSE_SCALING_BASELINE.md`

---

## 13. 上游来源声明与许可证合规

### 13.1 上游来源（Attribution）

本项目基于上游仓库 **ONNXim** 进行二次开发与工程化扩展：

- 上游仓库：`https://github.com/PSAL-POSTECH/ONNXim`
- 本仓库定位：在 ONNXim 的周期级仿真框架基础上，扩展通信相关算子/NPU 工作流（如 Newton–Schulz、MMSE、Series-Inverse、MatMul 等）

为保证可追溯性，本仓库保留了对上游代码来源的说明与许可证文本。

### 13.2 许可证合规说明（MIT）

本仓库沿用并保留 `LICENSE`（MIT License）文本。根据 MIT 许可证要求：

1. 任何复制、分发或再发布的软件副本中，均需包含原版权声明与许可证声明；
2. 软件按 “AS IS” 提供，不附带任何明示或默示担保；
3. 对本仓库新增/修改部分的贡献，不改变上游代码既有版权与许可条款。

### 13.3 第三方组件说明

`extern/` 目录包含多个第三方子组件（如 `booksim`、`protobuf`、`ramulator*` 等），其许可证可能与主仓库不同。使用、分发或二次发布时，请分别查阅对应子组件目录内的许可证/声明文件并遵循其条款。

---


