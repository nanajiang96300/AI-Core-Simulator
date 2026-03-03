# Newton–Schulz 矩阵求逆算子与 ONNXim 执行流程详解

> 适用于当前仓库中的 `NewtonSchulzOp` / `NewtonSchulzModel` 与 Ascend 910B (24 核) 配置

本说明文档面向未来在 ONNXim 中开发 / 调试矩阵类算子的同学，系统性说明：

- Newton–Schulz 矩阵求逆算子的结构和执行流程；
- 算子依赖的工程内组件（库 / 模块）及其作用；
- ONNXim 整体是如何从模型到 tile、从 tile 到 Core、从 Core 到 DRAM / NOC 逐层运作的（这一部分会写得比较细）。

---

## 1. Newton–Schulz 算子总体结构

相关文件：

- `src/operations/NewtonSchulzOp.cc` / `NewtonSchulzOp.h`
- `src/models/NewtonSchulzModel.cc`
- 配置：`configs/ascend_910b_quiet.json`
- 示例模型：`example/newton_schulz_test.json`

### 1.1 算子目标

该算子实现 Newton–Schulz 迭代法对矩阵进行求逆近似：

- 给定 $A \in \mathbb{R}^{N\times N}$，初始近似 $X_0$；
- 多轮迭代：
  $$
  R_k = 2I - A X_k\\
  X_{k+1} = X_k R_k
  $$
- 最终输出 $X_{K}$ 作为 $A^{-1}$ 的近似。

在本仓库的 C++ 测试中：

- $N = 32$，矩阵为 32×32；
- 默认迭代次数 `iterations = 10`；
- 支持 batch 维度：输入张量为 `[B, N, N]`，一次可处理多个矩阵；
- 针对 `Ascend 910B` 配置（24 个 `systolic_ws` 核）进行体系结构仿真。

### 1.2 NewtonSchulzOp 的构造与形状推断

构造函数核心逻辑（简化）：

```cpp
NewtonSchulzOp::NewtonSchulzOp(...)
  : Operation(config, model, node_proto/name, target_core) {
  _optype = "NewtonSchulz";
  parse_attributes();        // 读 iterations / batch_size
  infer_shapes_from_model(); // 从输入 tensor dims 推断 batch / 矩阵形状
}
```

- `parse_attributes()`：
  - 从 `_attributes` 中读取：
    - `iterations`：迭代次数，默认 10；
    - `batch_size`：batch 数，若未指定则用 `DEFAULT_BATCH_SIZE=96`；
- `infer_shapes_from_model()`：
  - 若 `_matrix_shape` 为空且有 `_model`：
    - 读取第 0 个输入 tensor 的维度 `dims`；
    - 若 `dims == [B, N, K]`（3 维），则：
      - `_batch_size = B`；
      - `_matrix_shape = {N, K}`；
    - 否则 `_matrix_shape = dims`。

**注意：** 对 Newton–Schulz C++ 模型来说，`model` 会创建 3D tensor `[batch_size, 32, 32]`，因此 `_batch_size` 和 `_matrix_shape` 均来自这里，而不是硬编码。

### 1.3 Tile 划分：按 batch 维度生成 tile

`initialize_tiles()` 是算子与 scheduler 之间的关键接口：

```cpp
void NewtonSchulzOp::initialize_tiles(MappingTable&)
{
  std::vector<int> core_load(_config.num_cores, 0);

  for (uint32_t b = 0; b < _batch_size; ++b) {
    uint32_t assigned_core = b % _config.num_cores; // Round-Robin 分配核心

    auto tile = std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED,
      .optype = _optype,
      .layer_id = _id,
      .batch = b,
      .core_id = static_cast<int>(assigned_core),
      ...
    });

    initialize_instructions(tile.get(), Mapping{});

    if (!tile->instructions.empty()) {
      _tiles.push_back(std::move(tile));
      core_load[assigned_core]++;
    }
  }

  spdlog::info("NewtonSchulzOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
}
```

要点：

- **tiling 维度：batch**
  - 对于输入 `[B, N, N]`，生成 `_batch_size = B` 个 tile；
  - 每个 tile 对应 **一个 batch 的完整矩阵**（没有再做行/列子块划分）。

- **多核分配：Round-Robin**
  - `core_id = batch_id % num_cores`；
  - 在 24 核、B=96 的场景下，每核 4 个 tile，负载大致均衡；

- 对每个 tile 调用 `initialize_instructions()` 填充指令流，随后加入 `_tiles` 列表，供 scheduler 调度。

这一步相当于把“有 B 组矩阵要做 Newton–Schulz”这件事，拆成 B 份可独立调度的工作单元（tile），从而摆脱早期“一个大 tile 内部自己循环 B 次、scheduler 看不到 batch”的伪并行问题。

### 1.4 指令生成：Load / Compute / Store 三阶段

`initialize_instructions(Tile* tile, ...)` 负责为“第 b 个 batch 的 tile”生成完整指令序列。

1. **解析矩阵大小与 batch 偏移**

```cpp
const uint32_t N = _matrix_shape[...-2];
const uint32_t K = _matrix_shape[...-1];

addr_type matrix_size_bytes = (addr_type)N * K * _config.precision;
addr_type batch_offset = (addr_type)tile->batch * matrix_size_bytes;

addr_type a_base  = get_operand_addr(_INPUT_OPERAND+0) + batch_offset;
addr_type x_base  = get_operand_addr(_INPUT_OPERAND+1) + batch_offset;
addr_type c_base  = get_operand_addr(_INPUT_OPERAND+2);          // 广播 2I
addr_type out_base= get_operand_addr(_OUTPUT_OPERAND+0) + batch_offset;
```

- `matrix_size_bytes`：一张矩阵的字节数；
- `batch_offset`：当前 batch 在 DRAM 中的偏移；
- A / X / X_out 根据 batch 加偏移；C 作为广播常量不加偏移。

2. **SPAD / ACCUM_SPAD 内部布局**

```cpp
addr_type addr_A = SPAD_BASE;
addr_type addr_X = addr_A + matrix_size_bytes;
addr_type addr_C = addr_X + matrix_size_bytes;
addr_type addr_R = addr_C + matrix_size_bytes;
addr_type addr_T = ACCUM_SPAD_BASE;
```

- 每个 tile 独占自己 core 的 SPAD 区间；
- A / X / C / R / T 在 SPAD / ACCUM_SPAD 内顺序排布。

3. **Load Phase（MTE2 / 红块）**

使用 `emit_movin_full` 辅助函数：

```cpp
auto emit_movin_full = [&](addr_type dram_base, addr_type spad_dest,
                           uint32_t operand_id) {
  std::set<addr_type> addrs;
  for (uint32_t r = 0; r < N; ++r) {
    for (uint32_t c = 0; c < K; c += elems_per_access) {
      ... // 计算每一行的列起始地址
      addrs.insert(dram_base + off);
    }
  }
  tile->instructions.push_back(MOVIN 指令...);
};

// 依次加载 A, X_init, C
emit_movin_full(a_base, addr_A, _INPUT_OPERAND+0);
emit_movin_full(x_base, addr_X, _INPUT_OPERAND+1);
emit_movin_full(c_base, addr_C, _INPUT_OPERAND+2);

// MTE2->Cube barrier
PIPE_BARRIER("NS_BARRIER_MTE2CUBE", type=1);
```

效果：

- 对当前 batch 的整块矩阵，按行遍历生成 DRAM 地址集合；
- 一条 MOVIN 指令对应一批离散地址（多次 DRAM 读）；
- 完成三次 MOVIN 后通过 PIPE_BARRIER 同步到 Cube。

4. **Compute Phase（Cube + Vector / 绿 + 浅绿）**

```cpp
for (uint32_t iter = 0; iter < _iterations; ++iter) {
  // T = A * X
  GEMM_PRELOAD("NS_T", dest=addr_T, src={addr_A, addr_X}, tile_m=N, tile_k=K, tile_n=K);
  PIPE_BARRIER("NS_BARRIER_CUBE2VEC", type=2);

  // R = C - T
  ADD("NS_R", dest=addr_R, src={addr_C, addr_T}, compute_size=N*K, src_from_accum=true);
  PIPE_BARRIER("NS_BARRIER_VEC2CUBE", type=3);

  // X_new = X * R
  GEMM_PRELOAD("NS_X", dest=addr_T, src={addr_X, addr_R}, tile_m=N, tile_k=K, tile_n=K);
}
```

- GEMM_PRELOAD：交给 CubeCore（矩阵阵列）执行 32×32×32 的矩阵乘；
- ADD：交给 VectorCore 执行 1024 元素的逐元素运算；
- 两个 PIPE_BARRIER 分别负责 Cube→Vector、Vector→Cube 的同步。

5. **Store Phase（MTE3 / 蓝块）**

```cpp
std::set<addr_type> out_addrs;
for (uint32_t r = 0; r < N; ++r) {
  for (uint32_t c = 0; c < K; c += elems_per_access) {
    ...
    out_addrs.insert(out_base + off);
  }
}

MOVOUT("NS_OUT", src=addr_T, dest=out_addrs, last_inst=true, barrier_type=4);
```

- 与 Load 相反，聚集一批 DRAM 目标地址，发一条 MOVOUT 指令；
- `last_inst=true` 明确标记该 tile 的生命周期结束，Core 收到后会将 tile 状态置为 FINISH 并通知 scheduler。

至此，一个 tile（一个 batch 的整块矩阵）完成了 **Load → 多轮 Compute → Store** 的完整生命周期。

---

## 2. 算子依赖的工程组件与作用

Newton–Schulz 算子本身很薄，真正的执行能力来自 ONNXim 的一系列基础组件。这里按依赖关系梳理算子用到了哪些模块、各自负责什么。

### 2.1 `Operation` 基类

- 文件：`src/operations/Operation.h`（此处不展开代码）。
- NewtonSchulzOp 继承自 Operation：
  - 统一管理：
    - `_id`（layer id）、`_name`、`_optype`；
    - `_inputs` / `_outputs` 张量 id；
    - `_tiles`：当前算子生成的所有 tile；
    - `_attributes`：来自 ONNX 节点或 C++ 模型配置的属性表；
  - 提供帮助函数：
    - `get_operand_addr()`：根据 operand 类型和 index 查询 DRAM 基地址；
    - `make_address()`：根据 (row, col, ...) 计算 tensor 内偏移。

NewtonSchulzOp 利用这些接口完成：
- 从 tensor / 节点配置中拿到矩阵形状、基地址；
- 用 `_tiles` 存放所有 batch tile，暴露给 scheduler。

### 2.2 `Model`：算子与张量的容器

- 文件：`src/Model.h` / `src/Model.cc`，以及专用 `NewtonSchulzModel.cc`。
- 作用：
  - 解析 ONNX 模型或 C++ 描述，创建 `Tensor` 和 `Operation`；
  - 维护 `_operation_map`（id → op）、`_tensor_map`（id → tensor）；
  - 维护 `_executable_layer`：当前可执行的一批 op（通常是一层）；
  - 在运行时被 scheduler 查询“下一层要执行哪个 op”。

Newton–Schulz 的专用模型 `NewtonSchulzModel`：
- 不依赖 ONNX 文件，而是：
  - 根据 `example/newton_schulz_test.json` 中的 `batch_size` / `iterations` 构造 3D tensors `[B, 32, 32]`；
  - 创建唯一一个 `NewtonSchulzOp`，绑定这四个 tensor（A/X_init/C/X_out）；
  - 调用 `op->initialize_tiles(_mapping_table)` 生成 B 个 tile；
  - 把该 op 塞进 `_operation_map` 和 `_executable_layer` 中。

### 2.3 `Core`：单个 AI Core 的执行模拟

- 文件：`src/Core.h` 及其子类（如 `SystolicWS.cc`）。
- 负责模拟一个物理 AI Core 上的：
  - Tile 队列 `_tiles`，`issue()` / `cycle()` / `pop_finished_tile()`；
  - Compute pipeline：
    - `_compute_pipeline`：Cube 指令队列（GEMM_PRELOAD 等）；
    - `_vector_pipeline`：Vector 指令队列（ADD 等）；
  - Load / Store pipeline：
    - `_ld_inst_queue`：MTE2（Load）指令队列（MOVIN 等）；
    - `_st_inst_queue`：MTE3（Store）指令队列（MOVOUT 等）；
  - DRAM 接口：
    - `_request_queue` / `_response_queue`：与 DRAM/NOC 之间的 MemoryAccess 流；

流程上：
- `issue(tile)`：从 scheduler 取出一个 `Tile`，将其所有 `Instruction` 放入内部各队列；
- `cycle()`：每个时钟周期：
  - 驱动 compute / vector / ld / st 四条 pipeline 前进一步；
  - 根据 `Instruction` 类型发起 DRAM 请求 / 处理响应；
  - 更新统计数据（matmul active cycles、vector active cycles 等）；
  - 当一个 tile 所有指令执行完、且 `last_inst` 触发时，将 tile 放入 `_finished_tiles`，供外部收集。

Newton–Schulz 的 tile 在 core 内经历的，就是这四条流水线的交错执行；TraceLogger 监控的就是这些 pipeline 上的 active 区间，并最终写成 CSV。

### 2.4 `Dram` / `Interconnect`：内存与片上网络

- `src/Dram.h` / `Dram.cc` / `DramRamulator*.cc`：
  - 模拟后端 DRAM（简单模型、Ramulator1、Ramulator2 HBM2 等）；
  - 维护多个 channel 的队列、row hits/misses/conflicts 统计；
  - 提供 `push()` / `pop()` / `cycle()` 接口。

- `src/Interconnect.h` / `Interconnect.cc` / `Booksim2Interconnect.cc`：
  - 模拟 core ↔ DRAM 之间的片上网络（simple / booksim2）；
  - 按 port 转发 MemoryAccess，参与 back‑pressure 与带宽统计。

Newton–Schulz 算子通过 Core 的 load/store 指令间接访问 DRAM：
- MOVIN / MOVOUT → Core 发起 `MemoryAccess` → Interconnect → Dram；
- DRAM 响应再通过 Interconnect 回到 Core，驱动后续 pipeline。

### 2.5 `Scheduler`：跨 core 调度 tile

- 文件：`src/Scheduler.h` / `Scheduler.cc`（未展开）。
- 主要职责：
  - 从 `Model::_executable_layer` 中取出当前可执行 op；
  - 将每个 op 的 `_tiles` 分配到各个 core 的待发队列；
  - 提供：
    - `get_tile(core_id)`：为某核弹出一个待执行 tile；
    - `finish_tile(core_id, layer_id)`：当 core 报告某 tile 完成时，更新该 layer 状态；
    - `empty()`：判断是否还有待调度工作。

对于 Newton–Schulz：
- 由于 `NewtonSchulzModel` 只注册了一个 op，scheduler 看到的 executable layer 就是单层；
- 该 op 内的 96 个 tile 经 `initialize_tiles()` 预先分配了 `core_id`，scheduler 会按照 RR 顺序把 tile 发往对应 core；
- 这就是为什么在 96‑batch 实验中，24 核的时间线形状非常对称 —— 调度粒度是 tile 而不是整层。

### 2.6 `Simulator`：整个系统的时钟驱动器

- 文件：`src/Simulator.cc`。
- 核心成员：
  - `_cores`：`std::vector<std::unique_ptr<Core>>`，所有 AI Core；
  - `_dram`：DRAM 模型；
  - `_icnt`：Interconnect 模型；
  - `_scheduler`：硬件调度器；
  - `_models`：待执行的 `Model` 堆；
  - `_core_cycles` / `_core_time` / `_dram_time` / `_icnt_time`：三个子系统的本地时间；

初始化时：

```cpp
Simulator::Simulator(SimulationConfig config, bool language_mode)
  : _config(config), _core_cycles(0), _language_mode(language_mode) {
  // 1. 创建 DRAM (Simple / Ramulator1 / Ramulator2)
  // 2. 创建 Interconnect (Simple / Booksim2)
  // 3. 创建所有 Core (SystolicWS / SystolicOS)
  // 4. 创建 Scheduler
  // 5. 初始化 Model 堆（空）
}
```

主执行循环：

```cpp
void Simulator::run_simulator() {
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::cycle() {
  // 从环境变量读取可选 watchdog：ONNXIM_MAX_CORE_CYCLES
  while (running()) {
    set_cycle_mask();    // 确定本轮谁 advance：Core / DRAM / ICNT

    if (_cycle_mask & CORE_MASK) {
      handle_model();    // 把到时的 Model 丢给 Scheduler

      for (core_id in cores) {
        // 1. 取走 core 完成的 tile，通知 Scheduler
        // 2. 如果 Scheduler 有 tile 且 core 能接，则 issue 一个 tile
        // 3. core.cycle()：推进 pipeline
      }
      _core_cycles++;
    }

    if (_cycle_mask & DRAM_MASK) _dram->cycle();
    if (_cycle_mask & ICNT_MASK) { 转发 MemoryAccess，_icnt->cycle(); }

    // 如果超出 watchdog 上限则中止
  }

  // 打印统计、写 Trace CSV
}
```

其中 `handle_model()` 会在合适的仿真时间点：

- 将用户通过命令行参数 `--mode` 指定的 `Model`（如 `NewtonSchulzModel`）`initialize_model()`；
- 调用 `_scheduler->schedule_model(...)` 注册到调度器；
- 之后的 tile 分发与回收都在 scheduler/core 之间进行。

运行结束后：

- 通过 `Core::print_stats()` 打印每核的 matmul/向量/idle 等统计；
- 通过 DRAM / Interconnect 的 `print_stat()` 打印带宽利用情况；
- 若设置了 `ONNXIM_TRACE_CSV` 环境变量，则 `TraceLogger::dump_to_csv()` 将所有 unit 的事件写进 CSV，可用 `visualizer_png.py` 绘制 pipeline 图。

---

## 3. ONNXim 如何“从模型到硬件”一步步运作（详细版）

这节把上面的组件串起来，从用户视角讲一遍完整路径：

> **配置 + 模型描述 → Model 初始化 → Operation / Tile 构造 → Scheduler 调度 → Core pipeline 执行 → DRAM / NOC 交互 → Trace / 统计输出**。

### 3.1 用户入口：命令行与配置

典型运行：

```bash
ONNXIM_TRACE_CSV=results/newton_schulz/910b/profiling_log_newton_910b_batch96.csv \
ONNXIM_MAX_CORE_CYCLES=5000000 \
./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/newton_schulz_test.json \
  --mode newton_schulz_test \
  --log_level info
```

- `--config`：硬件配置（core 数、阵列尺寸、频率、DRAM 类型等）；
- `--models_list`：要加载的模型列表 JSON（这里指向 Newton–Schulz 的 C++ 模型描述）；
- `--mode newton_schulz_test`：选择 `NewtonSchulzModel` 这条代码路径；
- 环境变量：
  - `ONNXIM_TRACE_CSV`：指定 profiling CSV 输出路径；
  - `ONNXIM_MAX_CORE_CYCLES`：可选 watchdog 上限。

### 3.2 Model 初始化过程

当仿真时间到达模型的 request time 时，`Simulator::handle_model()` 会：

1. 创建一个 `NewtonSchulzModel` 实例；
2. 调用 `NewtonSchulzModel::initialize_model(weight_table)`：
   - 根据 `example/newton_schulz_test.json` 中的配置：
     - `name`: `"newton_schulz_32x32"`
     - `batch_size`: 如 96
     - `attributes.iterations`: `"10"`
   - 构造 4 个 3D tensor：[B,32,32] 的 A / X_init / C / X_out；
   - 构造一个 `NewtonSchulzOp`：
     - 传入 `attributes`（含 iterations / batch_size）；
     - 关联这 4 个 tensor 作为输入/输出；
     - 调用 `NewtonSchulzOp::initialize_tiles()` 生成 B 个 tile；
   - 将该 op 加入 `_operation_map`，并将其指针放入 `_executable_layer`。
3. 调用 `_scheduler->schedule_model(std::move(model), 1)`，把 `Model` 交给调度器管理。

此时，调度器对 Newton–Schulz 的理解是：
- 有一个 model `newton_schulz_32x32`；
- 这个 model 只有一个 layer（`NewtonSchulzOp`）；
- 该 layer 内预先准备好了 B 个 tile，每个 tile 标好了 batch 和 target core。

### 3.3 Scheduler 如何把 tile 塞进 Core

在每个 CORE 周期：

1. `Scheduler` 先检查是否有 model 准备好下一层（本例中只有一个 layer，很快就就绪）；
2. 对每个 core：
   - 查询 core 是否有已完成的 tile：`core->pop_finished_tile()`；
     - 若 tile.status == FINISH，则调用 `scheduler->finish_tile(core_id, tile.layer_id)` 更新 layer 完成度；
   - 如果 scheduler 还有待发的 tile 且 core 可以接任务：

     ```cpp
     if (!_scheduler->empty()) {
       is_accum_tile = _scheduler->is_accum_tile(core_id, 0);
       if (_cores[core_id]->can_issue(is_accum_tile)) {
         std::unique_ptr<Tile> tile = _scheduler->get_tile(core_id);
         if (tile->status == Tile::Status::INITIALIZED) {
           _cores[core_id]->issue(std::move(tile));
         }
       }
     }
     ```

3. 当该 layer 的所有 tile 都报告 FINISH 后，Scheduler 认为这一层完成；
4. 由于 Newton–Schulz 只有这一层，model 随即标记为完成，`running()` 不再将其视为 busy。

### 3.4 Core 内部如何执行一个 tile

当 `core.issue(tile)` 被调用时：

1. Core 将 tile 里的 `instructions` 根据类型分发到各自的队列：
   - MOVIN → `_ld_inst_queue`；
   - MOVOUT → `_st_inst_queue`；
   - GEMM_PRELOAD → `_compute_pipeline`；
   - ADD → `_vector_pipeline`；
   - PIPE_BARRIER → 插入到对应 pipeline，配合状态机控制；
2. 之后每个 `core.cycle()` 会：
   - 优先处理 ld/st 队列：生成/接收 MemoryAccess，与 DRAM/NOC 交互；
   - 根据当前 pipeline 可用性和依赖，弹出下一条 compute / vector 指令执行；
   - 更新内部统计计数（matmul active cycles、vector active cycles、memory idle cycles 等）；
   - 当检测到本 tile 的 `last_inst` 执行完，会：
     - 将 tile 的 `status` 置为 FINISH；
     - 推入 `_finished_tiles`，等待外部 `pop_finished_tile()` 取走。

### 3.5 DRAM / Interconnect 与 Core 的交互

在 ICNT 周期和 DRAM 周期，Simulator 会：

1. 对每个 core 的每个注入端口：
   - 若 core 有 memory request：从 core 的 `_request_queue` 取一个 `MemoryAccess`，尝试送入 `_icnt`；
   - 若该端口有来自 `_icnt` 的响应：送回 `core->push_memory_response()`；
2. 对每个 DRAM channel：
   - 从 `_icnt` 收请求送入 `_dram`；
   - 从 `_dram` 收响应送回 `_icnt`；
3. 分别调用 `_icnt->cycle()` 和 `_dram->cycle()` 推进网络和内存控制器状态；
4. 定期根据收发请求数和 `dram_req_size` 打印带宽利用率统计。

Core 的 load/store pipeline 会根据 MemoryAccess 完成情况决定下一步：
- Load 的数据到达后，SPAD 中对应区域被标记为 ready，可触发后续 GEMM；
- Store 的请求完成后，tile 的写回状态更新；
- 所有相关任务完成时，tile 才能真正 FINISH。

### 3.6 TraceLogger 与可视化

- Core / DRAM / Interconnect 在关键事件（指令开始 / 结束、MemoryAccess 请求 / 响应）处调用 `TraceLogger::instance().log(...)` 记录：
  - `unit`（如 `Core0_Cube` / `Core1_MTE3`）；
  - `name`（指令名，如 `NS_T` / `NS_R` / `NS_OUT`）；
  - `start_cycle` / `end_cycle`。
- 仿真结束时，如果设置了 `ONNXIM_TRACE_CSV`，Simulator 会：

  ```cpp
  const char* trace_env = std::getenv("ONNXIM_TRACE_CSV");
  if (trace_env != nullptr) {
    TraceLogger::instance().dump_to_csv(out_path);
  }
  ```

- 之后可使用 `visualizer_png.py` 快速渲染：

  ```bash
  python3 visualizer_png.py \
    -i results/newton_schulz/910b/profiling_log_newton_910b_batch96.csv \
    -o results/newton_schulz/910b/pipeline_newton_910b_batch96.png
  ```

- 脚本会：
  - 正规化列名：Unit / Name / StartCycle / EndCycle；
  - 把 `unit` 拆成 `Core + Engine`：
    - `Cube` → `CubeCore`（绿色）；
    - `Vector` → `VectorCore`（浅绿）；
    - `MTE2` → `MTE2 (Load)`（红）；
    - `MTE3` → `MTE3 (Store)`（蓝）；
  - 对每个 `(Core, Engine)` 组合画一条时间线，展示 Load / Compute / Store 的重叠情况。

---

## 4. 小结

- Newton–Schulz 算子本身只负责两件关键事：
  1. 解析 batch / 矩阵形状与迭代次数；
  2. 为每个 batch 生成独立的 tile，并为每个 tile 生成完整的 Load / Compute / Store 指令序列。

- 算子之上的 `Model` / `Scheduler` / `Core` / `Dram` / `Interconnect` / `Simulator` 则组成了 ONNXim 的执行框架：
  - Model 组织算子与张量；
  - Op 组织 tile；
  - Scheduler 把 tile 分发到多核；
  - Core 用多条 pipeline 执行 tile 指令，并与 DRAM / NOC 通信；
  - Simulator 协调 Core/DRAM/NOC 的时钟、调度 Model，并最终输出 trace 与统计。

- 对 Newton–Schulz 这种 batch‑friendly 的矩阵逆算子，通过 **按 batch 划分 tile + 单 op batched 结构**，可以很好地发挥 ONNXim 的 tile 调度能力，在保证算法语义正确的前提下实现 24 核上的高并行度和可视化友好的执行行为。
