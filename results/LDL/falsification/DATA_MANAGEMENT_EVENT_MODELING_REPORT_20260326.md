# 仿真数据管理事件建模报告（依赖 / 流动 / 内存 / 总线竞争）

## 1. 目的与范围

本文解释 ONNXim 中“数据管理相关事件”是如何在周期仿真里被建模的，重点覆盖：

1. 数据依赖（RAW/可发射条件）
2. 数据流动（Core ↔ ICNT ↔ DRAM）
3. 内存管理（SPAD/ACC SPAD 分配、有效位、请求计数）
4. 总线与通道竞争（仲裁、队列、带宽占用）

主要代码路径：

- `src/Simulator.cc`
- `src/Core.cc`, `src/SystolicWS.cc`
- `src/Sram.cc`
- `src/Interconnect.cc`
- `src/Dram.cc`
- `src/TraceLogger.cc`, `src/TraceLogger.h`
- `src/scheduler/Scheduler.cc`

---

## 2. 全局周期推进框架（事件触发主循环）

### 2.1 多时钟域推进

在 `Simulator::cycle()` 中，通过 `set_cycle_mask()` 比较 `_core_time / _dram_time / _icnt_time`，决定本次是否执行：

- Core 子循环（`CORE_MASK`）
- DRAM 子循环（`DRAM_MASK`）
- ICNT 子循环（`ICNT_MASK`）

这使 Core/DRAM/ICNT 能以不同频率推进（见 `Simulator` 构造时 `*_period` 计算）。

### 2.2 停机与保护

- 运行条件：`Simulator::running()` 聚合 `models/core/icnt/dram/scheduler` 的 busy 状态。
- 安全保护：`ONNXIM_MAX_CORE_CYCLES` 在 `Simulator::cycle()` 内检查，防止无进展死循环。

---

## 3. 数据依赖建模（能不能发射）

## 3.1 指令级 RAW 依赖

`Core::can_issue_compute()` 是核心依赖检查点：

- 遍历 `inst->src_addrs`
- 若 `src_from_accum=true` 且地址在 `ACCUM_SPAD_BASE`，检查 `_acc_spad.check_hit(...)`
- 否则检查 `_spad.check_hit(...)`

只要有一个源未命中（未 valid），该计算指令不能进入执行流水。

## 3.2 结构资源约束（不是数据依赖但同样阻塞）

`SystolicWS::can_issue_compute()` 进一步叠加资源限制：

- `GEMM/GEMM_PRELOAD`：受 `_compute_pipeline.size()` 上限约束
- Vector 指令：当 `_vector_pipeline` 非空时不再发新 vector 指令（串行）

因此“发不出去”可能来自：

1. 源数据未 ready（RAW）
2. 对应执行流水占满（结构冲突）

## 3.3 barrier 的建模方式

算子会插入 `Opcode::PIPE_BARRIER`（例如 `LDL_BARRIER_*` / `CHOL_NB_BARRIER_*`）。

在 `SystolicWS::get_vector_compute_cycles()` 中，`PIPE_BARRIER` 记为 1 周期 Vector 指令，从而：

- 可在 trace 中显式可见
- 强制执行序列分段
- 不引入真实算术但会占用 vector 发射时隙

---

## 4. 数据流动建模（从指令到内存往返）

## 4.1 从 Tile 到 Core 队列

`Core::cycle()` 会从 tile front 指令按 opcode 分流：

- `MOVIN` → `_ld_inst_queue`
- `MOVOUT` → `_st_inst_queue`
- 其他计算类 → `_ex_inst_queue`

tile 完成条件：`instructions.empty() && inst_finished`。

## 4.2 Load 请求生成（MOVIN）

在 `Core::handle_ld_inst_queue()`：

1. 对目标 SPAD 调 `prefetch(dest_addr, allocated_size, count)` 预分配条目
2. 为每个 DRAM 地址生成 `MemoryAccess`（`request=true, write=false`）
3. 入 `_request_queue`

关键点：`count = front->size`（请求片数），后续每收到一个返回包会 `remain_req_count--`，直到 0 才 valid。

## 4.3 Store 请求生成（MOVOUT）

在 `Core::handle_st_inst_queue()`：

- 只有当源地址 `check_hit` 成功才发写回请求
- 每个写请求 `write=true, request=true`
- `_waiting_write_reqs++`，直到回包在 `push_memory_response()` 中回减

## 4.4 Core ↔ ICNT ↔ DRAM 搬运

`Simulator::cycle()` 的 ICNT 分支中按顺序执行：

1. Core 请求入网（`_icnt->push(...)`）
2. 网内响应回 Core（`_cores[core]->push_memory_response(...)`）
3. ICNT 到 DRAM（`_dram->push(...)`）
4. DRAM 返回到 ICNT（`_dram->top/pop` + `_icnt->push(...)`）

这是完整的请求-响应双向闭环。

---

## 5. 内存管理建模（SPAD 状态机）

`Sram` 不是字节级数据模拟，而是“块分配 + 有效位 + outstanding 请求计数”模型。

## 5.1 分配与容量

`Sram::prefetch()`：

- 检查剩余容量（`check_remain`）
- 建立 `SramEntry{valid=false, size, remain_req_count=count}`
- `_current_size` 累加

## 5.2 有效位推进

- `Sram::fill()`：每回一个包，`remain_req_count--`；降到 0 时置 `valid=true`
- `Sram::count_up()`：对已有目标做“写后失效/重计数”语义（被覆盖写入前先 invalid）

## 5.3 双缓冲

`Core::issue()` 中按 layer/fused-op 与 `accum` 标志切换 `spad_id/accum_spad_id`，并在切换时 `flush`，形成双缓冲语义。

---

## 6. 总线竞争与互连竞争建模

## 6.1 SimpleInterconnect（轻量仲裁模型）

`SimpleInterconnect` 使用 `in_buffers[src][dest]` + `out_buffers[dest]`：

- 每个 `(src,dest)` 独立 FIFO，入队时附 `finish_cycle`
- `cycle()` 对每个 `dest` 做轮询仲裁（RR，`_rr_next_src`）
- 每周期每个 `dest` 仅可从一个 `src` 出队一次

这体现了：

- 目的端口争用（多个源争同一目的）
- 争用导致的排队延迟（finish_cycle 串行推进）

注：`is_full()` 当前返回 `false`，即未建模有限 buffer 溢出背压。

## 6.2 Booksim2Interconnect（细粒度网络竞争）

`Booksim2Interconnect` 将请求映射为 READ/WRITE/REPLY 包，交给 Booksim2：

- `is_full()` 由 Booksim 队列状态给出
- `run()` 内部进行拓扑路由/仲裁/冲突建模

若使用该模式，总线/NoC 竞争会更真实。

## 6.3 DRAM 通道竞争

- `Dram::get_channel_id()` 对地址做哈希映射到 channel
- `DramRamulator/DramRamulator2::is_full()` 由后端内存模型反馈可接收性
- Ramulator 后端内部处理 bank/row 冲突、时序约束

日志里的 `BW utilization` 即通道层面的吞吐利用率观测。

---

## 7. 事件日志（Trace）是如何记的

## 7.1 计算事件

- Cube：`Core::finish_compute_pipeline()` 记录 `TraceLogger::log_event("CoreX_Cube", inst_id, start, finish)`
- Vector：`Core::finish_vector_pipeline()` 记录 `CoreX_Vector`

## 7.2 等待事件

在 `SystolicWS::cycle()`，当 GEMM 由于流水排队导致 `front->start_cycle > _core_cycle`，记录：

- 单元：`CoreX_Wait`
- 名称：`CubeWait`

## 7.3 内存事件

在 `Core::push_memory_response()`：

- 读响应：`CoreX_MTE2`, name=`Load`
- 写响应：`CoreX_MTE3`, name=`Store`

时间区间：`[request.start_cycle, current_core_cycle]`，表示从 core 发起到 core 消费响应的总往返时延。

---

## 8. 结合一次请求的端到端流程（代码级）

以一条 `MOVIN` 触发的读为例：

1. `Operation::initialize_instructions` 产生 `MOVIN`
2. `Scheduler` 将 tile 分发到 core（`Scheduler::get_tile`）
3. `Core::cycle()` 发射到 `_ld_inst_queue`
4. `Core::handle_ld_inst_queue()` 预分配 SPAD，并生成 `MemoryAccess` 入 `_request_queue`
5. `Simulator::cycle()` 把请求送入 ICNT，再送 DRAM
6. DRAM 返回后经 ICNT 回 core
7. `Core::push_memory_response()` 调 `Sram::fill()` 更新 valid
8. 当 `remain_req_count` 归零，源地址可被 `check_hit` 命中
9. 依赖该地址的计算指令在 `can_issue_compute()` 通过，进入执行流水
10. `finish_*_pipeline` 完成时写 Trace 事件

---

## 9. 当前模型的能力边界（解释结果时应注意）

1. `SimpleInterconnect::is_full=false`，无显式 buffer backpressure；
2. `Interconnect::running()` 在 simple/booksim2 两个实现都返回 `false`，生命周期主要由 core/scheduler/dram 驱动；
3. memory trace（MTE2/MTE3）是“端到端往返时延”而非拆分到 NoC/DRAM 的独立区间。

这不影响相对对比趋势，但在做“绝对网络拥塞归因”时需谨慎。

---

## 10. 结论

ONNXim 的数据管理事件建模是“**状态机 + 队列 + 周期推进 + 事件采样**”的组合：

- 依赖：由 SPAD valid/计数和流水占用共同决定是否可发射；
- 流动：请求在 Core→ICNT→DRAM→ICNT→Core 双向闭环中推进；
- 内存：SPAD 用 `remain_req_count` 建模多包到齐语义；
- 竞争：由 ICNT 仲裁、DRAM 可接收性与后端内存模型共同体现。

因此，trace 中的 `Cube/Vector/Wait/MTE2/MTE3` 事件可以直接映射回上述代码路径，用于做性能归因与建模一致性验证。
