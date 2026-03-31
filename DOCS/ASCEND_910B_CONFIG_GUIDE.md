# `ascend_910b.json` 配置项说明

本文档对应文件：`/project/RAN_Mulator/configs/ascend_910b.json`

目标：解释每个配置项的含义、作用范围与调优影响，便于后续改模型/改硬件参数时快速定位。

---

## 1) 顶层总览

该配置可分为 8 大块：

1. 全局规模与频率（`num_cores`, `core_freq` 等）
2. CUBE 延迟模型（`ascend_cube_model`）
3. Core 微架构参数（`core_config.core_X.*`）
4. DRAM/内存系统参数（`dram_*`）
5. 互连参数（`icnt_*`）
6. 数据表示与布局（`precision`, `layout`）
7. 调度策略（`scheduler`）
8. 片上共享 SRAM（`sram_size`）

---

## 2) 顶层字段逐项说明

### `num_cores`
- 含义：NPU 核心总数。
- 当前值：`24`。
- 影响：并行上限、tile 分发宽度、总吞吐潜力。

### `ascend_cube_model`
用于启用和配置 CUBE（矩阵乘）延迟模型。

#### `ascend_cube_model.enabled`
- 含义：是否启用 Ascend 风格 CUBE 延迟模型。
- 当前值：`true`。

#### `ascend_cube_model.cube_m`
- 含义：单次 CUBE 基本块 M 维。
- 当前值：`16`。
- 影响：与算子 `tile_m` 的匹配效率。

#### `ascend_cube_model.cube_n`
- 含义：单次 CUBE 基本块 N 维。
- 当前值：`16`。

#### `ascend_cube_model.cube_k`
- 含义：单次 CUBE 基本块 K 维。
- 当前值：`16`。

#### `ascend_cube_model.cube_base_latency`
- 含义：CUBE 基础延迟（模型基准项）。
- 当前值：`1`。
- 影响：每条 GEMM/GEMM_PRELOAD 指令周期估算。

### `core_freq`
- 含义：核心频率（通常以 MHz 解释）。
- 当前值：`1200`。
- 影响：cycle 到时间（us/ms）换算。

### `core_print_interval`
- 含义：核心侧统计日志打印周期（按 cycle）。
- 当前值：`10000`。
- 影响：调试日志频度，不影响功能正确性。

### `core_config`
- 含义：每个核心的微架构参数字典。
- 结构：`core_0` 到 `core_23`（共 24 项），每项字段相同。
- 当前文件中 24 个 core 参数一致，表示同构多核。

> 注意：`core_0 ~ core_23` 每项并不是“不同语义字段”，只是“不同核心实例”。真正需要理解的是它们内部的参数模板（见第 3 节）。

### `dram_type`
- 含义：内存后端类型。
- 当前值：`ramulator2`。

### `dram_freq`
- 含义：DRAM 时钟频率（通常 MHz）。
- 当前值：`1200`。

### `dram_channels`
- 含义：DRAM 通道数。
- 当前值：`32`。
- 影响：并行带宽能力、冲突概率。

### `dram_req_size`
- 含义：单次 DRAM 请求粒度（字节）。
- 当前值：`64`。
- 影响：MOVIN/MOVOUT 请求切分方式。

### `dram_latency`
- 含义：DRAM 访问基础延迟（cycle 模型参数）。
- 当前值：`100`。

### `dram_size`
- 含义：DRAM 容量（项目约定单位，常见为 GB 或模型单位）。
- 当前值：`64`。

### `dram_nbl`
- 含义：NoC/DRAM 相关并发级别参数（项目自定义）。
- 当前值：`1`。
- 备注：通常保持默认，除非你在做内存系统模型实验。

### `dram_print_interval`
- 含义：DRAM 统计打印周期（cycle）。
- 当前值：`9600`。

### `dram_config_path`
- 含义：Ramulator2 配置文件路径。
- 当前值：`../configs/ramulator2_configs/HBM2.yaml`。
- 影响：行缓冲策略、时序约束、bank/channel 行为。

### `icnt_type`
- 含义：片上互连类型。
- 当前值：`simple`。

### `icnt_latency`
- 含义：互连基础延迟（cycle）。
- 当前值：`1`。

### `icnt_freq`
- 含义：互连频率（通常 MHz）。
- 当前值：`1200`。

### `icnt_injection_ports_per_core`
- 含义：每个 core 注入互连的端口数。
- 当前值：`8`。
- 影响：高并发下请求注入能力。

### `icnt_config_path`
- 含义：互连拓扑配置路径（Booksim2 配置）。
- 当前值：`../configs/booksim2_configs/fly_c1_m2.icnt`。

### `precision`
- 含义：数据精度字节数（bytes per element）。
- 当前值：`2`（通常对应 FP16/BF16 类精度）。
- 影响：带宽占用、`compute_size` 对应的数据量。

### `layout`
- 含义：张量布局。
- 当前值：`NHWC`。
- 影响：地址生成与某些算子的维度解释。

### `scheduler`
- 含义：调度器策略。
- 当前值：`simple`。
- 影响：层间执行方式、可并行层处理方式。

### `sram_size`
- 含义：全局/共享 SRAM 容量（字节）。
- 当前值：`134217728`（约 128 MB）。

### `dram_bandwidth`
- 含义：DRAM 带宽模型参数（项目定义单位，常与配置频率共同决定有效带宽）。
- 当前值：`1200`。

---

## 3) `core_config.core_X.*` 子字段逐项说明

以下字段在 `core_0 ~ core_23` 中重复出现且含义一致：

### `core_type`
- 含义：核心计算阵列类型。
- 当前值：`systolic_ws`（weight-stationary systolic）。

### `core_width`
- 含义：阵列宽度（PE 列数）。
- 当前值：`16`。

### `core_height`
- 含义：阵列高度（PE 行数）。
- 当前值：`16`。

### `spad_size`
- 含义：该 core 的 SPAD 容量（字节）。
- 当前值：`262144`（256 KB）。

### `accum_spad_size`
- 含义：该 core 的累加缓冲 ACCUM SPAD 容量（字节）。
- 当前值：`262144`（256 KB）。

### `sram_width`
- 含义：本地 SRAM 访问宽度（字节或总线宽参数，依实现解释）。
- 当前值：`64`。

### `vector_process_bit`
- 含义：向量单元单拍处理位宽（bit）。
- 当前值：`2048`。

### `add_latency`
- 含义：向量/标量加法延迟（cycle）。
- 当前值：`1`。

### `mul_latency`
- 含义：乘法延迟（cycle）。
- 当前值：`1`。

### `mac_latency`
- 含义：乘加延迟（cycle）。
- 当前值：`1`。

### `exp_latency`
- 含义：指数运算延迟（cycle）。
- 当前值：`2`。

### `gelu_latency`
- 含义：GELU 运算延迟（cycle）。
- 当前值：`2`。

### `div_latency`
- 含义：除法延迟（cycle）。
- 当前值：`4`。

### `add_tree_latency`
- 含义：加法树归约延迟（cycle）。
- 当前值：`2`。

### `scalar_sqrt_latency`
- 含义：标量平方根延迟（cycle）。
- 当前值：`4`。

### `scalar_add_latency`
- 含义：标量加法延迟（cycle）。
- 当前值：`1`。

### `scalar_mul_latency`
- 含义：标量乘法延迟（cycle）。
- 当前值：`1`。

---

## 4) 常见改动建议（实用）

### A. 想看“算力瓶颈”
优先调：
- `core_width`, `core_height`
- `core_freq`
- `ascend_cube_model.cube_base_latency`

### B. 想看“内存瓶颈”
优先调：
- `dram_channels`
- `dram_req_size`
- `dram_latency`
- `dram_config_path`（HBM timing）

### C. 想看“向量阶段占比”
优先调：
- `vector_process_bit`
- `add_latency`, `div_latency`, `exp_latency`, `gelu_latency`

### D. 想看“互连拥塞”
优先调：
- `icnt_type`
- `icnt_latency`
- `icnt_injection_ports_per_core`
- `icnt_config_path`

---

## 5) 你这份配置的简短画像

- 24 核、每核 16×16 的同构 `systolic_ws`。
- 每核 `SPAD + ACCUM` 各 256KB。
- 向量宽度 2048 bit，标量/向量基本算子延迟较短。
- DRAM 采用 Ramulator2 + HBM2，32 通道。
- 适合做高并行 GEMM 类算子和多 batch 的吞吐实验。

---

## 6) 版本维护建议

建议后续每次改配置时，补一段变更注释（changelog）到文档末尾，例如：

- `2026-03-24`: `dram_req_size 64 -> 128`，用于评估大突发请求对带宽利用率影响。
- `2026-03-24`: `vector_process_bit 2048 -> 1024`，用于观察向量算子瓶颈变化。

这样做可以让实验结果更可追溯。
