# 添加新算子的标准流程（MyNewOperator 模板）

本文基于 ONNXim 当前框架，总结“新增一个算子”的完整流程，并给出一个可复用的模板类 `MyNewOperator`。你可以将其复制改名，快速迭代新算子（例如 ZF 均衡、LS 估计等）。

---

## 1. 必须修改/新增的文件清单

1. `src/operations/MyNewOperator.h`
  - 声明一个继承自 `Operation` 的新算子类。
  - 约定 `optype` 字符串（例如 `"MyNewOp"`），用于复制和调试。
2. `src/operations/MyNewOperator.cc`
  - 实现构造函数、`initialize_tiles` 和 `initialize_instructions`。
  - 将逻辑拆成 `load / compute / store` 三个阶段，分别负责指令生成。
3. **`src/CMakeLists.txt`（构建系统，必改）**
  - 本仓库当前使用 `file(GLOB_RECURSE ...)` 自动抓取 `src/*.cc`，通常会自动包含新文件。
  - 如果你的分支改成了手动 `set(SRC_FILES ...)`，必须把 `operations/MyNewOperator.cc` 加进去，否则会出现“undefined reference to MyNewOperator” 或根本没编译进可执行文件。

  示例：

  ```cmake
  set(SRC_FILES
     # ... 其他源码 ...
     operations/MyNewOperator.cc
  )
  ```

4. `src/operations/OperationFactory.cc`
  - 在 `OperationFactory::create_operation` 中，将 ONNX 中的 `op_type` 字符串映射到新算子类（例如 `"MyNewOp"`）。
  - 在 `OperationFactory::copy_operation` 中，为新算子支持 layer 复制（可选但推荐）。
5. `models/*` 或自定义 `Model`（如 `ChannelModel`）
  - 如果是纯 ONNX 模式，只要 ONNX 图里包含该 op_type，即可自动触发工厂创建。
  - 如果是纯 C++ 构图（类似 `ChannelModel`），需要在 `initialize_model` 中手动构造 `MyNewOperator` 实例并调用 `initialize_tiles`。
6. `example/*.json`（模型列表）
  - 在 `example/models_list.json` 或其他 `*.json` 中添加一个条目，让 `main` 按该配置创建并注册你的新模型/算子。

---

## 2. 模板类：`MyNewOperator` 头文件

文件：`src/operations/MyNewOperator.h`

```cpp
#pragma once

#include "Operation.h"

// Template example for creating a new operator.
// Replace "MyNewOperator" and fields with your own op.
class MyNewOperator : public Operation {
 public:
  // ONNX-based constructor (normal path when loading from an ONNX graph)
  MyNewOperator(SimulationConfig config,
                Model* model,
                onnx::NodeProto& node_proto,
                uint32_t target_core = 0);

  // Attribute-based constructor (used by custom models like ChannelModel)
  MyNewOperator(SimulationConfig config,
                Model* model,
                const std::string& name,
                std::map<std::string, std::string>& attributes,
                uint32_t target_core = 0);

  // Mapping-based constructor (used for unit tests or synthetic runs)
  MyNewOperator(SimulationConfig config,
                MappingTable& mapping_table,
                const std::vector<uint32_t>& input_shape,
                const std::vector<uint32_t>& weight_shape,
                const std::vector<uint32_t>& output_shape,
                uint32_t target_core = 0);

  // 每个具体算子必须实现：根据 Mapping 生成 Tile
  void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  // 为每个 Tile 生成指令（load/compute/store）。
  void initialize_instructions(Tile* tile, Mapping mapping) override;

  // 这三个 helper 不是基类接口，只是推荐结构化写法：
  void plan_tiling(MappingTable& mapping_table, Mapping& mapping_out);
  void emit_load_instructions(Tile* tile, const Mapping& mapping);
  void emit_compute_instructions(Tile* tile, const Mapping& mapping);
  void emit_store_instructions(Tile* tile, const Mapping& mapping);

 private:
  // 缓存形状 / 参数
  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _weight_shape;
  std::vector<uint32_t> _output_shape;

  // 示例属性：是否使用 bias
  bool _use_bias{false};
};
```

---

## 3. 模板类：`MyNewOperator` 源文件

文件：`src/operations/MyNewOperator.cc`

```cpp
#include "MyNewOperator.h"

#include "../Model.h"
#include "../Tensor.h"

MyNewOperator::MyNewOperator(SimulationConfig config,
                             Model* model,
                             onnx::NodeProto& node_proto,
                             uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "MyNewOp";  // 在 factory::copy_operation 中使用

  // TODO: 从 node_proto / Model 中解析输入、权重、输出张量的形状
  // 示例：_input_shape = ...; _weight_shape = ...; _output_shape = ...;
}

MyNewOperator::MyNewOperator(SimulationConfig config,
                             Model* model,
                             const std::string& name,
                             std::map<std::string, std::string>& attributes,
                             uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "MyNewOp";
  // TODO: 从 attributes 读取形状信息
}

MyNewOperator::MyNewOperator(SimulationConfig config,
                             MappingTable& mapping_table,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& weight_shape,
                             const std::vector<uint32_t>& output_shape,
                             uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  _optype = "MyNewOp";
  _input_shape = input_shape;
  _weight_shape = weight_shape;
  _output_shape = output_shape;

  // 在这种构造路径下，你通常需要手动在 Model 里创建 Tensor
  // （可参考 GemmWSTest 或 LSEstimatorOp 的做法）。
}

void MyNewOperator::plan_tiling(MappingTable& mapping_table, Mapping& mapping_out) {
  // 标准写法：根据逻辑维度构造 Mapping::LoopCounts 作为 key
  Mapping::LoopCounts key{
      .N = _output_shape.empty() ? 1u : _output_shape.back(),
      .C = _weight_shape.empty() ? 1u : _weight_shape.front(),
      .M = _weight_shape.size() > 1 ? _weight_shape[1] : 1u,
      .S = 1,
      .R = 1,
      .Q = 1,
      .P = 1,
      .target_core = target_core};

  try {
    mapping_out = mapping_table.at(key);
  } catch (const std::out_of_range&) {
    spdlog::error("[MyNewOp] Mapping key not found: N={} C={} M={} P={} Q={} S={} R={}",
                  key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
}

void MyNewOperator::initialize_tiles(MappingTable& mapping_table) {
  Mapping mapping;
  plan_tiling(mapping_table, mapping);

  int core_id = -1;
  for (uint32_t n = 0; n < mapping.tile_out_loop.N; ++n) {
    for (uint32_t m = 0; m < mapping.tile_out_loop.M; ++m) {
      for (uint32_t c = 0; c < mapping.tile_out_loop.C; ++c) {
        if (c == 0) {
          core_id = (core_id + 1) % _config.num_cores;
        }

        auto tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = _optype,
            .layer_id = _id,
            .batch = static_cast<int>(n),
            .Q = 1,
            .P = 1,
            .M = static_cast<int>(m),
            .C = static_cast<int>(c),
            .S = 1,
            .R = 1,
            .accum = c != 0,
            .core_id = core_id});

        initialize_instructions(tile.get(), mapping);
        if (!tile->instructions.empty()) {
          _tiles.push_back(std::move(tile));
        }
      }
    }
  }
}

void MyNewOperator::emit_load_instructions(Tile* tile, const Mapping& mapping) {
  // 示例：用 MOVIN 把输入/权重从 DRAM 拉到 SPAD
  // addr_type a_base = get_operand_addr(_INPUT_OPERAND + 0);
  // addr_type w_base = get_operand_addr(_INPUT_OPERAND + 1);
  // 使用 make_address(index, dims) 生成 DRAM 地址
  (void)tile;
  (void)mapping;
}

void MyNewOperator::emit_compute_instructions(Tile* tile, const Mapping& mapping) {
  // 示例：下发 GEMM_PRELOAD 或其他计算指令
  // Instruction inst{ .opcode = Opcode::GEMM_PRELOAD, ... };
  (void)tile;
  (void)mapping;
}

void MyNewOperator::emit_store_instructions(Tile* tile, const Mapping& mapping) {
  // 示例：用 MOVOUT 把结果从 ACCUM_SPAD 写回 DRAM
  (void)tile;
  (void)mapping;
}

void MyNewOperator::initialize_instructions(Tile* tile, Mapping mapping) {
  // 统一入口：load -> compute -> store
  emit_load_instructions(tile, mapping);
  emit_compute_instructions(tile, mapping);
  emit_store_instructions(tile, mapping);
}
```

---

## 4. 在 `OperationFactory` 中注册新算子

ONNX 模式下，ONNX 节点的 `op_type` 会通过 `OperationFactory` 映射到具体 C++ 类。

文件：`src/operations/OperationFactory.cc`

1. 头部包含新算子头文件：

```cpp
#include "MyNewOperator.h"
```

2. 在 `OperationFactory::create_operation` 中增加分支：

```cpp
std::unique_ptr<Operation> OperationFactory::create_operation(
    Model* model, onnx::NodeProto& node_proto, uint32_t target_core) {
  if (node_proto.op_type() == "Conv" || node_proto.op_type() == "FusedConv") {
    // ... 现有逻辑 ...
  } else if (node_proto.op_type() == "Gemm" ||
             node_proto.op_type() == "FusedGemm") {
    // ... 现有逻辑 ...
  } else if (node_proto.op_type() == "MyNewOp") {
    // 新增：将 ONNX 中 op_type = "MyNewOp" 映射到 C++ 实现
    return std::make_unique<MyNewOperator>(_config, model, node_proto, target_core);
  }
  // ... 其他分支 ...
}
```

3. （推荐）在 `OperationFactory::copy_operation` 中支持复制：

```cpp
std::unique_ptr<Operation> OperationFactory::copy_operation(Operation* op) {
  if (op->get_optype() == "Conv" || op->get_optype() == "FusedConv") {
    // ...
  } else if (op->get_optype() == "Gemm" || op->get_optype() == "FusedGemm") {
    // ...
  } else if (op->get_optype() == "MyNewOp") {
    return std::make_unique<MyNewOperator>(*dynamic_cast<MyNewOperator*>(op));
  }
  // ... 其他分支 ...
}
```

> 注意：如需支持 `copy_operation`，需要在 `MyNewOperator` 中添加拷贝构造函数，或使用默认生成的拷贝语义（保持成员可拷贝）。

---

## 5. 在 Model / ChannelModel 中使用新算子

### 5.1 ONNX 模式（推荐路径）

1. 在你的 ONNX 图中添加一个节点：
   - `op_type = "MyNewOp"`
   - 输入 / 输出 Tensor 与前后算子连接好。
2. 确保 `OperationFactory::create_operation` 中已经有对应分支。
3. 正常通过 `Model` 加载：

```cpp
std::string onnx_path = fmt::format("{}/{}/{}.onnx", model_base_path, model_name, model_name);
std::string mapping_path = fmt::format("{}/{}/{}.mapping", model_base_path, model_name, model_name);
MappingTable mapping_table = MappingTable::parse_mapping_file(mapping_path, config);

auto model = std::make_unique<Model>(onnx_path, model_config, config, model_name, mapping_table);
```

### 5.2 纯 C++ 构图模式（类似 `ChannelModel`）

以 `ChannelModel` 为例，在 `initialize_model` 中：

```cpp
std::string op_name = name_gen(get_name(), "MyNewOpLayer");
// 1. 先在 Model 里创建 Tensor（**必须**），并绑定到算子

// 示例：一个输入 Tensor 和一个输出 Tensor，实际项目中通常有多个输入
std::vector<uint32_t> in_shape = {32, 64};
std::vector<uint32_t> out_shape = {32, 128};

// 创建输入 Tensor（例如来自模型全局输入或前一层输出）
auto in_tensor = std::make_unique<Tensor>(
  _id, name_gen(op_name, "in"), in_shape, _config.precision, true);
uint32_t in_id = in_tensor->get_id();
_tensor_table[in_id] = std::move(in_tensor);  // 等价于 Model::add_tensor

// 创建输出 Tensor
auto out_tensor = std::make_unique<Tensor>(
  _id, name_gen(op_name, "out"), out_shape, _config.precision, false);
uint32_t out_id = out_tensor->get_id();
_tensor_table[out_id] = std::move(out_tensor);

// 2. 创建 Operator，并将 Tensor ID 绑定到输入/输出槽
auto op = std::make_unique<MyNewOperator>(_config, this, op_name, _attributes, _target_core);
op->add_input(in_id);
op->add_output(out_id);

// 3. 绑定完 Tensor 之后，才能安全初始化 Tile
op->initialize_tiles(_mapping_table);

uint32_t op_id = op->get_id();
_operation_map[op_id] = std::move(op);

_executable_layer.clear();
for (auto& [key, val] : _operation_map) {
  if (val->check_executable()) {
    _executable_layer.push_back(val.get());
  }
}
```

这样在 `--mode ls_test` 或其他自定义模式下，即可完全绕过 ONNX，直接在 C++ 里拼图。

---

## 6. 在 `models_list.json` 中配置调用

以默认模式为例（`main.cc` 中的 default 模式）：

文件：`example/models_list.json`（或你自己的 models_list）

```json
{
  "models": [
    {
      "name": "my_new_model",
      "request_time": 0
    }
  ]
}
```

- `name`：对应 `models/` 目录下的子目录名，例如：
  - ONNX 路径：`models/my_new_model/my_new_model.onnx`
  - 映射文件：`models/my_new_model/my_new_model.mapping`
- `request_time`：请求到达时间（单位 us），调度器会根据这个时间把模型加入请求队列。

如果是 `ls_test` 这类 C++ 构图模式，只要在 `models_list.json` 改名，`main.cc` 里对应模式会使用该名字实例化你的自定义 `Model`（例如 `ChannelModel` + `MyNewOperator`）。

---

## 7. 推荐的开发步骤小结

1. **复制模板**：
   - 从 `MyNewOperator.h/.cc` 复制一份，重命名为你的算子（例如 `ZFEqualizerOp`）。
2. **填充形状和属性解析 + Tensor 绑定**：
  - 在构造函数中解析 ONNX 节点或 attribute 中的形状、超参数。
  - 在纯 C++ 构图模式下，**务必**在 `Model` 中创建输入/输出 `Tensor`，并通过 `add_input(id) / add_output(id)` 把它们挂到算子上，然后再调用 `initialize_tiles`。
3. **实现 `plan_tiling`**：
   - 根据算子维度构造 `Mapping::LoopCounts`，从 `MappingTable` 中查表。
4. **实现 `emit_load_instructions`**：
   - 用 `get_operand_addr` + `make_address` + `MOVIN` 指令把输入/权重搬到 SPAD。
5. **实现 `emit_compute_instructions`**：
  - 用 `GEMM` / `GEMM_PRELOAD` / 向量指令描述计算；确保 `tile_m/k/n`、`compute_size` 合理。
  - ONNXim 当前采用“按 Tile 指令顺序执行”的同步模型：同一 Tile 内，`MOVIN` → 计算 → `MOVOUT` 的顺序本身就是 barrier，不需要额外自定义 `PIPE_BARRIER` 指令；可以参考 `GemmWS::initialize_instructions` 的写法。
6. **实现 `emit_store_instructions`**：
  - 用 `MOVOUT` 从 ACCUM_SPAD 写回 DRAM（`src_from_accum = true`），并在最后一个指令上正确设置 `last_inst` 标志，以便 Core 知道 Tile 已完成。
7. **在 `OperationFactory` 注册**：
   - 添加 `op_type` -> C++ 类的映射；可选支持 `copy_operation`。
8. **准备模型和 mapping**：
   - 在 ONNX 中加入该算子，或者在自定义 `Model` 中手动构图。
   - 为该算子的维度准备合适的 `.mapping` 文件，供 `MappingTable` 使用。
9. **运行仿真并验证**：
   - 使用 `--log_level debug` 查看调度、Tile 下发和 DRAM 行为。
   - 关注 `MatMul active cycle`、`Memory unit idle cycle`、`Systolic Array Utilization` 等统计。

按照上述流程，你可以快速为 ONNXim 添加、迭代新的硬件算子，并稳定地获得可解释的性能统计。