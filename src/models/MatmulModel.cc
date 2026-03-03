#include "MatmulModel.h"

#include <chrono>

void MatmulModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
}

void MatmulModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  auto start = std::chrono::high_resolution_clock::now();

  uint32_t batch_size = 1;
  uint32_t matrix_m = 256;  // lhs rows
  uint32_t matrix_k = 32;   // reduction dim
  uint32_t matrix_n = 256;  // rhs cols

  if (_model_config.contains("batch_size")) {
    batch_size = static_cast<uint32_t>(_model_config["batch_size"]);
  }
  if (_model_config.contains("matrix_m")) {
    matrix_m = static_cast<uint32_t>(_model_config["matrix_m"]);
  }
  if (_model_config.contains("matrix_k")) {
    matrix_k = static_cast<uint32_t>(_model_config["matrix_k"]);
  }
  if (_model_config.contains("matrix_n")) {
    matrix_n = static_cast<uint32_t>(_model_config["matrix_n"]);
  }

  const std::vector<uint32_t> input_shape{batch_size, matrix_m, matrix_k};
  const std::vector<uint32_t> weight_shape{matrix_k, matrix_n};
  const std::vector<uint32_t> output_shape{batch_size, matrix_m, matrix_n};

  uint32_t root_id = get_root_node_id();

  auto make_tensor = [&](const std::string& base,
                         const std::vector<uint32_t>& shape,
                         bool produced) -> uint32_t {
    auto t = std::make_unique<Tensor>(
        root_id,
        name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(shape),
        _config.precision,
        produced);
    if (produced) {
      t->set_produced();
    }
    uint32_t id = t->get_id();
    add_tensor(std::move(t));
    return id;
  };

  uint32_t lhs_id = make_tensor("A", input_shape, true);
  uint32_t rhs_id = make_tensor("B", weight_shape, true);

  std::map<std::string, std::string> attrs;
  attrs["has_bias"] = "0";
  attrs["input_shape"] = dims_to_string(input_shape);
  attrs["weight_shape"] = dims_to_string(weight_shape);
  attrs["output_shape"] = dims_to_string(output_shape);

  std::string op_name = name_gen(get_name(), "Matmul");
  auto op = std::make_unique<GemmWS>(_config, this, op_name, attrs,
                                     /*target_core=*/0);

  op->add_input(lhs_id);
  op->add_input(rhs_id);
  op->initialize_tiles(_mapping_table);

  uint32_t op_id = op->get_id();
  _operation_map[op_id] = std::move(op);

  _executable_layer.clear();
  for (auto& [key, val] : _operation_map) {
    if (val->check_executable()) {
      _executable_layer.push_back(val.get());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("MatmulModel initialization time: {:2f} seconds", duration.count());
}
