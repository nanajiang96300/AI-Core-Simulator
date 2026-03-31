#include "DeepUnfoldModel.h"

#include <chrono>

void DeepUnfoldModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
  // No external weights for synthetic DeepUnfold test model.
}

void DeepUnfoldModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  auto start = std::chrono::high_resolution_clock::now();

  uint32_t M = 256;
  uint32_t U = 32;
  uint32_t batch_size = 96;

  if (_model_config.contains("matrix_m")) {
    M = static_cast<uint32_t>(_model_config["matrix_m"]);
  }
  if (_model_config.contains("matrix_k")) {
    U = static_cast<uint32_t>(_model_config["matrix_k"]);
  }
  if (_model_config.contains("batch_size")) {
    batch_size = static_cast<uint32_t>(_model_config["batch_size"]);
  }

  const std::vector<uint32_t> shape_mu{batch_size, M, U};
  const std::vector<uint32_t> shape_uu_batch{batch_size, U, U};
  const std::vector<uint32_t> shape_uu{U, U};
  const std::vector<uint32_t> matrix_shape{M, U};

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

  std::map<std::string, std::string> attrs;
  attrs["batch_size"] = std::to_string(batch_size);

  if (_model_config.contains("attributes")) {
    if (_model_config["attributes"].contains("layers")) {
      attrs["layers"] = _model_config["attributes"]["layers"].get<std::string>();
    }
    if (_model_config["attributes"].contains("vector_repeats")) {
      attrs["vector_repeats"] =
          _model_config["attributes"]["vector_repeats"].get<std::string>();
    }
  }

  uint32_t h_id = make_tensor("H", shape_mu, true);
  uint32_t reg_id = make_tensor("RegI", shape_uu, true);
  uint32_t y_id = make_tensor("Y", shape_mu, true);
  uint32_t x0_id = make_tensor("X0", shape_uu_batch, true);
  uint32_t out_id = make_tensor("X_hat", shape_uu_batch, false);

  std::string op_name = name_gen(get_name(), "DeepUnfoldNPUOp");
  auto op = std::make_unique<DeepUnfoldNPUOp>(_config, this, op_name, attrs,
                                              /*target_core=*/0);
  op->set_matrix_shape(matrix_shape);

  op->add_input(h_id);
  op->add_input(reg_id);
  op->add_input(y_id);
  op->add_input(x0_id);
  op->add_output(out_id);

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
  spdlog::info("DeepUnfoldModel initialization time: {:2f} seconds", duration.count());
}
