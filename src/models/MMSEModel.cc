#include "MMSEModel.h"

#include <chrono>

void MMSEModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
  // No external weights; all tensors are synthetic.
}

void MMSEModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  auto start = std::chrono::high_resolution_clock::now();

  uint32_t M = 32;
  uint32_t K = 32;
  if (_model_config.contains("matrix_m")) {
    M = static_cast<uint32_t>(_model_config["matrix_m"]);
  }
  if (_model_config.contains("matrix_k")) {
    K = static_cast<uint32_t>(_model_config["matrix_k"]);
  }

  uint32_t batch_size = 32;
  if (_model_config.contains("batch_size")) {
    batch_size = static_cast<uint32_t>(_model_config["batch_size"]);
  }

  // H, Y use [batch, M, K]; Gram and inverse live in [batch, K, K].
  const std::vector<uint32_t> shape_H{batch_size, M, K};
  const std::vector<uint32_t> shape_KK{batch_size, K, K};
  const std::vector<uint32_t> matrix_shape{M, K};

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
  if (_model_config.contains("attributes") &&
      _model_config["attributes"].contains("iterations")) {
    attrs["iterations"] =
        _model_config["attributes"]["iterations"].get<std::string>();
  } else {
    attrs["iterations"] = "10";
  }
  attrs["batch_size"] = std::to_string(batch_size);

  // Inputs: H (batch,M,K), X_init_32 (batch,K,K), C_32 (batch,K,K),
  //         Y (batch,M,K); Output X_hat (batch,K,K).
  uint32_t h_id = make_tensor("H", shape_H, true);
  uint32_t x_init_id = make_tensor("X_init", shape_KK, true);
  uint32_t c_id = make_tensor("C", shape_KK, true);
  uint32_t y_id = make_tensor("Y", shape_H, true);
  uint32_t out_id = make_tensor("X_hat", shape_KK, false);

  std::string op_name = name_gen(get_name(), "MMSEOp");
  auto op = std::make_unique<MMSEOp>(_config, this, op_name, attrs,
                                     /*target_core=*/0);
  op->set_matrix_shape(matrix_shape);

  op->add_input(h_id);
  op->add_input(x_init_id);
  op->add_input(c_id);
  op->add_input(y_id);
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
  spdlog::info("MMSEModel initialization time: {:2f} seconds", duration.count());
}
