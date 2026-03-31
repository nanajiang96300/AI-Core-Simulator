#include "CholeskyNoBlockModel.h"

#include <chrono>

void CholeskyNoBlockModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
}

void CholeskyNoBlockModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;
  auto start = std::chrono::high_resolution_clock::now();

  uint32_t M = 64;
  uint32_t U = 16;
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

  const std::vector<uint32_t> shape_h{batch_size, M, U};
  const std::vector<uint32_t> shape_uu{batch_size, U, U};
  const std::vector<uint32_t> shape_reg{U, U};
  const std::vector<uint32_t> matrix_shape{M, U};

  uint32_t root_id = get_root_node_id();

  auto make_tensor = [&](const std::string& base,
                         const std::vector<uint32_t>& shape,
                         bool produced) -> uint32_t {
    auto tensor = std::make_unique<Tensor>(
        root_id,
        name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(shape),
        _config.precision,
        produced);
    if (produced) {
      tensor->set_produced();
    }
    uint32_t id = tensor->get_id();
    add_tensor(std::move(tensor));
    return id;
  };

  std::map<std::string, std::string> attrs;
  attrs["batch_size"] = std::to_string(batch_size);

  if (_model_config.contains("attributes")) {
    if (_model_config["attributes"].contains("strict_iso_lowering")) {
      attrs["strict_iso_lowering"] =
          _model_config["attributes"]["strict_iso_lowering"].get<std::string>();
    }
  }

  uint32_t h_id = make_tensor("H", shape_h, true);
  uint32_t reg_id = make_tensor("RegI", shape_reg, true);
  uint32_t out_id = make_tensor("Ainv", shape_uu, false);

  std::string op_name = name_gen(get_name(), "CholeskyInvNoBlockOp");
  auto op = std::make_unique<CholeskyInvNoBlockOp>(_config, this, op_name, attrs,
                                                   /*target_core=*/0);
  op->set_matrix_shape(matrix_shape);

  op->add_input(h_id);
  op->add_input(reg_id);
  op->add_output(out_id);

  op->initialize_tiles(_mapping_table);

  uint32_t op_id = op->get_id();
  _operation_map[op_id] = std::move(op);

  _executable_layer.clear();
  for (auto& [key, value] : _operation_map) {
    if (value->check_executable()) {
      _executable_layer.push_back(value.get());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("CholeskyNoBlockModel initialization time: {:2f} seconds", duration.count());
}
