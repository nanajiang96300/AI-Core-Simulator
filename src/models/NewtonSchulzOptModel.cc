#include "NewtonSchulzOptModel.h"

#include <chrono>

NewtonSchulzOptModel::~NewtonSchulzOptModel() = default;

void NewtonSchulzOptModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {}

void NewtonSchulzOptModel::initialize_model(
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

  const std::vector<uint32_t> tensor_shape{batch_size, M, K};
  const std::vector<uint32_t> matrix_shape{M, K};

  uint32_t root_id = get_root_node_id();

  auto make_tensor = [&](const std::string& base,
                         bool produced) -> uint32_t {
    auto t = std::make_unique<Tensor>(
        root_id,
        name_gen(get_name(), base),
        const_cast<std::vector<uint32_t>&>(tensor_shape),
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
    attrs["iterations"] = _model_config["attributes"]["iterations"].get<std::string>();
  } else {
    attrs["iterations"] = "10";
  }
  attrs["batch_size"] = std::to_string(batch_size);

  uint32_t a_id = make_tensor("A", true);
  uint32_t x_id = make_tensor("X_init", true);
  uint32_t c_id = make_tensor("C", true);
  uint32_t out_id = make_tensor("X_out", false);

  std::string op_name = name_gen(get_name(), "NewtonSchulzOpt");
  auto op = std::make_unique<NewtonSchulzOptOp>(_config, this, op_name, attrs,
                                                /*target_core=*/0);

  op->set_matrix_shape(matrix_shape);

  op->add_input(a_id);
  op->add_input(x_id);
  op->add_input(c_id);
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
  spdlog::info("NewtonSchulzOptModel initialization time: {:2f} seconds", duration.count());
}
