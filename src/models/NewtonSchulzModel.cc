#include "NewtonSchulzModel.h"

#include <chrono>

void NewtonSchulzModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
  // No external weights for this synthetic Newton-Schulz test.
}

void NewtonSchulzModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;  // unused
  auto start = std::chrono::high_resolution_clock::now();

  // For this experiment, build a single batched Newton-Schulz inverse
  // operator, and let the operator itself manage batch tiling and
  // multi-core dispatch. This avoids the previous "double batching"
  // (model-level batch loop × op-level _batch_size) and makes the
  // behavior closer to LSEstimator's LS-style micro-benchmark.

  // Matrix dimension for this C++ test.
  // Default is 32x32, but can be overridden via model_config, e.g.:
  // {
  //   "batch_size": 96,
  //   "matrix_m": 256,
  //   "matrix_k": 32,
  //   "attributes": { "iterations": "10" }
  // }
  uint32_t M = 32;
  uint32_t K = 32;
  if (_model_config.contains("matrix_m")) {
    M = static_cast<uint32_t>(_model_config["matrix_m"]);
  }
  if (_model_config.contains("matrix_k")) {
    K = static_cast<uint32_t>(_model_config["matrix_k"]);
  }

  // Batch size comes from model_config if provided; otherwise default to 32.
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

  // Attributes for NewtonSchulzOp: iterations from model_config if present.
  std::map<std::string, std::string> attrs;
  if (_model_config.contains("attributes") &&
      _model_config["attributes"].contains("iterations")) {
    attrs["iterations"] = _model_config["attributes"]["iterations"].get<std::string>();
  } else {
    attrs["iterations"] = "10";
  }

  // Optionally expose batch_size to the op; when inputs are 3-D
  // [B, N, N], NewtonSchulzOp will also infer _batch_size = B from
  // the tensor shape, so this is mostly documentation / override.
  attrs["batch_size"] = std::to_string(batch_size);

  // Create a single batched operator.
  uint32_t a_id = make_tensor("A", true);
  uint32_t x_id = make_tensor("X_init", true);
  uint32_t c_id = make_tensor("C", true);
  uint32_t out_id = make_tensor("X_out", false);

  std::string op_name = name_gen(get_name(), "NewtonSchulz");
  auto op = std::make_unique<NewtonSchulzOp>(_config, this, op_name, attrs,
                                             /*target_core=*/0);

  // Because the C++ model wires inputs after constructing the op,
  // shape inference from Model::get_tensor() would not see any inputs
  // yet. Setting the matrix shape here ensures
  // initialize_instructions() has a valid [N, N] matrix while the
  // batch dimension B is handled implicitly.
  op->set_matrix_shape(matrix_shape);

  // Wire tensors into the op.
  op->add_input(a_id);
  op->add_input(x_id);
  op->add_input(c_id);
  op->add_output(out_id);

  // Initialize tiles/instructions for this custom op.
  op->initialize_tiles(_mapping_table);

  uint32_t op_id = op->get_id();
  _operation_map[op_id] = std::move(op);

  // Build executable layer: this model has a single op; if its
  // inputs are ready we can schedule it immediately.
  _executable_layer.clear();
  for (auto& [key, val] : _operation_map) {
    if (val->check_executable()) {
      _executable_layer.push_back(val.get());
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("NewtonSchulzModel initialization time: {:2f} seconds", duration.count());
}
