#include "ChannelModel.h"

#include <chrono>

void ChannelModel::initialize_weight(
    std::vector<std::unique_ptr<Tensor>>& /*weight_table*/) {
  // No external weights for this simple LS test model.
}

void ChannelModel::initialize_model(
    std::vector<std::unique_ptr<Tensor>>& weight_table) {
  (void)weight_table;  // unused
  auto start = std::chrono::high_resolution_clock::now();

  // Build a single LS estimator operator; inputs/outputs are created inside
  // LSEstimatorOp and marked as produced (for inputs).
  std::string op_name = name_gen(get_name(), "LSEstimator");
  auto op = std::make_unique<LSEstimatorOp>(_config, this, op_name, _target_core);

  // Explicitly initialize tiles for this custom op (no ONNX mapping entry).
  // This ensures tiles and instructions are generated before simulation.
  op->initialize_tiles(_mapping_table);

  uint32_t op_id = op->get_id();
  _operation_map[op_id] = std::move(op);

  // Build executable layer: only this op, which is ready because its inputs
  // are marked produced in the constructor.
  _executable_layer.clear();
  for (auto& [key, val] : _operation_map) {
    if (val->check_executable()) {
      _executable_layer.push_back(val.get());
    }
  }

  // Debug: check LS estimator input tensor produced flags and executable layer size
  if (!_operation_map.empty()) {
    auto &ls_op = _operation_map.begin()->second;
    if (ls_op) {
      auto a_tensor = get_tensor(ls_op->get_input(0)->get_id());
      auto b_tensor = get_tensor(ls_op->get_input(1)->get_id());
      spdlog::info("[LS Debug] LS Op Inputs: A produced = {}, B produced = {}",
                   a_tensor ? a_tensor->get_produced() : -1,
                   b_tensor ? b_tensor->get_produced() : -1);
    }
  }
  spdlog::info("[LS Debug] Executable layer size: {}", _executable_layer.size());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("ChannelModel initialization time: {:2f} seconds", duration.count());
}
