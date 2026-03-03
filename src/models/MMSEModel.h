#pragma once

#include "../Model.h"
#include "../operations/MMSEOp.h"

// Simple C++-only MMSE test model.
//
// Builds a single batched MMSEOp with input tensors shaped
// [batch_size, M, K]. Intended for micro-architectural studies.
class MMSEModel : public Model {
 public:
  MMSEModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
