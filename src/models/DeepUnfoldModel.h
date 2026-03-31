#pragma once

#include "../Model.h"
#include "../operations/DeepUnfoldNPUOp.h"

// C++-only DeepUnfold test model.
//
// Builds one batched DeepUnfoldNPUOp and exposes tuning attributes through
// JSON (layers, vector_repeats).
class DeepUnfoldModel : public Model {
 public:
  DeepUnfoldModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
