#pragma once

#include "../Model.h"
#include "../operations/DeepUnfoldNPUOptOp.h"

// C++-only optimized DeepUnfold test model.
//
// Mirrors baseline DeepUnfoldModel, but binds to DeepUnfoldNPUOptOp and
// accepts optimization attributes in model config.
class DeepUnfoldNPUOptModel : public Model {
 public:
  DeepUnfoldNPUOptModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
