#pragma once

#include "../Model.h"
#include "../operations/LDLDecompOp.h"

// C++-only LDL decomposition test model.
//
// Builds a single batched LDLDecompOp for communication-oriented
// matrix preprocessing experiments.
class LDLModel : public Model {
 public:
  LDLModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
