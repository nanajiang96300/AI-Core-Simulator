#pragma once

#include "../Model.h"
#include "../operations/NewtonSchulzOptOp.h"

// Optimized variant of the Newton-Schulz C++ model.
// This is wired identically to `NewtonSchulzModel` but instantiates
// `NewtonSchulzOptOp` instead, so we can compare timeline/cycles
// between baseline and optimized operators.
class NewtonSchulzOptModel : public Model {
 public:
  NewtonSchulzOptModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

    virtual ~NewtonSchulzOptModel();

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
