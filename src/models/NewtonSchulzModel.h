#pragma once

#include "../Model.h"
#include "../operations/NewtonSchulzOp.h"

// A simple model that contains a single Newton-Schulz operator
// built directly in C++ (no ONNX graph). This is intended for
// micro-architectural pipeline studies and timeline visualization.
class NewtonSchulzModel : public Model {
 public:
  NewtonSchulzModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
