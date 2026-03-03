#pragma once

#include "../Model.h"
#include "../operations/LSEstimatorOp.h"

// A simple model that contains a single LS estimator operator
// built directly in C++ (no ONNX graph).
class ChannelModel : public Model {
 public:
  ChannelModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
