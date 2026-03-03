#pragma once

#include "../Model.h"
#include "../operations/SeriesInverseOp.h"

// SeriesInverseModel: C++-only model wrapping a single SeriesInverseOp
// for micro-architectural studies and direct comparison with NewtonSchulzModel.
class SeriesInverseModel : public Model {
 public:
  SeriesInverseModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
