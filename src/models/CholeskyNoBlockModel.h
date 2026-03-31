#pragma once

#include "../Model.h"
#include "../operations/CholeskyInvNoBlockOp.h"

class CholeskyNoBlockModel : public Model {
 public:
  CholeskyNoBlockModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
