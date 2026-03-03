#pragma once

#include "../Model.h"
#include "../operations/GemmWS.h"

class MatmulModel : public Model {
 public:
  MatmulModel(json model_config, SimulationConfig config, std::string name)
      : Model(model_config, config, name) {}

  virtual ~MatmulModel() = default;

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) override;
};
