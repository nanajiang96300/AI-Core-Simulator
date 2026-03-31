#pragma once

#include "Operation.h"

class CholeskyInvNoBlockOp : public Operation {
 public:
  CholeskyInvNoBlockOp(SimulationConfig config,
                       Model* model,
                       const std::string& name,
                       std::map<std::string, std::string>& attributes,
                       uint32_t target_core = 0);

  CholeskyInvNoBlockOp(SimulationConfig config,
                       MappingTable& mapping_table,
                       const std::vector<uint32_t>& matrix_shape,
                       uint32_t target_core = 0);

  void initialize_tiles(MappingTable& mapping_table) override;

  void set_matrix_shape(const std::vector<uint32_t>& shape) { _matrix_shape = shape; }
  void set_batch_size(uint32_t batch) { _batch_size = batch; }

 protected:
  void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  void parse_attributes();
  void infer_shapes_from_model();

  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  bool _strict_iso_lowering{false};
};
