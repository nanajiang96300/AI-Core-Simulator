#pragma once

#include "Operation.h"

// DeepUnfoldNPUOptOp
//
// Optimized variant of DeepUnfoldNPUOp for NPU-oriented scheduling:
//  1) Keep baseline load/gram/regularization stages.
//  2) Use grouped layer updates (multi-layer fusion) to reduce barrier and
//     vector-stage frequency.
//  3) Optionally perform sparse vector corrections at configured intervals.
class DeepUnfoldNPUOptOp : public Operation {
 public:
  DeepUnfoldNPUOptOp(SimulationConfig config,
                     Model* model,
                     const std::string& name,
                     std::map<std::string, std::string>& attributes,
                     uint32_t target_core = 0);

  DeepUnfoldNPUOptOp(SimulationConfig config,
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

  std::vector<uint32_t> _matrix_shape;  // [M, U]
  uint32_t _batch_size{96};
  uint32_t _layers{12};
  uint32_t _layer_group{2};
  uint32_t _vector_interval{2};
};
