#pragma once

#include "Operation.h"

// DeepUnfoldNPUOp (simulation-oriented, NPU pipeline style)
//
// This operator models a deep-unfolding style MMSE-like detector with an
// LDL-inspired instruction flow:
//  1) Load H / Y / regularizer / initial inverse guess
//  2) Build Gram + regularized matrix
//  3) Iterate unfolding layers using alternating CUBE/VECTOR phases
//  4) Build W and estimate X_hat
//
// The simulator tracks cycle/instruction behavior, so the implementation
// focuses on realistic dependency and memory/computation structure.
class DeepUnfoldNPUOp : public Operation {
 public:
  DeepUnfoldNPUOp(SimulationConfig config,
                  Model* model,
                  const std::string& name,
                  std::map<std::string, std::string>& attributes,
                  uint32_t target_core = 0);

  DeepUnfoldNPUOp(SimulationConfig config,
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
  uint32_t _vector_repeats{1};
};
