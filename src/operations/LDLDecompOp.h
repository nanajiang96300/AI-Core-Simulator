#pragma once

#include "Operation.h"

// LDLDecompOp (Block-LDL inspired, simulation-oriented)
//
// This operator models the execution pattern of:
//  1) Gram + regularization: A = H^H H + lambda I
//  2) Block-LDL decomposition (2x2 blocks)
//  3) Backward-substitution-based inverse assembly
//
// The simulator is cycle / instruction driven, so this implementation focuses
// on realistic memory/compute phases and dependency barriers rather than exact
// complex-valued numerical fidelity.
class LDLDecompOp : public Operation {
 public:
  LDLDecompOp(SimulationConfig config,
              Model* model,
              const std::string& name,
              std::map<std::string, std::string>& attributes,
              uint32_t target_core = 0);

  LDLDecompOp(SimulationConfig config,
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

  // [M, U] for H, where U is user dimension and A is [U, U].
  std::vector<uint32_t> _matrix_shape;
  uint32_t _batch_size{96};
  uint32_t _block_size{2};
  uint32_t _bwd_steps{1};
  uint32_t _pack_blocks{0};
};
