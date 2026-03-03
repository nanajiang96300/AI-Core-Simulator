#pragma once

#include "Operation.h"

// MMSE estimator operator (synthetic, C++-only).
//
// For each batch, this op models a simplified MMSE pipeline:
//  1) Newton–Schulz-style matrix inverse on a [M, K] tile (reusing
//     the same instruction structure as NewtonSchulzOp).
//  2) A follow-up GEMM + ADD stage to represent the MMSE filter
//     application (e.g., W * Y + bias).
//
// Numerically the simulator does not track values; the goal is to
// approximate the compute / memory pattern and pipeline shape.
class MMSEOp : public Operation {
 public:
  // Attribute-based constructor for custom C++ models.
  MMSEOp(SimulationConfig config,
         Model* model,
         const std::string& name,
         std::map<std::string, std::string>& attributes,
         uint32_t target_core = 0);

  // Mapping-based constructor (not used for now, but kept for symmetry).
  MMSEOp(SimulationConfig config,
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

  std::vector<uint32_t> _matrix_shape;  // [M, K]
  uint32_t _iterations{10};
  uint32_t _batch_size{96};
};
