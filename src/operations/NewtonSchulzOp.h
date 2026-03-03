#pragma once

#include "Operation.h"

// Newton-Schulz matrix inverse operator.
//
// Implements a fixed-length iteration of the form:
//   X_{k+1} = X_k * (2I - A * X_k)
// with three inputs:
//   Input 0: A         (matrix to invert)
//   Input 1: X_init    (initial guess X0)
//   Input 2: C = 2I    (constant matrix)
// and one output:
//   Output 0: X_out    (final estimate after N iterations).
//
// The actual numerical correctness is not the focus here; the goal is to
// generate a deterministic instruction stream with clear load/compute/store
// phases and explicit dependency-based "barriers" between them.
class NewtonSchulzOp : public Operation {
 public:
  // ONNX-based constructor (normal path when loading from an ONNX graph).
  NewtonSchulzOp(SimulationConfig config,
                 Model* model,
                 onnx::NodeProto& node_proto,
                 uint32_t target_core = 0);

  // Attribute-based constructor (for custom C++ models or tests).
  NewtonSchulzOp(SimulationConfig config,
                 Model* model,
                 const std::string& name,
                 std::map<std::string, std::string>& attributes,
                 uint32_t target_core = 0);

  // Mapping-based constructor (synthetic runs with explicit shapes).
  NewtonSchulzOp(SimulationConfig config,
                 MappingTable& mapping_table,
                 const std::vector<uint32_t>& matrix_shape,
                 uint32_t target_core = 0);

  // Tile construction entry point.
  // Now supports generating multiple tiles (batches) distributed across all available cores.
  void initialize_tiles(MappingTable& mapping_table) override;

  // Explicitly set matrix shape for synthetic/C++ models (e.g., NewtonSchulzModel).
  // When this is called before initialize_tiles(), shape inference from the
  // Model is skipped and this shape is used directly.
  void set_matrix_shape(const std::vector<uint32_t>& shape) { _matrix_shape = shape; }

  // [新增] 允许外部显式设置 Batch Size (如果不用 ONNX 属性)
  void set_batch_size(uint32_t batch) { _batch_size = batch; }

 protected:
  // Per-tile instruction generation.
  void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  void parse_attributes();
  void infer_shapes_from_model();

  // Matrix shape (assumed square: [N, N]).
  std::vector<uint32_t> _matrix_shape;

  // Number of Newton-Schulz iterations.
  uint32_t _iterations{10};

  // [新增] Batch Size，默认为 96 (匹配你的 910B 场景)
  uint32_t _batch_size{96};
};