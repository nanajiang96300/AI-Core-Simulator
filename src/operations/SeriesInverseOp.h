#pragma once

#include "Operation.h"

// SeriesInverseOp: matrix inverse via fixed-length Neumann series.
//
// Iteration form (per batch):
//   X_{k+1} = X_k + (C - A * X_k)
// where C is typically the identity (or 2I after normalization).
//
// Numerically我们并不追踪真正的数值精度，主要目标是：
// - 复用 GEMM_PRELOAD + ADD 指令，形成与 NewtonSchulzOp 不同的
//   负载/流水形态；
// - 保持与 NewtonSchulzOp 相同的接口（A, X_init, C → X_out），便于对比。
class SeriesInverseOp : public Operation {
 public:
  // ONNX-based constructor.
  SeriesInverseOp(SimulationConfig config,
                  Model* model,
                  onnx::NodeProto& node_proto,
                  uint32_t target_core = 0);

  // Attribute-based constructor (C++ models / tests).
  SeriesInverseOp(SimulationConfig config,
                  Model* model,
                  const std::string& name,
                  std::map<std::string, std::string>& attributes,
                  uint32_t target_core = 0);

  // Mapping-based constructor (synthetic runs with explicit shapes).
  SeriesInverseOp(SimulationConfig config,
                  MappingTable& mapping_table,
                  const std::vector<uint32_t>& matrix_shape,
                  uint32_t target_core = 0);

  // Batched tile construction (one tile per batch, RR across cores).
  void initialize_tiles(MappingTable& mapping_table) override;

  // Helpers for C++ models.
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
