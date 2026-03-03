#pragma once

#include "Operation.h"

class LSEstimatorOp : public Operation {
 public:
  // Constructor for LS estimator operator built from C++ (no ONNX node)
  LSEstimatorOp(SimulationConfig config, Model* model,
                const std::string& name, uint32_t target_core = 0);

  // Hard-coded tiling: create 8 tiles (M=32, tile_m=4) and assign to cores 0..7
  virtual void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  // Generate MOVIN / GEMM_PRELOAD / MOVOUT instructions for a single tile
  virtual void initialize_instructions(Tile* tile, Mapping mapping) override;

 private:
  std::vector<uint32_t> _a_shape;  // [32, 32]
  std::vector<uint32_t> _b_shape;  // [32, 512]
  std::vector<uint32_t> _c_shape;  // [32, 512]
};
