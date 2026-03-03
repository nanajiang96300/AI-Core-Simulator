#pragma once

#include "Operation.h"

// Template example for creating a new operator.
// Replace "MyNewOperator" and fields with your own op.
class MyNewOperator : public Operation {
 public:
  // ONNX-based constructor (normal path when loading from an ONNX graph)
  MyNewOperator(SimulationConfig config,
                Model* model,
                onnx::NodeProto& node_proto,
                uint32_t target_core = 0);

  // Attribute-based constructor (used by custom models like ChannelModel)
  MyNewOperator(SimulationConfig config,
                Model* model,
                const std::string& name,
                std::map<std::string, std::string>& attributes,
                uint32_t target_core = 0);

  // Mapping-based constructor (used for unit tests or synthetic runs)
  MyNewOperator(SimulationConfig config,
                MappingTable& mapping_table,
                const std::vector<uint32_t>& input_shape,
                const std::vector<uint32_t>& weight_shape,
                const std::vector<uint32_t>& output_shape,
                uint32_t target_core = 0);

  // Every concrete operator must implement tile initialization.
  void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  // Optional: override instruction initialization for each tile.
  void initialize_instructions(Tile* tile, Mapping mapping) override;

  // Convenience helpers to structure your implementation.
  // These are not part of the base Operation interface, but provide
  // a standard pattern for new ops.
  void plan_tiling(MappingTable& mapping_table, Mapping& mapping_out);
  void emit_load_instructions(Tile* tile, const Mapping& mapping);
  void emit_compute_instructions(Tile* tile, const Mapping& mapping);
  void emit_store_instructions(Tile* tile, const Mapping& mapping);

 private:
  // Cached shapes / parameters parsed from attributes or ONNX node.
  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _weight_shape;
  std::vector<uint32_t> _output_shape;

  // Example attribute flags.
  bool _use_bias{false};
};
