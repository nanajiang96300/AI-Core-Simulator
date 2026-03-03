#include "MyNewOperator.h"

#include "../Model.h"
#include "../Tensor.h"

MyNewOperator::MyNewOperator(SimulationConfig config,
                             Model* model,
                             onnx::NodeProto& node_proto,
                             uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "MyNewOp";  // used by OperationFactory::copy_operation

  // Parse shapes / attributes from ONNX node or attribute map.
  // Example: _input_shape = parse_dims(get_attribute("input_shape"));
  // For ONNX-based ops, you typically read tensor shapes from the model
  // (Model / Tensor) instead of attributes.
}

MyNewOperator::MyNewOperator(SimulationConfig config,
                             Model* model,
                             const std::string& name,
                             std::map<std::string, std::string>& attributes,
                             uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "MyNewOp";

  // Example of attribute-based shape parsing.
  // _input_shape = parse_dims(get_attribute("input_shape"));
  // _weight_shape = parse_dims(get_attribute("weight_shape"));
  // _output_shape = parse_dims(get_attribute("output_shape"));
}

MyNewOperator::MyNewOperator(SimulationConfig config,
                             MappingTable& mapping_table,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& weight_shape,
                             const std::vector<uint32_t>& output_shape,
                             uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  _optype = "MyNewOp";
  _input_shape = input_shape;
  _weight_shape = weight_shape;
  _output_shape = output_shape;

  // In this path you are expected to create input/output tensors
  // manually and register them with the Model (similar to LSEstimatorOp
  // or Gemm constructors used in tests).
}

void MyNewOperator::plan_tiling(MappingTable& mapping_table, Mapping& mapping_out) {
  // Standard pattern:
  // 1) Build a Mapping::LoopCounts key from logical dimensions.
  // 2) Look up the mapping in the MappingTable.
  // 3) Handle out_of_range to give a clear error if mapping is missing.
  Mapping::LoopCounts key{
      .N = _output_shape.empty() ? 1u : _output_shape.back(),
      .C = _weight_shape.empty() ? 1u : _weight_shape.front(),
      .M = _weight_shape.size() > 1 ? _weight_shape[1] : 1u,
      .S = 1,
      .R = 1,
      .Q = 1,
      .P = 1,
      .target_core = target_core};

  try {
    mapping_out = mapping_table.at(key);
  } catch (const std::out_of_range&) {
    spdlog::error("[MyNewOp] Mapping key not found: N={} C={} M={} P={} Q={} S={} R={}",
                  key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
}

void MyNewOperator::initialize_tiles(MappingTable& mapping_table) {
  Mapping mapping;
  plan_tiling(mapping_table, mapping);

  int core_id = -1;
  for (uint32_t n = 0; n < mapping.tile_out_loop.N; ++n) {
    for (uint32_t m = 0; m < mapping.tile_out_loop.M; ++m) {
      for (uint32_t c = 0; c < mapping.tile_out_loop.C; ++c) {
        if (c == 0) {
          core_id = (core_id + 1) % _config.num_cores;
        }

        auto tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = _optype,
            .layer_id = _id,
            .batch = static_cast<int>(n),
            .Q = 1,
            .P = 1,
            .M = static_cast<int>(m),
            .C = static_cast<int>(c),
            .S = 1,
            .R = 1,
            .accum = c != 0,
            .core_id = core_id});

        initialize_instructions(tile.get(), mapping);
        if (!tile->instructions.empty()) {
          _tiles.push_back(std::move(tile));
        }
      }
    }
  }
}

void MyNewOperator::emit_load_instructions(Tile* tile, const Mapping& mapping) {
  // Example skeleton for MOVIN of inputs/weights.
  // Use get_operand_addr(_INPUT_OPERAND + k) and make_address(index, dims)
  // to generate DRAM addresses, then push MOVIN instructions into
  // tile->instructions.
  (void)tile;
  (void)mapping;
}

void MyNewOperator::emit_compute_instructions(Tile* tile, const Mapping& mapping) {
  // Example skeleton for GEMM/GEMM_PRELOAD/vector compute instructions.
  // Use SystolicWS latency model via opcode = GEMM or GEMM_PRELOAD.
  (void)tile;
  (void)mapping;
}

void MyNewOperator::emit_store_instructions(Tile* tile, const Mapping& mapping) {
  // Example skeleton for MOVOUT of results from accumulator SPAD to DRAM.
  (void)tile;
  (void)mapping;
}

void MyNewOperator::initialize_instructions(Tile* tile, Mapping mapping) {
  // Standard pattern in ONNXim: instructions of a tile are executed in order
  // by the core pipeline. Keeping all MOVIN before compute, and all MOVOUT
  // after compute, effectively acts as a barrier between phases without any
  // extra PIPE_BARRIER opcode.
  emit_load_instructions(tile, mapping);    // DRAM -> SPAD/ACCUM_SPAD
  emit_compute_instructions(tile, mapping); // systolic / vector compute
  emit_store_instructions(tile, mapping);   // ACCUM_SPAD -> DRAM (mark last_inst)
}
