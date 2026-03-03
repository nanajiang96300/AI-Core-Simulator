#include "SeriesInverseOp.h"
#include "../Model.h"
#include <cstdlib>

#ifndef DEFAULT_BATCH_SIZE
#define DEFAULT_BATCH_SIZE 96
#endif

SeriesInverseOp::SeriesInverseOp(SimulationConfig config,
                                 Model* model,
                                 onnx::NodeProto& node_proto,
                                 uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "SeriesInverse";
  parse_attributes();
  infer_shapes_from_model();
}

SeriesInverseOp::SeriesInverseOp(SimulationConfig config,
                                 Model* model,
                                 const std::string& name,
                                 std::map<std::string, std::string>& attributes,
                                 uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "SeriesInverse";
  parse_attributes();
  infer_shapes_from_model();
}

SeriesInverseOp::SeriesInverseOp(SimulationConfig config,
                                 MappingTable& mapping_table,
                                 const std::vector<uint32_t>& matrix_shape,
                                 uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "SeriesInverse";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void SeriesInverseOp::parse_attributes() {
  auto it_iter = _attributes.find("iterations");
  if (it_iter != _attributes.end()) {
    try {
      _iterations = static_cast<uint32_t>(std::stoul(it_iter->second));
    } catch (...) {
      _iterations = 10;
    }
  } else {
    _iterations = 10;
  }

  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try {
      _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second));
    } catch (...) {
      _batch_size = DEFAULT_BATCH_SIZE;
    }
  } else {
    _batch_size = DEFAULT_BATCH_SIZE;
  }
}

void SeriesInverseOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;

  if (_inputs.size() >= 1) {
    Tensor* a_tensor = _model->get_tensor(_inputs[0]);
    if (a_tensor) {
      std::vector<uint32_t> dims = a_tensor->get_dims();
      if (dims.size() == 3) {
        _batch_size = dims[0];
        _matrix_shape = {dims[1], dims[2]};
      } else {
        _matrix_shape = dims;
      }
    }
  }
}

void SeriesInverseOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("SeriesInverseOp: Invalid core count 0!");
    return;
  }

  std::vector<int> core_load(_config.num_cores, 0);

  for (uint32_t b = 0; b < _batch_size; ++b) {
    uint32_t assigned_core = b % _config.num_cores;

    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _optype,
        .layer_id = _id,
        .fused_op_id = 0,
        .batch = b,
        .Q = 1,
        .P = 1,
        .M = 0,
        .C = 0,
        .S = 1,
        .R = 1,
        .stat = {},
        .instructions = {},
        .accum = false,
        .skip = false,
        .spad_id = 0,
        .accum_spad_id = 0,
        .core_id = static_cast<int>(assigned_core),
        .inst_finished = false});

    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) {
      _tiles.push_back(std::move(tile));
      core_load[assigned_core]++;
    }
  }

  spdlog::info("SeriesInverseOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  spdlog::info(
      "  > Load Distribution (First 4 cores): Core0: {}, Core1: {}, Core2: {}, Core3: {} ...",
      core_load[0], core_load[1], core_load[2], core_load[3]);
}

void SeriesInverseOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("SeriesInverseOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t N = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t K = _matrix_shape[_matrix_shape.size() - 1];

  addr_type matrix_size_bytes = static_cast<addr_type>(N) * K * _config.precision;
  addr_type batch_offset = static_cast<addr_type>(tile->batch) * matrix_size_bytes;

  addr_type a_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset;
  addr_type x_base = get_operand_addr(_INPUT_OPERAND + 1) + batch_offset;
  addr_type c_base = get_operand_addr(_INPUT_OPERAND + 2);  // C 作为广播常量
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset;

  addr_type addr_A = SPAD_BASE;
  addr_type addr_X = addr_A + matrix_size_bytes;
  addr_type addr_C = addr_X + matrix_size_bytes;
  addr_type addr_R = addr_C + matrix_size_bytes;
  addr_type addr_T = ACCUM_SPAD_BASE;  // 用于存放 A*X_k

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  auto emit_movin_full = [&](addr_type dram_base, addr_type spad_dest,
                             uint32_t operand_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < N; ++r) {
      for (uint32_t c = 0; c < K; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, K - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, _matrix_shape);
        addrs.insert(dram_base + off);
      }
    }
    if (!addrs.empty()) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = spad_dest,
          .size = static_cast<uint32_t>(addrs.size()),
          .src_addrs = std::vector<addr_type>(addrs.begin(), addrs.end()),
          .operand_id = operand_id,
          .base_addr = dram_base,
          .tile_m = N,
          .tile_k = K,
          .tile_n = 0,
          .my_tile = tile}));
    }
  };

  // Load phase: A, X_init, C
  emit_movin_full(a_base, addr_A, _INPUT_OPERAND + 0);
  emit_movin_full(x_base, addr_X, _INPUT_OPERAND + 1);
  emit_movin_full(c_base, addr_C, _INPUT_OPERAND + 2);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "SERIES_BARRIER_MTE2CUBE",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  // Compute phase: Neumann-style iterations
  for (uint32_t iter = 0; iter < _iterations; ++iter) {
    // 1) T = A * X_k  (X_k 当前保存在 SPAD addr_X)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "SERIES_AX",
        .dest_addr = addr_T,
        .compute_size = K,
        .src_addrs = std::vector<addr_type>{addr_A, addr_X},
        .tile_m = N,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = false,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "SERIES_BARRIER_AX2R",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 2}));

    // 2) R = C - T    (C 在 SPAD, T 在 ACCUM)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "SERIES_R",
        .dest_addr = addr_R,
        .compute_size = N * K,
        .src_addrs = std::vector<addr_type>{addr_C, addr_T},
        .tile_m = N,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = true,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "SERIES_BARRIER_R2X",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 3}));

    // 3) X_{k+1} = X_k + R   (都在 SPAD)
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "SERIES_X",
        .dest_addr = addr_X,
        .compute_size = N * K,
        .src_addrs = std::vector<addr_type>{addr_X, addr_R},
        .tile_m = N,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = false,
        .my_tile = tile}));
  }

  // Store phase: write X_T (in SPAD addr_X) back to DRAM
  std::set<addr_type> out_addrs;
  for (uint32_t r = 0; r < N; ++r) {
    for (uint32_t c = 0; c < K; c += static_cast<uint32_t>(elems_per_access)) {
      uint32_t col = std::min(c, K - 1);
      std::vector<uint32_t> index = {r, col};
      addr_type off = make_address(index, _matrix_shape);
      out_addrs.insert(out_base + off);
    }
  }

  if (!out_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .id = "SERIES_OUT",
        .dest_addr = addr_X,
        .size = static_cast<uint32_t>(out_addrs.size()),
        .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
        .base_addr = out_base,
        .tile_m = N,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = false,  // 从 SPAD 写回
        .last_inst = true,
        .my_tile = tile,
        .barrier_type = 4}));
  }

  if (tile->instructions.empty()) {
    spdlog::error("SeriesInverseOp: No instructions generated for Batch {} Core {}",
                  tile->batch, tile->core_id);
  }
}
