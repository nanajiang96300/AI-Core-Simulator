#include "MMSEOp.h"

#include "NewtonSchulzOp.h"  // for comments / consistency only
#include "../Model.h"
#include <numeric>

MMSEOp::MMSEOp(SimulationConfig config,
               Model* model,
               const std::string& name,
               std::map<std::string, std::string>& attributes,
               uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "MMSEOp";
  parse_attributes();
  infer_shapes_from_model();
}

MMSEOp::MMSEOp(SimulationConfig config,
               MappingTable& mapping_table,
               const std::vector<uint32_t>& matrix_shape,
               uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "MMSEOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void MMSEOp::parse_attributes() {
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
      _batch_size = 96;
    }
  } else {
    _batch_size = 96;
  }
}

void MMSEOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;

  if (_inputs.size() >= 1) {
    Tensor* h_tensor = _model->get_tensor(_inputs[0]);
    if (h_tensor) {
      std::vector<uint32_t> dims = h_tensor->get_dims();
      if (dims.size() == 3) {
        _batch_size = dims[0];
        _matrix_shape = {dims[1], dims[2]};
      } else {
        _matrix_shape = dims;
      }
    }
  }
}

void MMSEOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("MMSEOp: Invalid core count 0!");
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

  spdlog::info("MMSEOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  spdlog::info(
      "  > Load Distribution (First 4 cores): Core0: {}, Core1: {}, Core2: {}, Core3: {} ...",
      core_load[0], core_load[1], core_load[2], core_load[3]);
}

void MMSEOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("MMSEOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t K = _matrix_shape[_matrix_shape.size() - 1];
  // Byte sizes for different matrix shapes.
  addr_type size_mk = static_cast<addr_type>(M) * K * _config.precision;   // [M,K]
  addr_type size_kk = static_cast<addr_type>(K) * K * _config.precision;   // [K,K]
  addr_type size_km = static_cast<addr_type>(K) * M * _config.precision;   // [K,M]

  addr_type batch_offset_mk = static_cast<addr_type>(tile->batch) * size_mk;
  addr_type batch_offset_kk = static_cast<addr_type>(tile->batch) * size_kk;

  // Inputs (per batch):
  //  0: H          [M, K]
  //  1: X_init_32  [K, K]
  //  2: C_32       [K, K] (2I or sigma^2 I), broadcast across batch
  //  3: Y          [M, K]
  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mk;
  addr_type x0_base = get_operand_addr(_INPUT_OPERAND + 1) + batch_offset_kk;
  addr_type c_base = get_operand_addr(_INPUT_OPERAND + 2);  // broadcast [K,K]
  addr_type y_base = get_operand_addr(_INPUT_OPERAND + 3) + batch_offset_mk;

  // Output: X_hat [K, K]
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_kk;

  // SPAD layout (per core)
  addr_type addr_H = SPAD_BASE;                 // [M,K]
  addr_type addr_Y = addr_H + size_mk;          // [M,K]
  addr_type addr_G = addr_Y + size_mk;          // [K,K]   = H^H H
  addr_type addr_Gtilde = addr_G + size_kk;     // [K,K]   = G + sigma^2 I
  addr_type addr_X0 = addr_Gtilde + size_kk;    // [K,K]   initial X0
  addr_type addr_C32 = addr_X0 + size_kk;       // [K,K]   2I (or sigma^2 I)
  addr_type addr_R32 = addr_C32 + size_kk;      // [K,K]   NS residual
  addr_type addr_W = addr_R32 + size_kk;        // [K,M]   = G_inv * H^H

  addr_type addr_T32 = ACCUM_SPAD_BASE;         // [K,K] in ACCUM (X_k, G_inv, X_hat)

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  std::vector<uint32_t> shape_mk = _matrix_shape;         // [M,K]
  std::vector<uint32_t> shape_kk{K, K};                   // [K,K]

  auto emit_movin_mk = [&](addr_type dram_base, addr_type spad_dest,
                           uint32_t operand_id, bool broadcast) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < M; ++r) {
      for (uint32_t c = 0; c < K; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, K - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, shape_mk);
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
          .tile_m = M,
          .tile_k = K,
          .tile_n = 0,
          .my_tile = tile}));
    }
  };

  auto emit_movin_kk = [&](addr_type dram_base, addr_type spad_dest,
                           uint32_t operand_id, bool broadcast) {
    (void)broadcast;  // for now, broadcast vs non-broadcast share the same pattern
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < K; ++r) {
      for (uint32_t c = 0; c < K; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, K - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, shape_kk);
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
          .tile_m = K,
          .tile_k = K,
          .tile_n = 0,
          .my_tile = tile}));
    }
  };

  // ========================
  // Load phase: H, X_init_32, C_32, Y
  // ========================
  emit_movin_mk(h_base, addr_H, _INPUT_OPERAND + 0, /*broadcast=*/false);
  emit_movin_kk(x0_base, addr_X0, _INPUT_OPERAND + 1, /*broadcast=*/false);
  emit_movin_kk(c_base, addr_C32, _INPUT_OPERAND + 2, /*broadcast=*/true);
  emit_movin_mk(y_base, addr_Y, _INPUT_OPERAND + 3, /*broadcast=*/false);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "MMSE_BARRIER_LOAD2HTH",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  // ========================
  // Phase 1: Gram matrix G = H^H H (KxK)
  // ========================
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "MMSE_HtH",
      .dest_addr = addr_G,
      .compute_size = K,
      .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = K,   // rows of H^H
      .tile_k = M,   // shared dim
      .tile_n = K,   // cols of H
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "MMSE_BARRIER_HTH2ADD",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 2}));

  // ========================
  // Phase 2: G_tilde = G + sigma^2 I  (32x32 add)
  // ========================
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "MMSE_G_PLUS_SIGMA",
      .dest_addr = addr_Gtilde,
      .compute_size = K * K,
      .src_addrs = std::vector<addr_type>{addr_G, addr_C32},
      .tile_m = K,
      .tile_k = K,
      .tile_n = K,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "MMSE_BARRIER_G2INV",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 3}));

  // ========================
  // Phase 3: 32x32 Newton-Schulz inverse on G_tilde
  //          X_{k+1} = X_k (2I - G_tilde * X_k)
  // ========================
  for (uint32_t iter = 0; iter < _iterations; ++iter) {
    addr_type x_src = (iter == 0) ? addr_X0 : addr_T32;
    bool use_accum_for_x = (iter > 0);

    // T = G_tilde * X_k
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "MMSE_INV_T",
        .dest_addr = addr_T32,
        .compute_size = K,
        .src_addrs = std::vector<addr_type>{addr_Gtilde, x_src},
        .tile_m = K,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = use_accum_for_x,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "MMSE_BARRIER_INV_CUBE2VEC",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 4}));

    // R32 = C32 - T
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "MMSE_INV_R",
        .dest_addr = addr_R32,
        .compute_size = K * K,
        .src_addrs = std::vector<addr_type>{addr_C32, addr_T32},
        .tile_m = K,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = true,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "MMSE_BARRIER_INV_VEC2CUBE",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));

    // X_{k+1} = X_k * R32
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "MMSE_INV_X",
        .dest_addr = addr_T32,
        .compute_size = K,
        .src_addrs = std::vector<addr_type>{x_src, addr_R32},
        .tile_m = K,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = use_accum_for_x,
        .my_tile = tile}));
  }

  // Now addr_T32 in ACCUM holds G_inv (KxK).
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "MMSE_BARRIER_INV2W",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 6}));

  // ========================
  // Phase 4: W = G_inv * H^H   (KxM)
  // ========================
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "MMSE_WH",
      .dest_addr = addr_W,
      .compute_size = M,
      .src_addrs = std::vector<addr_type>{addr_T32, addr_H},
      .tile_m = K,
      .tile_k = K,
      .tile_n = M,
      .src_from_accum = true,  // wait for G_inv in ACCUM
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "MMSE_BARRIER_W2Y",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 7}));

  // ========================
  // Phase 5: X_hat = W * Y    (KxK)
  // ========================
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "MMSE_WY",
      .dest_addr = addr_T32,
      .compute_size = K,
      .src_addrs = std::vector<addr_type>{addr_W, addr_Y},
      .tile_m = K,
      .tile_k = M,
      .tile_n = K,
      .src_from_accum = false,
      .my_tile = tile}));

  // ========================
  // Store phase: write back [K, K] result from ACCUM SPAD.
  // ========================
  std::set<addr_type> out_addrs;
  for (uint32_t r = 0; r < K; ++r) {
    for (uint32_t c = 0; c < K; c += static_cast<uint32_t>(elems_per_access)) {
      uint32_t col = std::min(c, K - 1);
      std::vector<uint32_t> index = {r, col};
      addr_type off = make_address(index, shape_kk);
      out_addrs.insert(out_base + off);
    }
  }

  if (!out_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .id = "MMSE_OUT",
        .dest_addr = addr_T32,
        .size = static_cast<uint32_t>(out_addrs.size()),
        .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
        .base_addr = out_base,
        .tile_m = K,
        .tile_k = K,
        .tile_n = K,
        .src_from_accum = true,
        .last_inst = true,
        .my_tile = tile,
        .barrier_type = 8}));
  }

  if (tile->instructions.empty()) {
    spdlog::error("MMSEOp: No instructions generated for Batch {} Core {}",
                  tile->batch, tile->core_id);
  }
}
