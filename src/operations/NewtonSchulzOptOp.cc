#include "NewtonSchulzOptOp.h"

#include "../Model.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>

#ifndef DEFAULT_BATCH_SIZE
#define DEFAULT_BATCH_SIZE 96
#endif

NewtonSchulzOptOp::NewtonSchulzOptOp(SimulationConfig config,
                                     Model* model,
                                     onnx::NodeProto& node_proto,
                                     uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "NewtonSchulzOpt";
  parse_attributes();
  infer_shapes_from_model();
}

NewtonSchulzOptOp::NewtonSchulzOptOp(SimulationConfig config,
                                     Model* model,
                                     const std::string& name,
                                     std::map<std::string, std::string>& attributes,
                                     uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "NewtonSchulzOpt";
  parse_attributes();
  infer_shapes_from_model();
}

NewtonSchulzOptOp::NewtonSchulzOptOp(SimulationConfig config,
                                     MappingTable& mapping_table,
                                     const std::vector<uint32_t>& matrix_shape,
                                     uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "NewtonSchulzOpt";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void NewtonSchulzOptOp::parse_attributes() {
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

void NewtonSchulzOptOp::infer_shapes_from_model() {
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

void NewtonSchulzOptOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  // Per-core super-tile with ping-pong (double buffering): each core
  // owns a single tile that processes all of its batches in
  // round-robin order (b = core_id, core_id + num_cores, ...). Within
  // that tile we use two SPAD regions (Ping/Pong) so that, while
  // computing batch i on one region, the MTE can preload batch i+1
  // into the other.

  if (_config.num_cores == 0) {
    spdlog::error("NewtonSchulzOptOp: Invalid core count 0!");
    return;
  }

  uint32_t created_tiles = 0;

  for (uint32_t core = 0; core < _config.num_cores; ++core) {
    // Check whether this core is assigned any batch at all.
    bool has_batch = false;
    for (uint32_t b = core; b < _batch_size; b += _config.num_cores) {
      has_batch = true;
      break;
    }
    if (!has_batch) continue;

    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = _optype,
        .layer_id = _id,
        .fused_op_id = 0,
        // For a per-core super-tile, "batch" is set to the first
        // logical batch handled by this core, but all of its
        // batches are driven by the instruction stream.
        .batch = core,
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
        .core_id = static_cast<int>(core),
        .inst_finished = false});

    initialize_instructions(tile.get(), Mapping{});

    if (!tile->instructions.empty()) {
      _tiles.push_back(std::move(tile));
      created_tiles++;
    }
  }

  spdlog::info(
      "NewtonSchulzOptOp '{}': Double-buffered ping-pong tiles {} across {} cores ({} "
      "logical batches).",
      _name, created_tiles, _config.num_cores, _batch_size);
}

void NewtonSchulzOptOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("NewtonSchulzOptOp: matrix shape not set for layer {}", _name);
    return;
  }

  // Double Check Core Validity
  if (tile->core_id < 0 || static_cast<uint32_t>(tile->core_id) >= _config.num_cores) {
    return;
  }

  const uint32_t N = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t K = _matrix_shape[_matrix_shape.size() - 1];

  const auto& core_cfg = _config.core_config[static_cast<uint32_t>(tile->core_id)];
  const bool use_cube_tiling = core_cfg.enable_ascend_cube_model;
  const uint32_t cube_m = std::max(1u, use_cube_tiling ? core_cfg.cube_m : core_cfg.core_height);
  const uint32_t cube_n = std::max(1u, use_cube_tiling ? core_cfg.cube_n : core_cfg.core_width);
  const uint32_t cube_k = std::max(1u, use_cube_tiling ? core_cfg.cube_k : core_cfg.core_width);

  // Calculate Sizes
  addr_type matrix_size_bytes = static_cast<addr_type>(N) * K * _config.precision;
  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  // ---------------------------------------------------------
  // Helper Lambda: MOVIN
  // ---------------------------------------------------------
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
          .my_tile = tile}));
    }
  };

  // ---------------------------------------------------------
  // Helper Lambda: MOVOUT
  // ---------------------------------------------------------
  auto emit_movout_full = [&](addr_type out_base, addr_type spad_src, bool last_inst) {
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
          .id = "NSOPT_OUT",
          .dest_addr = spad_src,
          .size = static_cast<uint32_t>(out_addrs.size()),
          .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
          .operand_id = _OUTPUT_OPERAND,
          .base_addr = out_base,
          .tile_m = N,
          .tile_k = K,
          .tile_n = K,
          .src_from_accum = true,
          .last_inst = last_inst,
          .my_tile = tile,
          .barrier_type = 4}));
    }
  };

  // ---------------------------------------------------------
  // Helper Lambda: Cube-style tiled GEMM (16x16x16 on Ascend model)
  // ---------------------------------------------------------
  auto emit_tiled_gemm = [&](const std::string& inst_id,
                             addr_type dest_addr,
                             const std::vector<addr_type>& src_addrs,
                             bool src_from_accum) {
    if (!use_cube_tiling) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::GEMM_PRELOAD,
          .id = inst_id,
          .dest_addr = dest_addr,
          .compute_size = K,
          .src_addrs = src_addrs,
          .tile_m = N,
          .tile_k = K,
          .tile_n = K,
          .src_from_accum = src_from_accum,
          .my_tile = tile}));
      return;
    }

    for (uint32_t m0 = 0; m0 < N; m0 += cube_m) {
      const uint32_t tile_m = std::min(cube_m, N - m0);
      for (uint32_t n0 = 0; n0 < K; n0 += cube_n) {
        const uint32_t tile_n = std::min(cube_n, K - n0);
        for (uint32_t k0 = 0; k0 < K; k0 += cube_k) {
          const uint32_t tile_k = std::min(cube_k, K - k0);
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::GEMM_PRELOAD,
              .id = inst_id,
              .dest_addr = dest_addr,
              .compute_size = tile_k,
              .src_addrs = src_addrs,
              .tile_m = tile_m,
              .tile_k = tile_k,
              .tile_n = tile_n,
              .src_from_accum = src_from_accum,
              .my_tile = tile}));
        }
      }
    }
  };

  // ---------------------------------------------------------
  // Helper Lambda: tiled vector ADD for R = C - T
  // ---------------------------------------------------------
  auto emit_tiled_add = [&](addr_type dest_addr, addr_type lhs_addr, addr_type rhs_addr) {
    if (!use_cube_tiling) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::ADD,
          .id = "NSOPT_R",
          .dest_addr = dest_addr,
          .compute_size = N * K,
          .src_addrs = std::vector<addr_type>{lhs_addr, rhs_addr},
          .tile_m = N,
          .tile_k = K,
          .tile_n = K,
          .src_from_accum = true,
          .my_tile = tile}));
      return;
    }

    for (uint32_t m0 = 0; m0 < N; m0 += cube_m) {
      const uint32_t tile_m = std::min(cube_m, N - m0);
      for (uint32_t n0 = 0; n0 < K; n0 += cube_n) {
        const uint32_t tile_n = std::min(cube_n, K - n0);
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .id = "NSOPT_R",
            .dest_addr = dest_addr,
            .compute_size = tile_m * tile_n,
            .src_addrs = std::vector<addr_type>{lhs_addr, rhs_addr},
            .tile_m = tile_m,
            .tile_k = tile_n,
            .tile_n = tile_n,
            .src_from_accum = true,
            .my_tile = tile}));
      }
    }
  };

  // ---------------------------------------------------------
  // 1. Identify Batches for this Core
  // ---------------------------------------------------------
  std::vector<uint32_t> local_batches;
  for (uint32_t b = static_cast<uint32_t>(tile->core_id); b < _batch_size;
       b += _config.num_cores) {
    local_batches.push_back(b);
  }
  if (local_batches.empty()) return;

  // ---------------------------------------------------------
  // 2. Address Layout
  // ---------------------------------------------------------
  // We allocate a dedicated [A, X] region in SPAD for every
  // local batch of this core so that each MOVIN destination
  // address is unique. This avoids the SPAD allocator's
  // "Destination allocated" panic when reusing the same
  // dest_addr across batches.
  //
  // SPAD (per-core view):
  //   [A(local_batches[0])][X(local_batches[0])]
  //   [A(local_batches[1])][X(local_batches[1])]
  //   ...
  //   [A(local_batches[L-1])][X(local_batches[L-1])]
  //   [R_Shared][C_Shared]
  // Accum: [T_Ping] | [T_Pong]

  addr_type c_base = get_operand_addr(_INPUT_OPERAND + 2);  // DRAM address for C

  // SPAD Addresses
  addr_type sp_base = SPAD_BASE;
  addr_type size_batch = 2 * matrix_size_bytes;  // A + X for one batch

  const size_t L = local_batches.size();
  std::vector<addr_type> addr_A_vec(L);
  std::vector<addr_type> addr_X_vec(L);

  for (size_t li = 0; li < L; ++li) {
    addr_type base = sp_base + static_cast<addr_type>(li) * size_batch;
    addr_A_vec[li] = base;
    addr_X_vec[li] = base + matrix_size_bytes;
  }

  addr_type addr_R = sp_base + static_cast<addr_type>(L) * size_batch;
  addr_type addr_C = addr_R + matrix_size_bytes;

  // Accumulator Addresses (Double Buffered T for ping-pong)
  addr_type ac_base = ACCUM_SPAD_BASE;
  addr_type addr_T_Ping = ac_base;
  addr_type addr_T_Pong = ac_base + matrix_size_bytes;

  // ---------------------------------------------------------
  // 3. Prologue: Preload Batch 0 into Ping
  // ---------------------------------------------------------
  {
    uint32_t b0 = local_batches[0];
    addr_type a0_dram = get_operand_addr(_INPUT_OPERAND + 0) +
                        static_cast<addr_type>(b0) * matrix_size_bytes;
    addr_type x0_dram = get_operand_addr(_INPUT_OPERAND + 1) +
                        static_cast<addr_type>(b0) * matrix_size_bytes;

    emit_movin_full(a0_dram, addr_A_vec[0], _INPUT_OPERAND + 0);
    emit_movin_full(x0_dram, addr_X_vec[0], _INPUT_OPERAND + 1);
    emit_movin_full(c_base, addr_C, _INPUT_OPERAND + 2);

    // Barrier: Wait for Prologue Load
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "NSOPT_BARRIER_PROLOGUE",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 1}));
  }

  // ---------------------------------------------------------
  // 4. Main Loop: Compute(i) || Load(i+1)
  // ---------------------------------------------------------
  for (size_t li = 0; li < local_batches.size(); ++li) {
    bool use_ping = (li % 2 == 0);  // Toggle buffers based on iteration
    bool has_next = (li + 1 < local_batches.size());

    // Current Buffers (unique per local batch)
    addr_type cur_A = addr_A_vec[li];
    addr_type cur_X = addr_X_vec[li];
    addr_type cur_T = use_ping ? addr_T_Ping : addr_T_Pong;  // Use Ping/Pong Accumulator

    // A. Issue Load for Next Batch (Background MTE)
    if (has_next) {
      uint32_t b_next = local_batches[li + 1];
      addr_type a_next_dram = get_operand_addr(_INPUT_OPERAND + 0) +
                              static_cast<addr_type>(b_next) * matrix_size_bytes;
      addr_type x_next_dram = get_operand_addr(_INPUT_OPERAND + 1) +
                              static_cast<addr_type>(b_next) * matrix_size_bytes;

      addr_type next_A = addr_A_vec[li + 1];
      addr_type next_X = addr_X_vec[li + 1];

      emit_movin_full(a_next_dram, next_A, _INPUT_OPERAND + 0);
      emit_movin_full(x_next_dram, next_X, _INPUT_OPERAND + 1);
    }

    // B. Compute Current Batch (Foreground Cube)
    for (uint32_t iter = 0; iter < _iterations; ++iter) {
      addr_type x_src = (iter == 0) ? cur_X : cur_T;
      bool use_accum = (iter > 0);

      // T = A * X
      emit_tiled_gemm("NSOPT_T", cur_T, std::vector<addr_type>{cur_A, x_src}, use_accum);

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 2}));

      // R = C - T
      emit_tiled_add(addr_R, addr_C, cur_T);

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 3}));

      // X_new = X * R
      emit_tiled_gemm("NSOPT_X", cur_T, std::vector<addr_type>{x_src, addr_R}, use_accum);
    }

    // ---------------------------------------------------------
    // C. Barrier: Protect Next Loop's Inputs
    // ---------------------------------------------------------
    // We only need to wait for "Load Next" to finish before "Compute Next" starts.
    if (has_next) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .id = "NSOPT_BARRIER_NEXT_LOAD",
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 1}));
    }

    // D. Issue Store for Current Batch (Background MTE)
    // This issues after the barrier, so it runs in parallel with the next
    // compute's startup. Since we use T_Ping/T_Pong separately, there is no
    // hazard between store(cur_T) and compute(next_T).
    uint32_t b_cur = local_batches[li];
    addr_type out_dram = get_operand_addr(_OUTPUT_OPERAND + 0) +
                         static_cast<addr_type>(b_cur) * matrix_size_bytes;
    bool is_last = (li + 1 == local_batches.size());
    emit_movout_full(out_dram, cur_T, is_last);
  }

  if (tile->instructions.empty()) {
    spdlog::error(
        "NewtonSchulzOptOp: No instructions generated for Core {} in layer {}",
        tile->core_id, _name);
  }
}
