#include "LDLDecompOp.h"

#include "../Model.h"

#include <algorithm>

LDLDecompOp::LDLDecompOp(SimulationConfig config,
                         Model* model,
                         const std::string& name,
                         std::map<std::string, std::string>& attributes,
                         uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "LDLDecompOp";
  parse_attributes();
  infer_shapes_from_model();
}

LDLDecompOp::LDLDecompOp(SimulationConfig config,
                         MappingTable& mapping_table,
                         const std::vector<uint32_t>& matrix_shape,
                         uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "LDLDecompOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void LDLDecompOp::parse_attributes() {
  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try {
      _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second));
    } catch (...) {
      _batch_size = 96;
    }
  }

  auto it_blk = _attributes.find("block_size");
  if (it_blk != _attributes.end()) {
    try {
      _block_size = std::max(1u, static_cast<uint32_t>(std::stoul(it_blk->second)));
    } catch (...) {
      _block_size = 2;
    }
  }

  auto it_bwd = _attributes.find("bwd_steps");
  if (it_bwd != _attributes.end()) {
    try {
      _bwd_steps = std::max(1u, static_cast<uint32_t>(std::stoul(it_bwd->second)));
    } catch (...) {
      _bwd_steps = 1;
    }
  }

  auto it_pack = _attributes.find("pack_blocks");
  if (it_pack != _attributes.end()) {
    try {
      _pack_blocks = static_cast<uint32_t>(std::stoul(it_pack->second));
    } catch (...) {
      _pack_blocks = 0;
    }
  }
}

void LDLDecompOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;

  if (_inputs.empty()) return;

  Tensor* h_tensor = _model->get_tensor(_inputs[0]);
  if (!h_tensor) return;

  std::vector<uint32_t> dims = h_tensor->get_dims();
  if (dims.size() == 3) {
    _batch_size = dims[0];
    _matrix_shape = {dims[1], dims[2]};
  } else {
    _matrix_shape = dims;
  }
}

void LDLDecompOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("LDLDecompOp: Invalid core count 0!");
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

  spdlog::info("LDLDecompOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  std::string load_msg = "  > Load Distribution:";
  for (uint32_t core = 0; core < _config.num_cores; ++core) {
    load_msg += " Core" + std::to_string(core) + ": " + std::to_string(core_load[core]);
  }
  spdlog::info("{}", load_msg);
}

void LDLDecompOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("LDLDecompOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t U = _matrix_shape[_matrix_shape.size() - 1];

  const uint32_t blk = std::max(1u, _block_size);
  const uint32_t n_blocks = std::max(1u, U / blk);

  uint32_t cube_dim_target = std::min(_config.core_config[tile->core_id].core_height,
                                      _config.core_config[tile->core_id].core_width);
  if (_config.core_config[tile->core_id].enable_ascend_cube_model) {
    cube_dim_target = std::min(
        cube_dim_target,
        std::min(_config.core_config[tile->core_id].cube_m,
                 std::min(_config.core_config[tile->core_id].cube_n,
                          _config.core_config[tile->core_id].cube_k)));
  }
  cube_dim_target = std::max(cube_dim_target, blk);
  const uint32_t auto_pack_blocks =
      (blk == 2) ? std::min(2u, std::max(1u, cube_dim_target / blk)) : 1u;
  const uint32_t cube_pack_blocks = (_pack_blocks > 0) ? _pack_blocks : auto_pack_blocks;

  addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;

  addr_type batch_offset_mu = static_cast<addr_type>(tile->batch) * size_mu;
  addr_type batch_offset_uu = static_cast<addr_type>(tile->batch) * size_uu;

  // Inputs:
  //  0: H      [B, M, U]
  //  1: RegI   [U, U] or [B, U, U] (broadcast-like usage in this model)
  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mu;
  addr_type reg_base = get_operand_addr(_INPUT_OPERAND + 1);

  // Output:
  //  0: A_inv  [B, U, U]
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_uu;

  // SPAD layout
  addr_type addr_H = SPAD_BASE;              // [M,U]
  addr_type addr_Reg = addr_H + size_mu;     // [U,U]
  addr_type addr_G = addr_Reg + size_uu;     // [U,U], Gram
  addr_type addr_A = addr_G + size_uu;       // [U,U], regularized
  addr_type addr_Dinv = addr_A + size_uu;    // [U,U], block inverse cache
  addr_type addr_tmp = addr_Dinv + size_uu;  // [U,U], scratch
  addr_type addr_Ainv = ACCUM_SPAD_BASE;     // [U,U], final accumulator

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  std::vector<uint32_t> shape_mu{M, U};
  std::vector<uint32_t> shape_uu{U, U};

  auto emit_movin = [&](addr_type dram_base,
                        addr_type spad_dest,
                        uint32_t rows,
                        uint32_t cols,
                        const std::vector<uint32_t>& shape,
                        uint32_t operand_id) {
    std::set<addr_type> addrs;
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; c += static_cast<uint32_t>(elems_per_access)) {
        uint32_t col = std::min(c, cols - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, shape);
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
          .tile_m = rows,
          .tile_k = cols,
          .tile_n = 0,
          .my_tile = tile}));
    }
  };

  auto pick_mul_opcode = [&](uint32_t tile_m, uint32_t tile_k, uint32_t tile_n) {
    if (tile_m <= 2 && tile_k <= 2 && tile_n <= 2) {
      return Opcode::MAC;
    }
    return Opcode::GEMM_PRELOAD;
  };

  auto pick_ldl_step_mul_opcode = [&](uint32_t tile_m, uint32_t tile_k, uint32_t tile_n) {
    if (blk == 1) {
      return Opcode::MAC;
    }
    return pick_mul_opcode(tile_m, tile_k, tile_n);
  };

  auto pick_ldl_micro_mul_opcode = [&](uint32_t tile_m, uint32_t tile_k, uint32_t tile_n) {
    if (blk <= 2 && tile_m <= 2 && tile_n <= 2) {
      return Opcode::MAC;
    }
    return pick_ldl_step_mul_opcode(tile_m, tile_k, tile_n);
  };

  // 1) Load H and regularizer.
  emit_movin(h_base, addr_H, M, U, shape_mu, _INPUT_OPERAND + 0);
  emit_movin(reg_base, addr_Reg, U, U, shape_uu, _INPUT_OPERAND + 1);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "LDL_BARRIER_LOAD2GRAM",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  // 2) Gram: G = H^H H
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "LDL_GRAM",
      .dest_addr = addr_G,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "LDL_BARRIER_GRAM2REG",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 2}));

  // 3) Regularization: A = G + lambda I
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "LDL_REG",
      .dest_addr = addr_A,
      .compute_size = U * U,
      .src_addrs = std::vector<addr_type>{addr_G, addr_Reg},
      .tile_m = U,
      .tile_k = U,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "LDL_BARRIER_REG2BLDL",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 3}));

  // 4) Block-LDL decomposition pattern (2x2 style by default).
  for (uint32_t j = 0; j < n_blocks; ++j) {
    const uint32_t d_update_k_len = (blk > 1) ? U : blk;
    const Opcode d_update_opcode = pick_ldl_micro_mul_opcode(blk, d_update_k_len, blk);
    const uint32_t d_update_compute =
      (d_update_opcode == Opcode::MAC) ? d_update_k_len : blk;
    // D_jj update must always produce addr_Ainv before LDL_D_INV_j consumes it.
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = d_update_opcode,
      .id = "LDL_D_UPDATE_" + std::to_string(j),
      .dest_addr = addr_Ainv,
      .compute_size = d_update_compute,
      .src_addrs = std::vector<addr_type>{addr_A, addr_A},
      .tile_m = blk,
      .tile_k = d_update_k_len,
      .tile_n = blk,
      .src_from_accum = false,
      .my_tile = tile}));

    // Aligned inv*mul path: first compute reciprocal term, then multiply.
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::SCALAR_DIV,
      .id = "LDL_D_DIAG_INV_" + std::to_string(j),
      .dest_addr = addr_tmp,
      .compute_size = blk * blk,
      .src_addrs = std::vector<addr_type>{addr_Reg, addr_Ainv},
      .tile_m = blk,
      .tile_k = blk,
      .tile_n = blk,
      .src_from_accum = true,
      .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MUL,
      .id = "LDL_D_INV_MUL_" + std::to_string(j),
      .dest_addr = addr_Dinv,
      .compute_size = blk * blk,
      .src_addrs = std::vector<addr_type>{addr_Ainv, addr_tmp},
      .tile_m = blk,
      .tile_k = blk,
      .tile_n = blk,
      .src_from_accum = true,
      .my_tile = tile}));

        for (uint32_t i = j + 1; i < n_blocks; i += cube_pack_blocks) {
      const uint32_t packed_blocks = std::min(cube_pack_blocks, n_blocks - i);
      const uint32_t packed_dim = blk * packed_blocks;
      const Opcode l_upd_opcode = pick_ldl_step_mul_opcode(packed_dim, packed_dim, packed_dim);
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = l_upd_opcode,
          .id = "LDL_L_UPDATE_" + std::to_string(i) + "_" + std::to_string(j) +
            "_PACK" + std::to_string(packed_blocks),
          .dest_addr = addr_tmp,
          .compute_size = packed_dim,
          .src_addrs = std::vector<addr_type>{addr_A, addr_Dinv},
          .tile_m = packed_dim,
          .tile_k = packed_dim,
          .tile_n = packed_dim,
          .src_from_accum = false,
          .my_tile = tile}));
    }

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "LDL_BARRIER_BLDL_STEP_" + std::to_string(j),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 4}));
  }

  // 5) Backward-substitution-based inverse assembly.
  // Match the algorithmic dependency used in evaluation code:
  // for each (i, j), accumulation must include all lower blocks k=i+1..n_blocks-1.
  for (int32_t col = static_cast<int32_t>(n_blocks) - 1; col >= 0; --col) {
    uint32_t j = static_cast<uint32_t>(col);

    // Diagonal block update x_jj: collapse sum over k into one long-K GEMM.
    const uint32_t diag_k_blocks = (n_blocks > (j + 1)) ? (n_blocks - (j + 1)) : 0;
    if (diag_k_blocks > 0) {
      const uint32_t diag_k_len = diag_k_blocks * blk;
      const Opcode diag_mul_opcode = pick_ldl_micro_mul_opcode(blk, diag_k_len, blk);
      const uint32_t diag_mul_compute = (diag_mul_opcode == Opcode::MAC) ? diag_k_len : blk;
      for (uint32_t rep = 0; rep < _bwd_steps; ++rep) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = diag_mul_opcode,
            .id = "LDL_BWD_DIAG_MUL_" + std::to_string(j) + "_" + std::to_string(rep),
            .dest_addr = addr_tmp,
            .compute_size = diag_mul_compute,
            .src_addrs = std::vector<addr_type>{addr_Dinv, addr_A},
            .tile_m = blk,
            .tile_k = diag_k_len,
            .tile_n = blk,
            .src_from_accum = false,
            .my_tile = tile}));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .id = "LDL_BWD_DIAG_ACC_" + std::to_string(j) + "_" + std::to_string(rep),
            .dest_addr = addr_Ainv,
            .compute_size = blk,
            .src_addrs = std::vector<addr_type>{addr_Dinv, addr_tmp},
            .tile_m = blk,
            .tile_k = blk,
            .tile_n = blk,
            .src_from_accum = false,
            .my_tile = tile}));
      }
    }

    // Ensure x_jj is fully produced before any x_ij reads it (RAW-safe).
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "LDL_BARRIER_BWD_DIAG2OFF_" + std::to_string(j),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));

    // Off-diagonal block update x_ij (i<j): collapse sum over k into one long-K GEMM.
    for (int32_t i = col - 1; i >= 0; --i) {
      uint32_t iu = static_cast<uint32_t>(i);
      const uint32_t off_k_blocks = (n_blocks > (iu + 1)) ? (n_blocks - (iu + 1)) : 0;
      if (off_k_blocks == 0) continue;
      const uint32_t off_k_len = off_k_blocks * blk;
      const Opcode off_mul_opcode = pick_ldl_micro_mul_opcode(blk, off_k_len, blk);
      const uint32_t off_mul_compute = (off_mul_opcode == Opcode::MAC) ? off_k_len : blk;
      for (uint32_t rep = 0; rep < _bwd_steps; ++rep) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = off_mul_opcode,
            .id = "LDL_BWD_OFF_MUL_" + std::to_string(iu) + "_" + std::to_string(j) + "_" +
                  std::to_string(rep),
            .dest_addr = addr_tmp,
        .compute_size = off_mul_compute,
            .src_addrs = std::vector<addr_type>{addr_Dinv, addr_A},
            .tile_m = blk,
            .tile_k = off_k_len,
            .tile_n = blk,
            .src_from_accum = false,
            .my_tile = tile}));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADD,
            .id = "LDL_BWD_OFF_ACC_" + std::to_string(iu) + "_" + std::to_string(j) + "_" +
                  std::to_string(rep),
            .dest_addr = addr_Ainv,
            .compute_size = blk,
            .src_addrs = std::vector<addr_type>{addr_Ainv, addr_tmp},
            .tile_m = blk,
            .tile_k = blk,
            .tile_n = blk,
            .src_from_accum = true,
            .my_tile = tile}));
      }
    }

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "LDL_BARRIER_BWD_COL_" + std::to_string(j),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "LDL_BARRIER_BWD2STORE",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 5}));

  std::set<addr_type> out_addrs;
  for (uint32_t r = 0; r < U; ++r) {
    for (uint32_t c = 0; c < U; c += static_cast<uint32_t>(elems_per_access)) {
      uint32_t col = std::min(c, U - 1);
      std::vector<uint32_t> index = {r, col};
      addr_type off = make_address(index, shape_uu);
      out_addrs.insert(out_base + off);
    }
  }

  if (!out_addrs.empty()) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .id = "LDL_OUT",
        .dest_addr = addr_Ainv,
        .size = static_cast<uint32_t>(out_addrs.size()),
        .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
        .base_addr = out_base,
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = true,
        .last_inst = true,
        .my_tile = tile,
        .barrier_type = 6}));
  }

  if (tile->instructions.empty()) {
    spdlog::error("LDLDecompOp: No instructions generated for Batch {} Core {}",
                  tile->batch, tile->core_id);
  }
}
