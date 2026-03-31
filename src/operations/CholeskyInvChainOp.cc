#include "CholeskyInvChainOp.h"

#include "../Model.h"

#include <algorithm>

CholeskyInvChainOp::CholeskyInvChainOp(SimulationConfig config,
                                       Model* model,
                                       const std::string& name,
                                       std::map<std::string, std::string>& attributes,
                                       uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "CholeskyInvChainOp";
  parse_attributes();
  infer_shapes_from_model();
}

CholeskyInvChainOp::CholeskyInvChainOp(SimulationConfig config,
                                       MappingTable& mapping_table,
                                       const std::vector<uint32_t>& matrix_shape,
                                       uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "CholeskyInvChainOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void CholeskyInvChainOp::parse_attributes() {
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

  auto it_solve = _attributes.find("solve_steps");
  if (it_solve != _attributes.end()) {
    try {
      _solve_steps = std::max(1u, static_cast<uint32_t>(std::stoul(it_solve->second)));
    } catch (...) {
      _solve_steps = 1;
    }
  }
}

void CholeskyInvChainOp::infer_shapes_from_model() {
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

void CholeskyInvChainOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("CholeskyInvChainOp: Invalid core count 0!");
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

  spdlog::info("CholeskyInvChainOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  std::string load_msg = "  > Load Distribution:";
  for (uint32_t core = 0; core < _config.num_cores; ++core) {
    load_msg += " Core" + std::to_string(core) + ": " + std::to_string(core_load[core]);
  }
  spdlog::info("{}", load_msg);
}

void CholeskyInvChainOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("CholeskyInvChainOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t U = _matrix_shape[_matrix_shape.size() - 1];

  const uint32_t blk = std::max(1u, _block_size);
  const uint32_t n_blocks = std::max(1u, (U + blk - 1) / blk);
  const uint32_t scalar_blk = 1;

  addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;
  addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;

  addr_type batch_offset_mu = static_cast<addr_type>(tile->batch) * size_mu;
  addr_type batch_offset_uu = static_cast<addr_type>(tile->batch) * size_uu;

  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mu;
  addr_type reg_base = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_uu;

  addr_type addr_H = SPAD_BASE;
  addr_type addr_Reg = addr_H + size_mu;
  addr_type addr_G = addr_Reg + size_uu;
  addr_type addr_A = addr_G + size_uu;
  addr_type addr_L = addr_A + size_uu;
  addr_type addr_tmp = addr_L + size_uu;
  addr_type addr_inv = addr_tmp + size_uu;
  addr_type addr_Y = addr_inv + size_uu;
  addr_type addr_Ainv = ACCUM_SPAD_BASE;

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

  emit_movin(h_base, addr_H, M, U, shape_mu, _INPUT_OPERAND + 0);
  emit_movin(reg_base, addr_Reg, U, U, shape_uu, _INPUT_OPERAND + 1);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "CHOL_CHAIN_BARRIER_LOAD2GRAM",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "CHOL_CHAIN_GRAM",
      .dest_addr = addr_G,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "CHOL_CHAIN_REG",
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
      .id = "CHOL_CHAIN_BARRIER_REG2FACTOR",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 3}));

  for (uint32_t jb = 0; jb < n_blocks; ++jb) {
    const uint32_t j_start = jb * blk;
    const uint32_t j_end = std::min(U, j_start + blk);

    for (uint32_t j = j_start; j < j_end; ++j) {
      for (uint32_t k = 0; k < j; ++k) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MAC,
            .id = "CHOL_CHAIN_POTRF_DIAG_UPD_" + std::to_string(j) + "_" + std::to_string(k),
            .dest_addr = addr_A,
            .compute_size = scalar_blk,
            .src_addrs = std::vector<addr_type>{addr_L, addr_L},
            .tile_m = scalar_blk,
            .tile_k = scalar_blk,
            .tile_n = scalar_blk,
            .src_from_accum = false,
            .my_tile = tile}));
      }

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_SQRT,
          .id = "CHOL_CHAIN_POTRF_DIAG_SQRT_" + std::to_string(j),
          .dest_addr = addr_L,
          .compute_size = scalar_blk,
          .src_addrs = std::vector<addr_type>{addr_A},
          .tile_m = scalar_blk,
          .tile_k = scalar_blk,
          .tile_n = scalar_blk,
          .src_from_accum = false,
          .my_tile = tile}));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = "CHOL_CHAIN_TRSM_DIAG_INV_" + std::to_string(j),
          .dest_addr = addr_inv,
          .compute_size = scalar_blk,
          .src_addrs = std::vector<addr_type>{addr_Reg, addr_L},
          .tile_m = scalar_blk,
          .tile_k = scalar_blk,
          .tile_n = scalar_blk,
          .src_from_accum = false,
          .my_tile = tile}));

      for (uint32_t i = j + 1; i < U; ++i) {
        for (uint32_t k = 0; k < j; ++k) {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MAC,
              .id = "CHOL_CHAIN_TRSM_NUM_UPD_" + std::to_string(i) + "_" +
                    std::to_string(j) + "_" + std::to_string(k),
              .dest_addr = addr_A,
              .compute_size = scalar_blk,
              .src_addrs = std::vector<addr_type>{addr_L, addr_L},
              .tile_m = scalar_blk,
              .tile_k = scalar_blk,
              .tile_n = scalar_blk,
              .src_from_accum = false,
              .my_tile = tile}));
        }

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .id = "CHOL_CHAIN_TRSM_MUL_" + std::to_string(i) + "_" + std::to_string(j),
            .dest_addr = addr_L,
            .compute_size = scalar_blk,
            .src_addrs = std::vector<addr_type>{addr_A, addr_inv},
            .tile_m = scalar_blk,
            .tile_k = scalar_blk,
            .tile_n = scalar_blk,
            .src_from_accum = false,
            .my_tile = tile}));
      }

      for (uint32_t i = j + 1; i < U; ++i) {
        for (uint32_t k = i; k < U; ++k) {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MAC,
              .id = "CHOL_CHAIN_RK_UPDATE_" + std::to_string(i) + "_" + std::to_string(k) +
                    "_" + std::to_string(j),
              .dest_addr = addr_A,
              .compute_size = scalar_blk,
              .src_addrs = std::vector<addr_type>{addr_L, addr_L},
              .tile_m = scalar_blk,
              .tile_k = scalar_blk,
              .tile_n = scalar_blk,
              .src_from_accum = false,
              .my_tile = tile}));
        }
      }

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .id = "CHOL_CHAIN_BARRIER_FACTOR_STEP_" + std::to_string(j),
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 4}));
    }

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "CHOL_CHAIN_BARRIER_FACTOR_BLOCK_" + std::to_string(jb),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 4}));
  }

  for (uint32_t cb = 0; cb < n_blocks; ++cb) {
    const uint32_t c_start = cb * blk;
    const uint32_t c_end = std::min(U, c_start + blk);
    for (uint32_t c = c_start; c < c_end; ++c) {
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::SCALAR_DIV,
          .id = "CHOL_CHAIN_FWD_DIAG_INV_" + std::to_string(c),
          .dest_addr = addr_Y,
          .compute_size = scalar_blk,
          .src_addrs = std::vector<addr_type>{addr_Reg, addr_L},
          .tile_m = scalar_blk,
          .tile_k = scalar_blk,
          .tile_n = scalar_blk,
          .src_from_accum = false,
          .my_tile = tile}));

      for (uint32_t i = c + 1; i < U; ++i) {
        const uint32_t k_len = i - c;
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MAC,
            .id = "CHOL_CHAIN_FWD_OFF_MAC_" + std::to_string(i) + "_" + std::to_string(c),
            .dest_addr = addr_tmp,
            .compute_size = k_len,
            .src_addrs = std::vector<addr_type>{addr_L, addr_Y},
            .tile_m = scalar_blk,
            .tile_k = k_len,
            .tile_n = scalar_blk,
            .src_from_accum = false,
            .my_tile = tile}));

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .id = "CHOL_CHAIN_FWD_OFF_MUL_" + std::to_string(i) + "_" + std::to_string(c),
            .dest_addr = addr_Y,
            .compute_size = scalar_blk,
            .src_addrs = std::vector<addr_type>{addr_tmp, addr_Y},
            .tile_m = scalar_blk,
            .tile_k = scalar_blk,
            .tile_n = scalar_blk,
            .src_from_accum = false,
            .my_tile = tile}));
      }

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::PIPE_BARRIER,
          .id = "CHOL_CHAIN_BARRIER_FWD_COL_" + std::to_string(c),
          .my_tile = tile,
          .is_barrier = true,
          .barrier_type = 5}));
    }

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "CHOL_CHAIN_BARRIER_FWD_BLOCK_" + std::to_string(cb),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));
  }

  for (uint32_t rep = 0; rep < _solve_steps; ++rep) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "CHOL_CHAIN_BWD_MAC_FULL_" + std::to_string(rep),
        .dest_addr = addr_Ainv,
        .compute_size = U,
        .src_addrs = std::vector<addr_type>{addr_Y, addr_Y},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "CHOL_CHAIN_BARRIER_SOLVE2STORE",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 6}));

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
        .id = "CHOL_CHAIN_OUT",
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
    spdlog::error("CholeskyInvChainOp: No instructions generated for Batch {} Core {}",
                  tile->batch, tile->core_id);
  }
}
