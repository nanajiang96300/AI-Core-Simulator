#include "LSEstimatorOp.h"

#include "../Model.h"
#include "../Tensor.h"

LSEstimatorOp::LSEstimatorOp(SimulationConfig config, Model* model,
                             const std::string& name, uint32_t target_core)
    : Operation(config, model, name, *new std::map<std::string, std::string>(),
                target_core) {
  // Hard-code shapes
  _optype = "LSEstimatorOp";
  _a_shape = {32, 32};   // A: [M, K]
  _b_shape = {32, 512};  // B: [K, N]
  _c_shape = {32, 512};  // C: [M, N]

  uint32_t root_id = _model->get_root_node_id();

  // Create input tensor A
  auto a_tensor = std::make_unique<Tensor>(
      root_id, name_gen(name, "A"), _a_shape, _config.precision, true);
  a_tensor->set_produced();
  uint32_t a_id = a_tensor->get_id();
  _model->add_tensor(std::move(a_tensor));
  add_input(a_id);

  // Create input tensor B
  auto b_tensor = std::make_unique<Tensor>(
      root_id, name_gen(name, "B"), _b_shape, _config.precision, true);
  b_tensor->set_produced();
  uint32_t b_id = b_tensor->get_id();
  _model->add_tensor(std::move(b_tensor));
  add_input(b_id);

  // Create output tensor C
  auto c_tensor = std::make_unique<Tensor>(
      root_id, name_gen(name, "C"), _c_shape, _config.precision, false);
  uint32_t c_id = c_tensor->get_id();
  _model->add_tensor(std::move(c_tensor));
  add_output(c_id);
}

void LSEstimatorOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  // M = 32, tile_m = 4 -> 8 tiles
  const uint32_t tiles = 8;
  const uint32_t num_cores = _config.num_cores;
  for (uint32_t tile_idx = 0; tile_idx < tiles; ++tile_idx) {
    int core_id = static_cast<int>(tile_idx % (num_cores ? num_cores : 1));

    auto tile = std::make_unique<Tile>(Tile{
        .status = Tile::Status::INITIALIZED,
        .optype = "LSEstimatorOp",
        .layer_id = _id,
        .fused_op_id = 0,
        .batch = 0,   // N axis not tiled
        .Q = 0,
        .P = 0,
        .M = tile_idx,  // block index along M
        .C = 0,
        .S = 1,
        .R = 1,
        .stat = {},
        .instructions = {},
        .accum = false,
        .skip = false,
        .spad_id = 0,
        .accum_spad_id = 0,
        .core_id = core_id,
        .inst_finished = false});

    initialize_instructions(tile.get(), Mapping{});
    if (!tile->instructions.empty()) {
      _tiles.push_back(std::move(tile));
    }
  }
}

void LSEstimatorOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  // Tiling parameters
  const uint32_t rows_per_tile = 4;   // each tile computes 4 rows along M
  const uint32_t K = 32;
  const uint32_t N = 512;

  const uint32_t m_block = tile->M;           // 0..7
  const uint32_t row_start = m_block * rows_per_tile;

  // DRAM base addresses
  addr_type a_base = get_operand_addr(_INPUT_OPERAND + 0);  // A
  addr_type b_base = get_operand_addr(_INPUT_OPERAND + 1);  // B
  addr_type c_base = get_operand_addr(_OUTPUT_OPERAND + 0); // C

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  // SPAD layout (per core)
  addr_type a_spad_base = SPAD_BASE;
  addr_type b_spad_base = SPAD_BASE + static_cast<addr_type>(32 * 32) * _config.precision;
  addr_type c_spad_base = ACCUM_SPAD_BASE;

  addr_type a_spad_addr = a_spad_base +
      static_cast<addr_type>(row_start * K) * _config.precision;
  addr_type b_spad_addr = b_spad_base;  // full B reused per core
  addr_type c_spad_addr = c_spad_base +
      static_cast<addr_type>(row_start * N) * _config.precision;

  // 1) MOVIN A block (4x32)
  {
    std::set<addr_type> a_addrs;
    for (uint32_t r = 0; r < rows_per_tile; ++r) {
      uint32_t global_r = row_start + r;
      if (global_r >= _a_shape[0]) break;
      for (uint32_t c = 0; c < K; c += elems_per_access) {
        uint32_t col = std::min(c, K - 1);
        std::vector<uint32_t> index = {global_r, col};
        addr_type off = make_address(index, _a_shape);
        a_addrs.insert(a_base + off);
      }
    }
    if (!a_addrs.empty()) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .start_cycle = 0,
          .finish_cycle = 0,
          .id = "",
          .dependent_ids = {},
          .dest_id = "",
          .dest_addr = a_spad_addr,
          .size = static_cast<uint32_t>(a_addrs.size()),
          .compute_size = 0,
          .src_addrs = std::vector<addr_type>(a_addrs.begin(), a_addrs.end()),
          .spad_id = 0,
          .accum_spad_id = 0,
          .operand_id = _INPUT_OPERAND,
          .base_addr = a_base,
          .tile_m = rows_per_tile,
          .tile_k = K,
          .tile_n = 0,
          .src_from_accum = false,
          .zero_init = false,
          .last_inst = false,
          .my_tile = tile}));
    }
  }

  // 2) MOVIN full B (32x512) per tile (worst-case/broadcast)
  {
    std::set<addr_type> b_addrs;
    for (uint32_t r = 0; r < K; ++r) {
      for (uint32_t c = 0; c < N; c += elems_per_access) {
        uint32_t col = std::min(c, N - 1);
        std::vector<uint32_t> index = {r, col};
        addr_type off = make_address(index, _b_shape);
        b_addrs.insert(b_base + off);
      }
    }
    if (!b_addrs.empty()) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .start_cycle = 0,
          .finish_cycle = 0,
          .id = "",
          .dependent_ids = {},
          .dest_id = "",
          .dest_addr = b_spad_addr,
          .size = static_cast<uint32_t>(b_addrs.size()),
          .compute_size = 0,
          .src_addrs = std::vector<addr_type>(b_addrs.begin(), b_addrs.end()),
          .spad_id = 0,
          .accum_spad_id = 0,
          .operand_id = _INPUT_OPERAND + 1,
          .base_addr = b_base,
          .tile_m = K,
          .tile_k = N,
          .tile_n = 0,
          .src_from_accum = false,
          .zero_init = false,
          .last_inst = false,
          .my_tile = tile}));
    }
  }

  // 3) GEMM_PRELOAD for this tile: C_tile = A_tile * B
  {
    auto inst = std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .start_cycle = 0,
        .finish_cycle = 0,
        .id = "",
        .dependent_ids = {},
        .dest_id = "",
        .dest_addr = c_spad_addr,
        .size = N,               // logical size along N
        .compute_size = N,       // used by SystolicWS latency model
        .src_addrs = std::vector<addr_type>{a_spad_addr, b_spad_addr},
        .spad_id = 0,
        .accum_spad_id = 0,
        .operand_id = _NO_OPERAND,
        .base_addr = 0,
        .tile_m = rows_per_tile,
        .tile_k = K,
        .tile_n = N,
        .src_from_accum = false,
        .zero_init = false,
        .last_inst = false,
        .my_tile = tile});

    tile->instructions.push_back(std::move(inst));
  }

  // 4) MOVOUT C block (4x512)
  {
    std::set<addr_type> c_addrs;
    for (uint32_t r = 0; r < rows_per_tile; ++r) {
      uint32_t global_r = row_start + r;
      if (global_r >= _c_shape[0]) break;
      for (uint32_t c = 0; c < N; c += elems_per_access) {
        uint32_t col = std::min(c, N - 1);
        std::vector<uint32_t> index = {global_r, col};
        addr_type off = make_address(index, _c_shape);
        c_addrs.insert(c_base + off);
      }
    }
    if (!c_addrs.empty()) {
      // Mark MOVOUT as last instruction of this tile
      auto inst = std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVOUT,
          .start_cycle = 0,
          .finish_cycle = 0,
          .id = "",
          .dependent_ids = {},
          .dest_id = "",
          .dest_addr = c_spad_addr,
          .size = static_cast<uint32_t>(c_addrs.size()),
          .compute_size = 0,
          .src_addrs = std::vector<addr_type>(c_addrs.begin(), c_addrs.end()),
          .spad_id = 0,
          .accum_spad_id = 0,
          .operand_id = _OUTPUT_OPERAND,
          .base_addr = c_base,
          .tile_m = rows_per_tile,
          .tile_k = 0,
          .tile_n = N,
          .src_from_accum = true,
          .zero_init = false,
          .last_inst = true,
          .my_tile = tile});

      tile->instructions.push_back(std::move(inst));
    }
  }
}
