#include "DeepUnfoldNPUOp.h"

#include "../Model.h"

#include <algorithm>

DeepUnfoldNPUOp::DeepUnfoldNPUOp(SimulationConfig config,
                                 Model* model,
                                 const std::string& name,
                                 std::map<std::string, std::string>& attributes,
                                 uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "DeepUnfoldNPUOp";
  parse_attributes();
  infer_shapes_from_model();
}

DeepUnfoldNPUOp::DeepUnfoldNPUOp(SimulationConfig config,
                                 MappingTable& mapping_table,
                                 const std::vector<uint32_t>& matrix_shape,
                                 uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "DeepUnfoldNPUOp";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void DeepUnfoldNPUOp::parse_attributes() {
  auto it_layers = _attributes.find("layers");
  if (it_layers != _attributes.end()) {
    try {
      _layers = std::max(1u, static_cast<uint32_t>(std::stoul(it_layers->second)));
    } catch (...) {
      _layers = 12;
    }
  }

  auto it_vec = _attributes.find("vector_repeats");
  if (it_vec != _attributes.end()) {
    try {
      _vector_repeats = std::max(1u, static_cast<uint32_t>(std::stoul(it_vec->second)));
    } catch (...) {
      _vector_repeats = 1;
    }
  }

  auto it_batch = _attributes.find("batch_size");
  if (it_batch != _attributes.end()) {
    try {
      _batch_size = static_cast<uint32_t>(std::stoul(it_batch->second));
    } catch (...) {
      _batch_size = 96;
    }
  }
}

void DeepUnfoldNPUOp::infer_shapes_from_model() {
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

void DeepUnfoldNPUOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  if (_config.num_cores == 0) {
    spdlog::error("DeepUnfoldNPUOp: Invalid core count 0!");
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

  spdlog::info("DeepUnfoldNPUOp '{}': Dispatched {} batches across {} cores.",
               _name, _batch_size, _config.num_cores);
  std::string load_msg = "  > Load Distribution:";
  for (uint32_t core = 0; core < _config.num_cores; ++core) {
    load_msg += " Core" + std::to_string(core) + ": " + std::to_string(core_load[core]);
  }
  spdlog::info("{}", load_msg);
}

void DeepUnfoldNPUOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("DeepUnfoldNPUOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t M = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t U = _matrix_shape[_matrix_shape.size() - 1];

  addr_type size_mu = static_cast<addr_type>(M) * U * _config.precision;  // [M,U]
  addr_type size_uu = static_cast<addr_type>(U) * U * _config.precision;  // [U,U]

  addr_type batch_offset_mu = static_cast<addr_type>(tile->batch) * size_mu;
  addr_type batch_offset_uu = static_cast<addr_type>(tile->batch) * size_uu;

  // Inputs:
  //  0: H      [B,M,U]
  //  1: RegI   [U,U]
  //  2: Y      [B,M,U]
  //  3: X0     [B,U,U]
  addr_type h_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset_mu;
  addr_type reg_base = get_operand_addr(_INPUT_OPERAND + 1);
  addr_type y_base = get_operand_addr(_INPUT_OPERAND + 2) + batch_offset_mu;
  addr_type x0_base = get_operand_addr(_INPUT_OPERAND + 3) + batch_offset_uu;

  // Output:
  //  0: X_hat  [B,U,U]
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset_uu;

  // SPAD layout.
  addr_type addr_H = SPAD_BASE;                     // [M,U]
  addr_type addr_Y = addr_H + size_mu;              // [M,U]
  addr_type addr_Reg = addr_Y + size_mu;            // [U,U]
  addr_type addr_A = addr_Reg + size_uu;            // [U,U] regularized Gram
  addr_type addr_Xk = addr_A + size_uu;             // [U,U]
  addr_type addr_res = addr_Xk + size_uu;           // [U,U] residual / vector temp
  addr_type addr_W = addr_res + size_uu;            // [U,M]
  addr_type addr_acc = ACCUM_SPAD_BASE;             // [U,U] cube accumulation

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
  emit_movin(y_base, addr_Y, M, U, shape_mu, _INPUT_OPERAND + 2);
  emit_movin(x0_base, addr_Xk, U, U, shape_uu, _INPUT_OPERAND + 3);

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_LOAD2GRAM",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  // A = H^H H + RegI
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "DU_GRAM",
      .dest_addr = addr_A,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_H, addr_H},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_GRAM2REG",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 2}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::ADD,
      .id = "DU_REG",
      .dest_addr = addr_A,
      .compute_size = U * U,
      .src_addrs = std::vector<addr_type>{addr_A, addr_Reg},
      .tile_m = U,
      .tile_k = U,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_REG2LAYER",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 3}));

  for (uint32_t layer = 0; layer < _layers; ++layer) {
    // Cube phase 1: T = A * X_k
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "DU_AX_" + std::to_string(layer),
        .dest_addr = addr_acc,
        .compute_size = U,
        .src_addrs = std::vector<addr_type>{addr_A, addr_Xk},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "DU_BARRIER_CUBE2VEC_" + std::to_string(layer),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 4}));

    // Vector phase: residual/update approximation.
    for (uint32_t rep = 0; rep < _vector_repeats; ++rep) {
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::ADD,
          .id = "DU_RES_" + std::to_string(layer) + "_" + std::to_string(rep),
          .dest_addr = addr_res,
          .compute_size = U * U,
          .src_addrs = std::vector<addr_type>{addr_Reg, addr_acc},
          .tile_m = U,
          .tile_k = U,
          .tile_n = U,
          .src_from_accum = true,
          .my_tile = tile}));
    }

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "DU_BARRIER_VEC2CUBE_" + std::to_string(layer),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));

    // Cube phase 2: X_{k+1} = X_k * residual
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "DU_XNEXT_" + std::to_string(layer),
        .dest_addr = addr_acc,
        .compute_size = U,
        .src_addrs = std::vector<addr_type>{addr_Xk, addr_res},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = false,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "DU_BARRIER_LAYER_SYNC_" + std::to_string(layer),
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 5}));

    // Write back current iterate to SPAD working buffer.
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "DU_STORE_XK_" + std::to_string(layer),
        .dest_addr = addr_Xk,
        .compute_size = U * U,
        .src_addrs = std::vector<addr_type>{addr_acc, addr_Reg},
        .tile_m = U,
        .tile_k = U,
        .tile_n = U,
        .src_from_accum = true,
        .my_tile = tile}));
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_INV2W",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 6}));

  // W = X_k * H^H  (modeled as [U,U] x [U,M] -> [U,M])
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "DU_W",
      .dest_addr = addr_W,
      .compute_size = M,
      .src_addrs = std::vector<addr_type>{addr_Xk, addr_H},
      .tile_m = U,
      .tile_k = U,
      .tile_n = M,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_W2XHAT",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 7}));

  // X_hat = W * Y
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::GEMM_PRELOAD,
      .id = "DU_XHAT",
      .dest_addr = addr_acc,
      .compute_size = U,
      .src_addrs = std::vector<addr_type>{addr_W, addr_Y},
      .tile_m = U,
      .tile_k = M,
      .tile_n = U,
      .src_from_accum = false,
      .my_tile = tile}));

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "DU_BARRIER_XHAT2STORE",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 8}));

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
        .id = "DU_OUT",
        .dest_addr = addr_acc,
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
        .barrier_type = 8}));
  }

  if (tile->instructions.empty()) {
    spdlog::error("DeepUnfoldNPUOp: No instructions generated for Batch {} Core {}",
                  tile->batch, tile->core_id);
  }
}
