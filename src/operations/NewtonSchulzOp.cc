#include "NewtonSchulzOp.h"

#include "../Model.h"
#include <cstdlib>
#include <numeric>

// 辅助宏：确保 batch_size 定义
#ifndef DEFAULT_BATCH_SIZE
#define DEFAULT_BATCH_SIZE 96
#endif

NewtonSchulzOp::NewtonSchulzOp(SimulationConfig config,
                               Model* model,
                               onnx::NodeProto& node_proto,
                               uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  _optype = "NewtonSchulz";
  parse_attributes();
  infer_shapes_from_model();
}

NewtonSchulzOp::NewtonSchulzOp(SimulationConfig config,
                               Model* model,
                               const std::string& name,
                               std::map<std::string, std::string>& attributes,
                               uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _optype = "NewtonSchulz";
  parse_attributes();
  infer_shapes_from_model();
}

NewtonSchulzOp::NewtonSchulzOp(SimulationConfig config,
                               MappingTable& mapping_table,
                               const std::vector<uint32_t>& matrix_shape,
                               uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  (void)mapping_table;
  _optype = "NewtonSchulz";
  _matrix_shape = matrix_shape;
  parse_attributes();
}

void NewtonSchulzOp::parse_attributes() {
  // 1. 解析迭代次数
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

  // 2. 解析 Batch Size (支持动态配置)
  // 优先级：属性设置 > 默认值 96
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

void NewtonSchulzOp::infer_shapes_from_model() {
  if (!_matrix_shape.empty() || !_model) return;

  if (_inputs.size() >= 1) {
    Tensor* a_tensor = _model->get_tensor(_inputs[0]);
    if (a_tensor) {
      std::vector<uint32_t> dims = a_tensor->get_dims();
      // 如果输入是 3 维 [Batch, N, K]，则覆盖 _batch_size
      if (dims.size() == 3) {
          _batch_size = dims[0];
          _matrix_shape = {dims[1], dims[2]}; 
      } else {
          _matrix_shape = dims;
      }
    }
  }
}

void NewtonSchulzOp::initialize_tiles(MappingTable& /*mapping_table*/) {
  // 核心逻辑修改：不再只生成一个 Tile，而是生成 _batch_size 个 Tile
  // 并使用 Round-Robin 策略分配给所有物理核心。
  
  if (_config.num_cores == 0) {
      spdlog::error("NewtonSchulzOp: Invalid core count 0!");
      return;
  }

  std::vector<int> core_load(_config.num_cores, 0); // 用于统计负载

  for (uint32_t b = 0; b < _batch_size; ++b) {
      // Round-Robin 分配：b=0 -> Core0, b=1 -> Core1 ... b=24 -> Core0
      uint32_t assigned_core = b % _config.num_cores;
      
      auto tile = std::make_unique<Tile>(Tile{
          .status = Tile::Status::INITIALIZED,
          .optype = _optype,
          .layer_id = _id,
          .fused_op_id = 0,
          .batch = b,      // 标记这个 Tile 属于哪个 Batch
          .Q = 1, .P = 1, .M = 0, .C = 0, .S = 1, .R = 1,
          .stat = {},
          .instructions = {},
          .accum = false,
          .skip = false,
          .spad_id = 0,
          .accum_spad_id = 0,
          .core_id = static_cast<int>(assigned_core), // 动态分配
          .inst_finished = false
      });

      initialize_instructions(tile.get(), Mapping{});
      
      if (!tile->instructions.empty()) {
          _tiles.push_back(std::move(tile));
          core_load[assigned_core]++;
      }
  }
  
  // 打印负载均衡情况，方便用户验证
  spdlog::info("NewtonSchulzOp '{}': Dispatched {} batches across {} cores.", 
               _name, _batch_size, _config.num_cores);
  spdlog::info("  > Load Distribution (First 4 cores): Core0: {}, Core1: {}, Core2: {}, Core3: {} ...", 
               core_load[0], core_load[1], core_load[2], core_load[3]);
}

void NewtonSchulzOp::initialize_instructions(Tile* tile, Mapping /*mapping*/) {
  if (_matrix_shape.size() < 2) {
    spdlog::error("NewtonSchulzOp: matrix shape not set for layer {}", _name);
    return;
  }

  const uint32_t N = _matrix_shape[_matrix_shape.size() - 2];
  const uint32_t K = _matrix_shape[_matrix_shape.size() - 1];

  // =========================================================
  // 关键修改：计算 Batch 偏移地址
  // 假设 DRAM 中矩阵是连续存放的：[Batch 0][Batch 1][Batch 2]...
  // 每个矩阵大小 = N * K * precision
  // =========================================================
  addr_type matrix_size_bytes = static_cast<addr_type>(N) * K * _config.precision;
  addr_type batch_offset = static_cast<addr_type>(tile->batch) * matrix_size_bytes;

  // DRAM base addresses (基础地址 + 偏移量)
  addr_type a_base = get_operand_addr(_INPUT_OPERAND + 0) + batch_offset;
  addr_type x_base = get_operand_addr(_INPUT_OPERAND + 1) + batch_offset;
  // C (2I) 通常是广播的常数矩阵，不需要 batch_offset，所有核共用一份
  addr_type c_base = get_operand_addr(_INPUT_OPERAND + 2); 
  // 输出地址也要偏移
  addr_type out_base = get_operand_addr(_OUTPUT_OPERAND + 0) + batch_offset;

  // SRAM 地址保持不变，因为每个 Tile 独占自己的 Core 的 SRAM
  addr_type addr_A = SPAD_BASE;
  addr_type addr_X = addr_A + matrix_size_bytes;
  addr_type addr_C = addr_X + matrix_size_bytes;
  addr_type addr_R = addr_C + matrix_size_bytes;
  addr_type addr_T = ACCUM_SPAD_BASE;

  int elems_per_access = _config.dram_req_size / _config.precision;
  if (elems_per_access <= 0) elems_per_access = 1;

  // Helper lambda (保持不变)
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
          .tile_m = N, .tile_k = K,
          .my_tile = tile}));
    }
  };

  // ========================
  // Load Phase
  // ========================
  emit_movin_full(a_base, addr_A, _INPUT_OPERAND + 0);
  emit_movin_full(x_base, addr_X, _INPUT_OPERAND + 1);
  emit_movin_full(c_base, addr_C, _INPUT_OPERAND + 2); // 广播加载 2I

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::PIPE_BARRIER,
      .id = "NS_BARRIER_MTE2CUBE",
      .my_tile = tile,
      .is_barrier = true,
      .barrier_type = 1}));

  // ========================
  // Compute Phase
  // ========================
  for (uint32_t iter = 0; iter < _iterations; ++iter) {
    // 在第 0 次迭代中，X 来自最初加载到 SPAD 的 addr_X；
    // 之后的迭代中，当前 X_k 已经被写入 ACCUM 的 addr_T，
    // 因此后续都应从 addr_T 读取以完成真正的多步迭代。
    addr_type x_src_for_AX = (iter == 0) ? addr_X : addr_T;
    addr_type x_src_for_XR = (iter == 0) ? addr_X : addr_T;
    bool use_accum_for_x = (iter > 0);

    // T = A * X
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "NS_T",
        .dest_addr = addr_T,
        .compute_size = K,
      .src_addrs = std::vector<addr_type>{addr_A, x_src_for_AX},
      .tile_m = N, .tile_k = K, .tile_n = K,
      .src_from_accum = use_accum_for_x,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "NS_BARRIER_CUBE2VEC",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 2}));

    // R = C - T
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::ADD,
        .id = "NS_R",
        .dest_addr = addr_R,
        .compute_size = N * K,
        .src_addrs = std::vector<addr_type>{addr_C, addr_T},
        .tile_m = N, .tile_k = K, .tile_n = K,
        .src_from_accum = true,
        .my_tile = tile}));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::PIPE_BARRIER,
        .id = "NS_BARRIER_VEC2CUBE",
        .my_tile = tile,
        .is_barrier = true,
        .barrier_type = 3}));

    // X_new = X * R
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::GEMM_PRELOAD,
        .id = "NS_X",
        .dest_addr = addr_T,
        .compute_size = K,
    .src_addrs = std::vector<addr_type>{x_src_for_XR, addr_R},
    .tile_m = N, .tile_k = K, .tile_n = K,
    .src_from_accum = use_accum_for_x,
        .my_tile = tile}));
  }

  // ========================
  // Store Phase
  // ========================
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
        .id = "NS_OUT",
        .dest_addr = addr_T,
        .size = static_cast<uint32_t>(out_addrs.size()),
        .src_addrs = std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
        .base_addr = out_base,
        .tile_m = N, .tile_k = K, .tile_n = K,
        .src_from_accum = true,
        .last_inst = true,
        .my_tile = tile,
        .barrier_type = 4}));
  }

  if (tile->instructions.empty()) {
    spdlog::error("NewtonSchulzOp: No instructions generated for Batch {} Core {}", tile->batch, tile->core_id);
  }
}