#include "SystolicWS.h"

#include "TraceLogger.h"

SystolicWS::SystolicWS(uint32_t id, SimulationConfig config)
    : Core(id, config) {}

void SystolicWS::cycle() {
    /*
  Compute unit
  */
  finish_compute_pipeline();
  /* Checking Vector compute pipeline */
  finish_vector_pipeline();
  /* Checking Scalar compute pipeline */
  finish_scalar_pipeline();
  /* LD in struction queue */
  handle_ld_inst_queue();
  /* EX instruction queue */
  if (!_ex_inst_queue.empty() && can_issue_compute(_ex_inst_queue.front())) { // execution dependecy check
    std::unique_ptr<Instruction> front = std::move(_ex_inst_queue.front());
    if (front->dest_addr >= ACCUM_SPAD_BASE) {
      if (_acc_spad.check_allocated(front->dest_addr, front->accum_spad_id)) {
        _acc_spad.count_up(front->dest_addr, front->accum_spad_id);
      } else {
        int ret = _acc_spad.prefetch(front->dest_addr, front->accum_spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _acc_spad.check_allocated(front->dest_addr, front->accum_spad_id), _acc_spad.check_remain(front->size, front->accum_spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _acc_spad.print_all(front->accum_spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      if (_spad.check_allocated(front->dest_addr, front->spad_id)) {
        _spad.count_up(front->dest_addr, front->spad_id);
      } else {
        int ret = _spad.prefetch(front->dest_addr, front->spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _spad.check_allocated(front->dest_addr, front->spad_id), _spad.check_remain(front->size, front->spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
          _spad.print_all(front->spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    if (front->opcode == Opcode::GEMM || front->opcode == Opcode::GEMM_PRELOAD) {
      if (!_compute_pipeline.empty()) {
        /* Preload can be hided */
        uint32_t offset = _compute_pipeline.back()->compute_size;
        offset = MAX(offset, 4);
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          // State mul-pre
          offset = MAX(offset, _config.core_config[_id].core_height);
          _stat_systolic_preload_issue_count++;
        }
        if (_compute_pipeline.back()->start_cycle+offset < _core_cycle) {
          front->start_cycle = _core_cycle;
          _stat_systolic_bubble_cycle += (_core_cycle - _compute_pipeline.back()->start_cycle+offset);
        } else
          front->start_cycle = _compute_pipeline.back()->start_cycle+offset;
      } else {
        front->start_cycle = _core_cycle;
        /* Preload weight to systolic array*/
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          /* Weight preload  from buffer latecny + WEight preload latency */
          front->start_cycle += _config.core_config[_id].core_height + _config.core_config[_id].core_height - 1;
          _stat_systolic_preload_issue_count++;
        }
      }

      if (front->start_cycle > _core_cycle) {
        TraceLogger::log_event(
            fmt::format("Core{}_Wait", _id),
            "CubeWait",
            _core_cycle,
            front->start_cycle);
      }

      front->finish_cycle = front->start_cycle + get_inst_compute_cycles(front);
      _compute_pipeline.push(std::move(front));
      _stat_systolic_inst_issue_count++;
    } else {
      const bool is_scalar_op =
          (front->opcode == Opcode::SCALAR_ADD || front->opcode == Opcode::SCALAR_MUL ||
           front->opcode == Opcode::SCALAR_DIV || front->opcode == Opcode::SCALAR_SQRT);
      front->start_cycle = _core_cycle;
      front->finish_cycle = front->start_cycle +
                            (is_scalar_op ? get_scalar_compute_cycles(front)
                                          : get_vector_compute_cycles(front));
      if (is_scalar_op)
        _scalar_pipeline.push(std::move(front));
      else
        _vector_pipeline.push(std::move(front));
    }
    _ex_inst_queue.pop();
  }

  /* ST in struction queue */
  handle_st_inst_queue();

  // xxx will it work well on double buffered code? no.
  bool is_idle = _compute_pipeline.empty() && _vector_pipeline.empty();
  bool is_running = running();
  bool is_compute_busy = false;
  bool is_vector_busy = false;
  bool is_scalar_busy = false;

  if (!_compute_pipeline.empty() && _compute_pipeline.front()->start_cycle <= _core_cycle)
    is_compute_busy = true;
  if (!_vector_pipeline.empty() && _vector_pipeline.front()->start_cycle <= _core_cycle)
    is_vector_busy = true;
  if (!_scalar_pipeline.empty() && _scalar_pipeline.front()->start_cycle <= _core_cycle)
    is_scalar_busy = true;

  if (is_compute_busy)
    _stat_systolic_active_cycle++;
  if (is_vector_busy)
    _stat_vec_compute_cycle++;
  if (is_scalar_busy)
    _stat_scalar_compute_cycle++;

  if (is_compute_busy || is_vector_busy || is_scalar_busy)
    _stat_compute_cycle++;

  if (_request_queue.empty())
    _stat_memory_idle_cycle++;

  if (!is_running)
    _stat_idle_cycle++;
  Core::cycle();
}

bool SystolicWS::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  if(Core::can_issue_compute(inst) == false)
    return false;
  if (inst->opcode == Opcode::GEMM || inst->opcode == Opcode::GEMM_PRELOAD) {
    if (_compute_pipeline.size() >= _config.core_config[_id].core_height) {
      return false;
    }
  } else if (inst->opcode == Opcode::SCALAR_ADD || inst->opcode == Opcode::SCALAR_MUL ||
             inst->opcode == Opcode::SCALAR_DIV || inst->opcode == Opcode::SCALAR_SQRT) {
    if(!_scalar_pipeline.empty()) {
      return false;
    }
  } else {
    if(!_vector_pipeline.empty()) {
      return false;
    }
  }
  return true;
}

cycle_type SystolicWS::get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) {
  if (_config.core_config[_id].enable_ascend_cube_model) {
    const uint32_t cube_m = std::max(1u, _config.core_config[_id].cube_m);
    const uint32_t cube_n = std::max(1u, _config.core_config[_id].cube_n);
    const uint32_t cube_k = std::max(1u, _config.core_config[_id].cube_k);
    const uint32_t base_latency = _config.core_config[_id].cube_base_latency;

    const uint32_t tile_m = std::max(1u, inst->tile_m);
    const uint32_t tile_n = std::max(1u, inst->tile_n);
    const uint32_t tile_k = std::max(1u, inst->tile_k);

    const cycle_type blocks_m = ceil_div(tile_m, cube_m);
    const cycle_type blocks_n = ceil_div(tile_n, cube_n);
    const cycle_type blocks_k = ceil_div(tile_k, cube_k);
    const cycle_type cube_steps = blocks_m * blocks_n * blocks_k;
    const cycle_type pipeline_fill_drain = cube_m + cube_n - 2;

    return base_latency + pipeline_fill_drain + std::max<cycle_type>(cube_steps, 1);
  }

  return _config.core_config[_id].core_height + _config.core_config[_id].core_width - 2 + MAX(inst->compute_size, 4);
}

cycle_type SystolicWS::get_vector_compute_cycles(std::unique_ptr<Instruction>& inst) {
  cycle_type vec_op_iter = calculate_vector_op_iterations(inst->compute_size);
  cycle_type add_tree_iter = calculate_add_tree_iterations(inst->compute_size);
  cycle_type add_tree, scalar_ops, vector_ops;
  switch (inst->opcode) {
    case Opcode::LAYERNORM:
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency;
      scalar_ops = 2 * _config.core_config[_id].scalar_mul_latency + _config.core_config[_id].scalar_sqrt_latency;
      // 1 addition, 1 subtraction, 1 division, 2 multiplication.
      vector_ops = vec_op_iter * (2 * _config.core_config[_id].add_latency + 3 * _config.core_config[_id].mul_latency) * inst->tile_m;
      return add_tree + scalar_ops + vector_ops;
    case Opcode::SOFTMAX:
      // 1 add tree, 1 compare tree
      add_tree = 2 * add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
      vector_ops =
        vec_op_iter * (_config.core_config[_id].add_latency + _config.core_config[_id].exp_latency + _config.core_config[_id].mul_latency);
      return add_tree + vector_ops;
    case Opcode::ADD:
      return vec_op_iter * _config.core_config[_id].add_latency;
    case Opcode::MUL:
      return vec_op_iter * _config.core_config[_id].mul_latency;
    case Opcode::MAC:
      return vec_op_iter * _config.core_config[_id].mac_latency;
    case Opcode::SWISH: //TODO: Implement SWISH
    case Opcode::GELU:
      return vec_op_iter * _config.core_config[_id].gelu_latency;
    case Opcode::COMP:
      return vec_op_iter * 1;
    case Opcode::ADDTREE:
      return add_tree_iter * _config.core_config[_id].add_tree_latency * inst->tile_m;
    case Opcode::DIV:
      return vec_op_iter * _config.core_config[_id].div_latency;
    case Opcode::EXP:
      return vec_op_iter * _config.core_config[_id].exp_latency;
    case Opcode::SQRT:
      return vec_op_iter * _config.core_config[_id].scalar_sqrt_latency;
    case Opcode::PIPE_BARRIER:
      // Synthetic barrier/NOP: occupy the vector pipeline for 1 cycle
      // to make barriers visible in traces without adding real work.
      return 1;
    
  }
  spdlog::info("not configured operation. {}", inst->id);
  // assert(0);
  return 0;
}

cycle_type SystolicWS::get_scalar_compute_cycles(std::unique_ptr<Instruction>& inst) {
  switch (inst->opcode) {
    case Opcode::SCALAR_ADD:
      return _config.core_config[_id].scalar_add_latency;
    case Opcode::SCALAR_MUL:
      return _config.core_config[_id].scalar_mul_latency;
    case Opcode::SCALAR_DIV:
      return _config.core_config[_id].div_latency;
    case Opcode::SCALAR_SQRT:
      return _config.core_config[_id].scalar_sqrt_latency;
    default:
      break;
  }
  return 1;
}

void SystolicWS::print_stats() {
  Core::print_stats();
  spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
               _stat_systolic_inst_issue_count);
  spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
               _stat_systolic_preload_issue_count);
}