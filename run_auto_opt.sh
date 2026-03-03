#!/usr/bin/env bash
set -e

cd /project/ONNXim/build && make -j"$(nproc)"
cd /project/ONNXim

export ONNXIM_TRACE_CSV="results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32_opt.csv"
export ONNXIM_MAX_CORE_CYCLES=200000

./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/newton_schulz_opt_test.json \
  --mode newton_schulz_opt_test

echo "=== Opt Done ==="
