#!/bin/bash
# run_auto.sh
# Auto build + simulate Newton–Schulz test on 910B

set -e

echo "=== Auto Building... ==="
cd /project/ONNXim/build
make -j"$(nproc)"

# Go back to repo root so config-relative paths work
cd /project/ONNXim

echo "=== Auto Running Simulator... ==="
export ONNXIM_TRACE_CSV="results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32_rerun.csv"
export ONNXIM_MAX_CORE_CYCLES=200000

./build/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --model example/newton_schulz_test.json \
  --mode newton_schulz_test

echo "=== Done ==="\n