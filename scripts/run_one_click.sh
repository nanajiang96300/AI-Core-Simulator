#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
SIM_BIN="$BUILD_DIR/bin/Simulator"

ACTION="all"            # all | sim | plot
OPERATOR="ldl"          # newton | newton_opt | matmul | mmse | ldl | ldl_noblock | deepunfold | deepunfold_opt | series_inverse | cholesky | cholesky_noblock | cholesky_noblock_iso | cholesky_chain | deepunfold_bj
CONFIG="configs/ascend_910b_quiet.json"
MODE_OVERRIDE=""
MODELS_LIST_OVERRIDE=""
TRACE_CSV=""
PLOT_PNG=""
MAX_CYCLES="200000"
LOG_LEVEL="info"
JOBS="$(nproc)"
BUILD_FIRST=1
DRY_RUN=0
EXPORT_CYCLE_TABLE=1
CYCLE_TABLE_OUT=""
CYCLE_MATRIX_M=64
CYCLE_MATRIX_U=8
CYCLE_REDUCER="median"

usage() {
  cat <<'EOF'
One-click runner for ONNXim simulation and plotting.

Usage:
  scripts/run_one_click.sh [options]

Options:
  --action <all|sim|plot>         Run simulation+plot, simulation only, or plot only (default: all)
  --operator <name>               Operator profile:
                                  newton | newton_opt | matmul | mmse | ldl | ldl_noblock | deepunfold | deepunfold_opt | series_inverse | cholesky | cholesky_noblock | cholesky_noblock_iso | cholesky_chain | deepunfold_bj | deepunfold_bj_npu_opt
  --config <path>                 Simulator config json (default: configs/ascend_910b_quiet.json)
  --models-list <path>            Override models list json for Simulator operators
  --mode <name>                   Override --mode for Simulator operators
  --trace <path>                  Trace CSV path (for Simulator operators)
  --png <path>                    Output png path for visualizer (or deepunfold output dir when operator=deepunfold_bj)
  --max-cycles <int>              ONNXIM_MAX_CORE_CYCLES (default: 200000)
  --log-level <level>             Simulator log level (default: info)
  --jobs <n>                      Build jobs (default: nproc)
  --export-cycle-table <0|1>      Auto export detailed cycle table for deepunfold/deepunfold_opt (default: 1)
  --cycle-table-out <path>        Custom output CSV for detailed cycle table
  --cycle-m <int>                 Formula dimension M used in detailed table (default: 64)
  --cycle-u <int>                 Formula dimension U used in detailed table (default: 8)
  --cycle-reducer <name>          Reducer for repeated events: median|max|mean|sum (default: median)
  --no-build                      Skip build step
  --dry-run                       Print commands only, do not execute
  -h, --help                      Show this help

Examples:
  # 1) LDL: build + sim + plot
  scripts/run_one_click.sh --operator ldl --action all

  # 2) MatMul: custom output paths
  scripts/run_one_click.sh --operator matmul --trace results/matmul_auto.csv --png results/matmul_auto.png

  # 3) DeepUnfold BJ compare (python flow, no Simulator)
  scripts/run_one_click.sh --operator deepunfold_bj --action all --png results/LDL/deepunfold_bj_compare_auto

  # 4) Approval-friendly preview
  scripts/run_one_click.sh --operator mmse --action all --dry-run
EOF
}

log() { echo "[one-click] $*"; }

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

resolve_profile() {
  case "$OPERATOR" in
    newton)
      : "${MODE_OVERRIDE:=newton_schulz_test}"
      : "${MODELS_LIST_OVERRIDE:=example/newton_schulz_test.json}"
      : "${TRACE_CSV:=results/newton_schulz/newton_32x32_auto.csv}"
      : "${PLOT_PNG:=results/newton_schulz/newton_32x32_auto.png}"
      ;;
    newton_opt)
      : "${MODE_OVERRIDE:=newton_schulz_opt_test}"
      : "${MODELS_LIST_OVERRIDE:=example/newton_schulz_opt_test.json}"
      : "${TRACE_CSV:=results/newton_schulz/newton_opt_32x32_auto.csv}"
      : "${PLOT_PNG:=results/newton_schulz/newton_opt_32x32_auto.png}"
      ;;
    matmul)
      : "${MODE_OVERRIDE:=matmul_test}"
      : "${MODELS_LIST_OVERRIDE:=example/matmul_256x32x256_test.json}"
      : "${TRACE_CSV:=results/matmul_256x32x256_auto.csv}"
      : "${PLOT_PNG:=results/matmul_256x32x256_auto.png}"
      ;;
    mmse)
      : "${MODE_OVERRIDE:=mmse_test}"
      : "${MODELS_LIST_OVERRIDE:=example/mmse_test.json}"
      : "${TRACE_CSV:=results/mmse/mmse_auto.csv}"
      : "${PLOT_PNG:=results/mmse/mmse_auto.png}"
      ;;
    ldl)
      : "${MODE_OVERRIDE:=ldl_test}"
      : "${MODELS_LIST_OVERRIDE:=example/ldl_test.json}"
      : "${TRACE_CSV:=results/LDL/ldl_auto.csv}"
      : "${PLOT_PNG:=results/LDL/ldl_auto.png}"
      ;;
    ldl_noblock)
      : "${MODE_OVERRIDE:=ldl_noblock_test}"
      : "${MODELS_LIST_OVERRIDE:=example/ldl_noblock_test.json}"
      : "${TRACE_CSV:=results/LDL/ldl_noblock_auto.csv}"
      : "${PLOT_PNG:=results/LDL/ldl_noblock_auto.png}"
      ;;
    deepunfold)
      : "${MODE_OVERRIDE:=deepunfold_test}"
      : "${MODELS_LIST_OVERRIDE:=example/deepunfold_test.json}"
      : "${TRACE_CSV:=results/DeepUnfold/deepunfold_npu_auto.csv}"
      : "${PLOT_PNG:=results/DeepUnfold/deepunfold_npu_auto.png}"
      ;;
    deepunfold_opt)
      : "${MODE_OVERRIDE:=deepunfold_opt_test}"
      : "${MODELS_LIST_OVERRIDE:=example/deepunfold_opt_test.json}"
      : "${TRACE_CSV:=results/DeepUnfold/deepunfold_npu_opt_auto.csv}"
      : "${PLOT_PNG:=results/DeepUnfold/deepunfold_npu_opt_auto.png}"
      ;;
    series_inverse)
      : "${MODE_OVERRIDE:=series_inverse_test}"
      : "${MODELS_LIST_OVERRIDE:=example/series_inverse_32x32.json}"
      : "${TRACE_CSV:=results/series_inverse_32x32_auto.csv}"
      : "${PLOT_PNG:=results/series_inverse_32x32_auto.png}"
      ;;
    cholesky)
      : "${MODE_OVERRIDE:=cholesky_test}"
      : "${MODELS_LIST_OVERRIDE:=example/cholesky_test.json}"
      : "${TRACE_CSV:=results/LDL/cholesky_auto.csv}"
      : "${PLOT_PNG:=results/LDL/cholesky_auto.png}"
      ;;
    cholesky_noblock)
      : "${MODE_OVERRIDE:=cholesky_noblock_test}"
      : "${MODELS_LIST_OVERRIDE:=example/cholesky_noblock_test.json}"
      : "${TRACE_CSV:=results/CHOL/cholesky_noblock_auto.csv}"
      : "${PLOT_PNG:=results/CHOL/cholesky_noblock_auto.png}"
      ;;
    cholesky_noblock_iso)
      : "${MODE_OVERRIDE:=cholesky_noblock_test}"
      : "${MODELS_LIST_OVERRIDE:=example/cholesky_noblock_iso_test.json}"
      : "${TRACE_CSV:=results/CHOL/cholesky_noblock_iso_auto.csv}"
      : "${PLOT_PNG:=results/CHOL/cholesky_noblock_iso_auto.png}"
      ;;
    cholesky_chain)
      : "${MODE_OVERRIDE:=cholesky_chain_test}"
      : "${MODELS_LIST_OVERRIDE:=example/cholesky_chain_test.json}"
      : "${TRACE_CSV:=results/CHOL/cholesky_chain_auto.csv}"
      : "${PLOT_PNG:=results/CHOL/cholesky_chain_auto.png}"
      ;;
    deepunfold_bj)
      : "${PLOT_PNG:=results/LDL/deepunfold_bj_compare_auto}"
      ;;
    deepunfold_bj_npu_opt)
      : "${PLOT_PNG:=results/DeepUnfold/bj_npu_opt_compare_auto}"
      ;;
    *)
      echo "Unsupported operator: $OPERATOR" >&2
      exit 1
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --action) ACTION="$2"; shift 2 ;;
      --operator) OPERATOR="$2"; shift 2 ;;
      --config) CONFIG="$2"; shift 2 ;;
      --models-list) MODELS_LIST_OVERRIDE="$2"; shift 2 ;;
      --mode) MODE_OVERRIDE="$2"; shift 2 ;;
      --trace) TRACE_CSV="$2"; shift 2 ;;
      --png) PLOT_PNG="$2"; shift 2 ;;
      --max-cycles) MAX_CYCLES="$2"; shift 2 ;;
      --log-level) LOG_LEVEL="$2"; shift 2 ;;
      --jobs) JOBS="$2"; shift 2 ;;
      --export-cycle-table) EXPORT_CYCLE_TABLE="$2"; shift 2 ;;
      --cycle-table-out) CYCLE_TABLE_OUT="$2"; shift 2 ;;
      --cycle-m) CYCLE_MATRIX_M="$2"; shift 2 ;;
      --cycle-u) CYCLE_MATRIX_U="$2"; shift 2 ;;
      --cycle-reducer) CYCLE_REDUCER="$2"; shift 2 ;;
      --no-build) BUILD_FIRST=0; shift 1 ;;
      --dry-run) DRY_RUN=1; shift 1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
  done
}

run_build_if_needed() {
  if [[ "$BUILD_FIRST" -eq 0 ]]; then
    log "Skip build (--no-build)."
    return
  fi
  log "Building project in $BUILD_DIR ..."
  run_cmd "cd '$BUILD_DIR' && make -j'$JOBS'"
}

run_simulator_flow() {
  local trace_dir
  trace_dir="$(dirname "$TRACE_CSV")"
  local lib_path
  lib_path="$ROOT_DIR/build/lib:$ROOT_DIR/build_new/lib:${LD_LIBRARY_PATH:-}"
  local sim_bin
  sim_bin="$SIM_BIN"
  if [[ -x "$ROOT_DIR/build_new/bin/Simulator" ]]; then
    sim_bin="$ROOT_DIR/build_new/bin/Simulator"
  fi

  if [[ "$ACTION" == "sim" || "$ACTION" == "all" ]]; then
    log "Running Simulator: op=$OPERATOR mode=$MODE_OVERRIDE"
    run_cmd "mkdir -p '$trace_dir'"
    run_cmd "cd '$ROOT_DIR' && LD_LIBRARY_PATH='$lib_path' ONNXIM_TRACE_CSV='$TRACE_CSV' ONNXIM_MAX_CORE_CYCLES='$MAX_CYCLES' '$sim_bin' --config '$CONFIG' --models_list '$MODELS_LIST_OVERRIDE' --mode '$MODE_OVERRIDE' --log_level '$LOG_LEVEL'"
  fi

  if [[ "$ACTION" == "plot" || "$ACTION" == "all" ]]; then
    local png_dir
    png_dir="$(dirname "$PLOT_PNG")"
    log "Plotting timeline png ..."
    run_cmd "mkdir -p '$png_dir'"
    run_cmd "cd '$ROOT_DIR' && python3 visualizer_png.py -i '$TRACE_CSV' -o '$PLOT_PNG'"
  fi

  if [[ "$EXPORT_CYCLE_TABLE" -eq 1 ]] && [[ "$ACTION" == "sim" || "$ACTION" == "all" ]]; then
    if [[ "$OPERATOR" == "deepunfold" || "$OPERATOR" == "deepunfold_opt" ]]; then
      local detail_csv
      detail_csv="$CYCLE_TABLE_OUT"
      if [[ -z "$detail_csv" ]]; then
        detail_csv="${TRACE_CSV%.csv}_detailed_cycles.csv"
      fi
      log "Exporting detailed cycle table ..."
      run_cmd "cd '$ROOT_DIR' && python3 scripts/export_deepunfold_cycle_table.py --trace '$TRACE_CSV' --output '$detail_csv' --mode auto --matrix-m '$CYCLE_MATRIX_M' --matrix-u '$CYCLE_MATRIX_U' --reducer '$CYCLE_REDUCER'"
    fi
  fi
}

run_deepunfold_bj_flow() {
  local out_dir="$PLOT_PNG"
  if [[ "$ACTION" == "plot" ]]; then
    log "operator=deepunfold_bj does not support standalone plot action; use sim/all."
    return
  fi

  log "Running BJ-DeepUnfold compare flow ..."
  run_cmd "mkdir -p '$out_dir'"
  run_cmd "cd '$ROOT_DIR' && python3 scripts/DeepUnfold/evaluate_bj_deepunfold_vs_chol_ldl.py --snr-db 0,5,10,15,20 --trials 3 --batch 24 --n-sc 32 --nt 16 --nr 64 --pilot-len 16 --bj-layers 12 --bj-block 4 --bj-adaptive-bounds --out-dir '$out_dir'"
}

run_deepunfold_bj_npu_opt_flow() {
  local out_dir="$PLOT_PNG"
  if [[ "$ACTION" == "plot" ]]; then
    log "operator=deepunfold_bj_npu_opt does not support standalone plot action; use sim/all."
    return
  fi

  log "Running BJ baseline vs NPU-opt compare flow ..."
  run_cmd "mkdir -p '$out_dir'"
  run_cmd "cd '$ROOT_DIR' && python3 scripts/DeepUnfold/compare_bj_baseline_vs_npu_opt.py --snr-db 0,5,10,15,20 --trials 2 --batch 12 --n-sc 16 --nt 16 --nr 64 --pilot-len 16 --npu-layers 12 --npu-adaptive-bounds --out-dir '$out_dir'"
}

main() {
  parse_args "$@"
  resolve_profile

  log "root=$ROOT_DIR action=$ACTION operator=$OPERATOR"

  if [[ "$OPERATOR" != "deepunfold_bj" ]]; then
    if [[ "$OPERATOR" == "deepunfold_bj_npu_opt" ]]; then
      run_deepunfold_bj_npu_opt_flow
      log "Done. out_dir=$PLOT_PNG"
      return
    fi

    run_build_if_needed
    run_simulator_flow
    log "Done. trace=$TRACE_CSV png=$PLOT_PNG"
  else
    run_deepunfold_bj_flow
    log "Done. out_dir=$PLOT_PNG"
  fi
}

main "$@"
