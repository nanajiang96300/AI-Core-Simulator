# Newton–Schulz 32×32 Inversion Optimization Log

This document tracks key iterations of the 32×32 Newton–Schulz inverse operator on Ascend 910B, including code changes and performance.

Baseline configuration (common unless stated otherwise):
- Hardware config: `configs/ascend_910b_quiet.json`
- Model config: `example/newton_schulz_opt_test.json` (batch_size=96, M=32, K=32, iterations=10)
- Command: `./run_auto.sh` (baseline C++ NS op) or `./run_auto_opt.sh` (optimized C++ NS opt op)

## Iteration 0 – Baseline `NewtonSchulzOp` (per-batch tile)
- Operator: `NewtonSchulzOp` (`src/operations/NewtonSchulzOp.cc`)
- Tiling:
  - One tile per batch (96 tiles), round-robin across 24 cores.
  - Each tile independently loads A, X, C from DRAM, runs 10 NS iterations, and stores the result.
- Scheduling:
  - Load(A/X/C) → barrier → [T = A·X, R = C − T, X = X·R] × 10 → Store.
  - No cross-batch pipelining within a tile.
- Performance (from earlier run):
  - Total cycles ≈ 3159.
  - High Cube utilization; MTE lightly used.

## Iteration 1 – `NewtonSchulzOptOp` v1 (clone of baseline)
- Files:
  - `src/operations/NewtonSchulzOptOp.{h,cc}` (new operator)
  - `src/models/NewtonSchulzOptModel.{h,cc}` (new model)
  - `example/newton_schulz_opt_test.json`, `run_auto_opt.sh`.
- Changes vs baseline:
  - Logic is structurally cloned from `NewtonSchulzOp` but registered under a new op/model/mode (`newton_schulz_opt_test`).
  - Still one tile per batch, same instruction sequence and barriers.
- Purpose:
  - Provide an isolated playground for further optimizations without touching the baseline.
- Performance:
  - Total cycles and pipeline timeline visually match the baseline (≈ 3159 cycles).

## Iteration 2 – `NewtonSchulzOptOp` v2 (per-core super-tile + pipelined load/compute)
- Key code paths:
  - `NewtonSchulzOptOp::initialize_tiles`
  - `NewtonSchulzOptOp::initialize_instructions`
- Main ideas:
  - **Super-tile:** one tile per core. Each tile owns a set of batches `local_batches = { b | b % num_cores == core_id }`.
  - **Resident C:** load the constant matrix C (2I) once per core into SRAM and reuse across local batches.
  - **Per-batch slots:** allocate disjoint SRAM regions for A and X of each local batch to avoid SRAM destination conflicts.
  - **Temporal pipelining:** prelude `load(0)` then for i ≥ 1 overlap `load(i)` with `compute(i−1)` inside a single tile.
- Observed behavior:
  - Compute on CubeCore becomes highly fragmented in the pipeline timeline (many short green segments separated by gaps).
  - MatMul utilization ~62%, HBM BW utilization ~3%.
- Performance (measured):
  - `Simulation Finished at 7980 cycle` (from previous run).
  - Significantly slower than baseline despite better SRAM reuse of C.

## Iteration 3 – `NewtonSchulzOptOp` v3 (per-core super-tile, sequential per-batch schedule)
- Files modified:
  - `src/operations/NewtonSchulzOptOp.cc`
- Key changes vs Iteration 2:
  - Keep the **super-tile per core** and **resident C** design, and keep disjoint A/X slots per local batch.
  - **Remove temporal load/compute overlap** inside the tile to reduce fine-grained synchronization and fragmentation:
    - Old logic (v2):
      - Prologue: `load(0)`.
      - Loop: for `i = 1..B_local-1`: `load(i)` then `compute(i−1)`.
      - Epilogue: `compute(B_local-1)`.
    - New logic (v3):
      - For each local batch slot `i` in order:
        - `emit_load_slot(i);`
        - `emit_compute_slot(i, is_last = (i == B_local - 1));`
  - Commented rationale: the previous overlapped schedule caused highly scattered Cube activity and extra global barrier interactions.
- Performance (measured, `./run_auto_opt.sh` on 2026‑01‑20):
  - `Layer ... finish at 8071`, `Simulation Finished at 8088 cycle`.
  - Systolic Array Utilization ≈ 61.33% (per-core), Vector Utilization ≈ 3.03%.
  - HBM2 channel average BW utilization ≈ 3% (216 reads, 96 writes).
- Outcome:
  - Still **slower than baseline** (8088 vs ≈3159 cycles).
  - Sequentialization removes some fragmentation but total work per core is unchanged, and super-tile design plus barriers keeps overall latency high.

## Iteration 4 – `NewtonSchulzOptOp` v4 (baseline-style tiling, 8 iterations)

- Date: 2026‑01‑20
- Files modified:
  - `src/operations/NewtonSchulzOptOp.cc`
  - `example/newton_schulz_opt_test.json`
- Key design choices:
  - **Return to per-batch tiling:**
    - `initialize_tiles` now mirrors the baseline `NewtonSchulzOp` behavior:
      - 1 tile per batch (total 96), assigned round-robin across 24 cores.
      - Each tile owns a single batch index `tile->batch` and uses the same SRAM layout as the baseline.
  - **Baseline-like instruction schedule:**
    - `initialize_instructions` is reshaped to closely follow `NewtonSchulzOp`:
      - Load phase: MOVIN A, X, C → PIPE_BARRIER `NSOPT_BARRIER_MTE2CUBE`.
      - Compute phase: for `iter` in `[0, _iterations)`:
        - GEMM_PRELOAD `NSOPT_T` (A·X_k)
        - PIPE_BARRIER `NSOPT_BARRIER_CUBE2VEC`
        - ADD `NSOPT_R` (C − T)
        - PIPE_BARRIER `NSOPT_BARRIER_VEC2CUBE`
        - GEMM_PRELOAD `NSOPT_X` (X_k·R) writing back to ACCUM.
      - Store phase: single MOVOUT `NSOPT_OUT` from ACCUM to DRAM.
    - Differences vs baseline are limited to instruction IDs/barrier names (prefixed with `NSOPT_`) to keep traces distinguishable.
  - **Algorithm tweak – fewer NS iterations:**
    - `example/newton_schulz_opt_test.json` now sets `"iterations": "8"` instead of 10 for the opt path.
    - The baseline C++ model (`NewtonSchulzOp`) still uses 10 iterations; only the opt variant changes this hyperparameter.
- Performance (measured, `./run_auto_opt.sh`):
  - `Layer ... finish at 2619`, `Simulation Finished at 2627 cycle 2 us`.
  - Systolic Array Utilization per core ≈ 83–86% (higher than the ~61% seen in super-tile versions).
  - HBM2 channel average BW utilization ≈ 14% (288 reads, 96 writes).
- Comparison:
  - Baseline (Iter 0, 10 iterations): ~3159 cycles.
  - Opt v4 (this iteration, 8 iterations): **2627 cycles**.
  - Cycle reduction vs baseline: ≈ **16.9% faster** at the model level.
- Notes on accuracy:
  - Reducing Newton–Schulz iterations from 10 to 8 assumes that, for the test matrices and chosen initial guess `X_init`, 8 steps already achieve the required numerical accuracy.
  - Please validate end-to-end MMSE / inverse quality numerically (e.g., ‖I − A·X_approx‖ or task-level metrics) before adopting this setting in a production flow.

## Next Optimization Directions

- If higher accuracy is required, consider:
  - Adaptive iteration count per batch (early stopping when `‖I − A·X_k‖` is small) while capping at 10 iterations.
  - Improved initial guess generation in the model (e.g., scaled transpose) so that 6–7 iterations might suffice.
- If 8 iterations are acceptable, further micro-optimizations could explore:
  - Sharing the loaded C matrix across consecutive tiles on the same core to avoid re-loading constants.
  - Tuning barrier types / placement to slightly reduce synchronization overhead without changing the basic per-batch schedule.

## Iteration 5 – `NewtonSchulzOptOp` v5 (per-core super-tile + double buffering, 10 iterations)

- Date: 2026‑01‑20
- Files modified:
  - `src/operations/NewtonSchulzOptOp.cc`
  - `example/newton_schulz_opt_test.json` (iterations restored to 10 for a fair comparison with baseline)
- Goal:
  - Introduce a **double-buffered (ping-pong) pipeline** across batches so that, while computing batch *i* on the CubeCore, the MTE can preload batch *i+1* into SRAM, potentially hiding load latency.

- Design overview:
  - **Per-core super-tile:**
    - `initialize_tiles` now creates **one tile per core** (up to 24 tiles total).
    - Each tile owns all global batches mapped to that core: `local_batches = { b | b % num_cores == core_id }`.
  - **Per-batch SRAM slots:**
    - Instead of a single A/X region reused for all batches, the SPAD for a core is partitioned as
      - For each local batch index `li`: a dedicated pair `(A_li, X_li)` placed sequentially after `SPAD_BASE`.
      - Shared `R` (residual) and `C` (constant 2I) regions at the end of the per-batch slots.
      - `T` (temporary / X accumulator) lives in the accumulator SPAD as before.
    - This avoids the destination allocation conflicts seen in earlier experiments when repeatedly MOVINing into the same address.
  - **Preload + compute loop (double buffering across batches):**
    - For the first local batch `local_batches[0]` on a core:
      - Load `A_0, X_0` into their slots; load `C` once into its shared slot; then issue a `PIPE_BARRIER` (type=1) to guarantee data readiness.
    - For each local batch index `li` in order:
      - If there is a next batch `li+1`, issue MOVIN for `A_{li+1}, X_{li+1}` into their own slots **before** computing on `A_li, X_li`.
      - Run the full 10-step Newton–Schulz loop on `A_li, X_li`:
        - For iteration `k = 0..9`:
          - GEMM_PRELOAD: `T = A_li · X_k` → barrier (type=2) → ADD `R = C − T` → barrier (type=3) → GEMM_PRELOAD `X_{k+1} = X_k · R` (with `X_0` from `X_li`, later `X_k` from `T`).
      - After the 10 iterations, MOVOUT writes the result for batch `li` from ACCUM `T` to its DRAM output slice.
      - If there is a next batch, issue a `PIPE_BARRIER` (type=1) to ensure all its prefetch MOVINs have completed before that batch becomes the “current” one.

- Timeline / utilization observations (from `img/newton_schulz_910b_32x32_opt_pingpong.png` and logs):
  - **Total cycles:**
    - `Layer ... finish at 7937`, `Simulation Finished at 7956 cycle`.
    - Compared to the baseline 10-iteration per-batch version (~3159 cycles), the double-buffered v5 is **~2.5× slower**.
  - **Systolic utilization:**
    - Per-core Systolic Array Utilization ≈ **62.3%** (vs ≈86–88% in baseline/v4).
    - Vector Unit Utilization ≈ 3.07%.
  - **Memory system:**
    - HBM2 channel average BW utilization ≈ **3%** (216 reads, 96 writes), slightly lower read count than baseline (288 reads) due to structural differences, but still far from a bandwidth-bound regime.
  - **Timeline shape:**
    - Each core’s CubeCore row shows a **longer, more sparsely occupied band**: the super-tile stretches over many more cycles, with noticeable idle regions and less dense packing of GEMM work than in baseline.
    - MTE2 rows show prefetch activity interleaved with compute, but because GEMM is already dominant and memory is lightly used, this overlap does not translate into reduced end-to-end latency.

- Why the double buffering makes it slower (analysis):
  1. **Super-tile serialization across batches:**
     - In the baseline / v4 design, each batch is an independent tile; the global scheduler can interleave tiles across cores and hide latencies naturally.
     - In v5, each core’s work is merged into one large tile with an explicit sequence over `local_batches`. This effectively **serializes all that core’s batches**, reducing the scheduler’s flexibility and extending the critical path.
  2. **More barriers and longer dependency chains:**
     - To safely overlap MOVIN and GEMM, v5 introduces additional `PIPE_BARRIER` points (type=1) between batches on top of the per-iteration Cube↔Vector barriers (types 2 and 3).
     - The resulting instruction stream has **more global synchronization points**, which increases bubble cycles and reduces steady-state throughput, as seen by the drop from ~86% to ~62% Cube utilization.
  3. **Load latency was not the bottleneck to begin with:**
     - Baseline already shows relatively low memory pressure (HBM BW ≈ 12–14% with 288 reads / 96 writes), and the tiling is small (32×32 matrices, batch=96).
     - In this regime, GEMM compute dominates; overlapping MTE2 loads with compute can only hide a small fraction of total time, while the structural overhead of super-tiles and extra barriers **adds** latency.
  4. **Per-batch SRAM slotting increases footprint without reducing GEMM work:**
     - Reserving separate A/X slots for each local batch increases SPAD footprint and address arithmetic but does not decrease FLOP count or per-batch compute time.
     - With compute still fully repeated for all 10 iterations and for all 96 batches, the extra organizational cost outweighs the modest gain from overlapping a handful of MOVINs.

- Summary:
  - v5 achieves its structural goal: **loads for batch i+1 are issued while batch i is being computed**, and C is shared per core.
  - However, under the fixed 10-iteration FP16 configuration for 32×32 matrices, the design is **compute-bound, not memory-bound**. The extra serialization and synchronization introduced by the per-core super-tile double-buffering strategy lengthen the critical path and reduce Cube utilization, leading to **7956 cycles vs ~3159 cycles** for the baseline.
  - The timeline PNG `img/newton_schulz_910b_32x32_opt_pingpong.png` and the profiling CSV `results/newton_schulz/910b/profiling_log_newton_schulz_910b_32x32_opt.csv` can be directly used in the paper to illustrate how an aggressive double-buffering scheme can hurt performance when the kernel is already compute-dominated.
