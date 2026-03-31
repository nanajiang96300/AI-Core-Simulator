# 多算子细节周期表导出指南

## 1. 是否需要不同算子用不同脚本？

结论：**建议“统一引擎 + 分算子入口脚本”**。

- 统一引擎：`scripts/export_operator_cycle_table.py`
  - 负责核心导出逻辑（事件聚合、周期统计、字段输出）。
- 分算子入口：
  - `scripts/export_cholesky_cycle_table.py`
  - `scripts/export_ldl_cycle_table.py`
  - `scripts/export_deepunfold_cycle_table_v2.py`

这样既能统一格式，又能针对不同算子维护不同公式映射规则。

---

## 2. 输出文件与字段

现在每次导出会生成两份 CSV：

1) **细粒度明细表**：`*_detailed_cycles_*.csv`
- `step_idx`：小步骤序号（明细行索引）
- `layer_idx`：阶段索引。`-1` 通常表示初始化/预处理（如 GRAM、REG），`0..N` 表示主循环阶段（例如第 j 列/第 j 层）
- `operator_mode`：识别到的算子模式（如 `chol_block`、`ldl_noblock`）
- `major_step`：所属大步骤（用于把多个小步骤聚成一个计算块）
- `event_key`：小步骤唯一键
- `onnx_op`：ONNX 语义运算类型（MatMul/Add/Div/Sqrt/...）
- `compute_op`：硬件执行类型（MATMUL/VECTOR_ADD/SCALAR_DIV/...）
- `formula`：该步骤公式
- `formula_dims`：公式维度（如 `64*8*8*64`）
- `compute_cycles`：该小步骤周期统计值（由 reducer 决定）
- `matched_events`：trace 事件名溯源
- `matched_units`：该小步骤匹配到的执行单元集合（如 `Core0_Vector|Core1_Vector|...`）

> `compute_op` 现已支持**按实际执行单元自动归一化**：
> - 若映射初值是 `VECTOR_*`，但 `matched_units` 全为 `*_Scalar`，则自动改为 `SCALAR_*`。
> - 若映射初值是 `SCALAR_*`，但 `matched_units` 全为 `*_Vector`，则自动改为 `VECTOR_*`。
> - 混合单元或无法判定时，保持原映射值。

2) **大步骤汇总表**：`*_major_summary.csv`
- `major_idx`：大步骤序号
- `layer_idx`：阶段索引（同上）
- `operator_mode`：算子模式
- `major_step`：大步骤键（如 `TRSM_3_1`、`D_BLOCK_7`）
- `compute_ops`：该大步骤涉及的执行类型集合
- `substep_count`：大步骤下包含的小步骤数量
- `major_cycle_sum`：该大步骤总周期（小步骤周期求和）
- `major_cycle_mean/max/min`：该大步骤内小步骤周期分布统计
- `substeps`：包含的小步骤 `event_key` 列表（可直接下钻到明细表）

---

## 3. 常用命令

### 3.1 Cholesky

```bash
cd /project/Asim
python3 scripts/export_cholesky_cycle_table.py \
  --trace results/scalar_refresh_20260331/cholesky_block_trace.csv \
  --output results/scalar_refresh_20260331/cholesky_block_detailed_cycles_v3.csv \
  --matrix-m 64 \
  --matrix-u 16 \
  --reducer median

# 可选：显式指定汇总表路径
# --summary-output results/scalar_refresh_20260331/cholesky_block_detailed_cycles_v3_major_summary.csv
```

### 3.2 LDL

```bash
cd /project/Asim
python3 scripts/export_ldl_cycle_table.py \
  --trace results/scalar_refresh_20260331/ldl_block_trace.csv \
  --output results/scalar_refresh_20260331/ldl_block_detailed_cycles_v3.csv \
  --matrix-m 64 \
  --matrix-u 16 \
  --reducer median
```

### 3.3 DeepUnfold

```bash
cd /project/Asim
python3 scripts/export_deepunfold_cycle_table_v2.py \
  --trace results/DeepUnfold/deepunfold_npu_opt_auto.csv \
  --output results/DeepUnfold/deepunfold_npu_opt_auto_detailed_cycles_v3.csv \
  --matrix-m 64 \
  --matrix-u 8 \
  --reducer median
```

### 3.4 一次性重导出（CHOL/LDL block+noblock）

```bash
cd /project/Asim

python3 scripts/export_cholesky_cycle_table.py \
  --trace results/scalar_refresh_20260331/cholesky_block_trace.csv \
  --output results/scalar_refresh_20260331/cholesky_block_detailed_cycles_v3.csv \
  --mode chol_block --matrix-m 64 --matrix-u 16

python3 scripts/export_cholesky_cycle_table.py \
  --trace results/scalar_refresh_20260331/cholesky_noblock_trace.csv \
  --output results/scalar_refresh_20260331/cholesky_noblock_detailed_cycles_v3.csv \
  --mode chol_nb --matrix-m 64 --matrix-u 16

python3 scripts/export_ldl_cycle_table.py \
  --trace results/scalar_refresh_20260331/ldl_block_trace.csv \
  --output results/scalar_refresh_20260331/ldl_block_detailed_cycles_v3.csv \
  --mode ldl_block --matrix-m 64 --matrix-u 16

python3 scripts/export_ldl_cycle_table.py \
  --trace results/scalar_refresh_20260331/ldl_noblock_trace.csv \
  --output results/scalar_refresh_20260331/ldl_noblock_detailed_cycles_v3.csv \
  --mode ldl_noblock --matrix-m 64 --matrix-u 16
```

---

## 4. 数据怎么看（先汇总后下钻）

- 先看 `*_major_summary.csv`：按 `major_cycle_sum` 降序，快速定位最耗时的大步骤。
- 再看 `substeps`：拿到该大步骤包含的 `event_key`。
- 到明细表 `*_detailed_cycles_*.csv` 过滤这些 `event_key`，看每个小步骤的 `formula/formula_dims/compute_cycles`。
- 结合 `layer_idx` 可判断是初始化阶段（`-1`）还是主迭代阶段（`>=0`），便于定位瓶颈是否在主流程。

示例（Cholesky）：
- `major_step=TRSM_5_2` 表示同一 $(i,j)=(5,2)$ 的 TRSM 大步骤。
- 该大步骤会聚合 `TRSM_NUM_UPD_5_2_k`（多个 k）以及 `TRSM_DIV_5_2` 等小步骤。

## 5. 口径建议

- 推荐 `--reducer median`：稳定反映多 core / 多 batch 的典型周期。
- 若做最坏时延分析，用 `--reducer max`。
- 若做累计负载分析，用 `--reducer sum`。

## 6. 关于“为什么仍有 VECTOR_MAC”

- `VECTOR_MAC` / `SCALAR_MAC` 的最终展示，以导出时的 `matched_units` 为准。
- 如果你看到 `VECTOR_MAC` 且 `matched_units` 全是 `Core*_Vector`，说明它确实是向量单元执行，不应改为 scalar。
- 只有当同类步骤实际跑在 `Core*_Scalar` 时，`compute_op` 才会自动显示为 `SCALAR_MAC`。
