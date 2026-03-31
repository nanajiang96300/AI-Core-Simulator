# DeepUnfold 周期明细导出工具说明

## 1. 目标

在每次仿真后自动导出“可定位问题”的细粒度周期表，包含：

- 运算类别（如 `MATMUL` / `VECTOR_ADD` / `SCALAR`）
- 运算公式（如 `AX_k = A @ X_k`）
- 公式维度（如 `8*64*64*8`）
- 周期（`compute_cycles`）
- 样本数、聚合口径、匹配到的 trace 事件名

---

## 2. 工具与脚本

### 2.1 导出脚本

- `scripts/export_deepunfold_cycle_table.py`

输入：

- 仿真 trace：`name, unit, start_cycle, end_cycle`

输出：

- 详细周期表 CSV（默认：`<trace>_detailed_cycles.csv`）

### 2.2 一键流程接入

- `scripts/run_one_click.sh`

对 `deepunfold` / `deepunfold_opt` 已默认开启自动导出：

- `--export-cycle-table 1`（默认）

---

## 3. 命令示例

### 3.1 单独导出

```bash
cd /project/Asim
python3 scripts/export_deepunfold_cycle_table.py \
  --trace results/DeepUnfold/deepunfold_npu_opt_auto.csv \
  --output results/DeepUnfold/deepunfold_npu_opt_auto_detailed_cycles.csv \
  --mode auto \
  --matrix-m 64 \
  --matrix-u 8 \
  --reducer median
```

### 3.2 跑仿真并自动导出

```bash
cd /project/Asim
bash scripts/run_one_click.sh \
  --operator deepunfold_opt \
  --action all \
  --cycle-m 64 \
  --cycle-u 8 \
  --cycle-reducer median
```

---

## 4. 输出字段说明

导出 CSV 关键列：

- `step_idx`：导出后的步骤序号
- `layer_idx`：层号（非层内步骤记为 `-1`）
- `trace_mode`：`du` 或 `duo`（自动识别）
- `event_key`：事件类型键（如 `AX`, `RES`, `XNEXT`）
- `onnx_op`：ONNX 风格运算类别（如 `MatMul`, `Add`）
- `compute_op`：计算单元类别（如 `MATMUL`, `VECTOR_ADD`）
- `formula`：步骤公式字符串
- `formula_dims`：维度字符串（由 `--matrix-m/--matrix-u` 生成）
- `compute_cycles`：按聚合器得到的周期
- `sample_count`：该步骤匹配到的事件数
- `reducer`：聚合方式（`median|max|mean|sum`）
- `matched_events`：匹配到的 trace 事件名

---

## 5. 聚合口径建议

- 默认使用 `median`：对多 batch、多 core 重复事件更稳健。
- 若关注最坏情况，可用 `max`。
- 若关注累计代价，可用 `sum`。

---

## 6. 当前已验证输出

- `results/DeepUnfold/deepunfold_npu_auto_detailed_cycles.csv`
- `results/DeepUnfold/deepunfold_npu_opt_auto_detailed_cycles.csv`

已验证字段齐全，包含公式、维度与周期，满足组内定位分析需求。
