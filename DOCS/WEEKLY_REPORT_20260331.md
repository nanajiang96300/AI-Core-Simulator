# ONNXim 周工作汇报（2026-03-25 ~ 2026-03-31）

> 汇报人：Asim/ONNXim 仿真优化方向  
> 统计口径：以当前工作区代码、脚本、结果文件与文档为准（未依赖远端仓库状态）

---

## 1. 本周目标与结论总览

### 1.1 本周目标

1. 解决 LDL 分块结果“过好”的可信度问题，回调到更合理性能区间（目标约 3000 cycles）。
2. 产出 Cholesky / 标准 LDL / 分块 LDL 的统一对比时序图与可复查统计。
3. 实现最小可用 Scalar 执行单元（独立于 Vector），增强 Ascend 风格建模真实性。
4. 完成可视化可解释性修正（特别是 Cube/Wait 关系与坐标布局）。

### 1.2 关键结论

- LDL 分块温和优化版本达到目标区间：`global_max_end = 3032`。
- 三路对比（同脚本同口径）显示：`LDL_OPT (3032) < LDL (4290) < CHOLESKY(no-block, 6284)`。
- 已打通 Scalar 最小闭环：`opcode -> issue -> pipeline -> cycles -> trace -> stats`。
- 当前 trace 中 `Cube` 与 `Wait` 存在真实时间交叠（4 对区间交叠），因此默认不强行同排。

---

## 2. 里程碑与时间线（按任务流）

### 2.1 LDL 分块“温和优化”收敛（核心性能调形）

- 主要策略：
  - 限制自动 pack 激进度（`auto_pack_blocks <= 2`）。
  - 调整 `D_UPDATE` 的 `k_len` 路由逻辑（避免过激并行导致不自然收益）。
  - 用配置组合控制并行步幅与打包力度：`bwd_steps=3`、`pack_blocks=2`。
- 关键配置：`example/ldl_test_moderate3.json`。
- 结果落点：
  - 目标达成：`~3000 cycles`。
  - 产物确认：`results/LDL/falsification/ldl_block_64x16_trace_moderate3.csv`。

### 2.2 三路时序对比与标签统一

- 新增/更新脚本：`scripts/compare_chol_stdldl_blockldl_timeline.py`。
- 完成项：
  - 首图切换为 no-block Cholesky。
  - 标题统一为 `Cholesky` / `LDL` / `LDL_Opt`。
  - 输出统一统计 CSV + PNG。
- 关键产物：
  - `results/LDL/falsification/chol_noblock_stdldl_blockldl_timeline_compare_moderate3_20260327.png`
  - `results/LDL/falsification/chol_noblock_stdldl_blockldl_timeline_compare_moderate3_20260327.csv`

### 2.3 Scalar 最小可用实现（架构可解释性升级）

- 设计原则：低侵入、可观测、可配置、可复用现有 latency 参数。
- 完成链路：
  - 指令：新增 `SCALAR_ADD/MUL/DIV/SQRT`。
  - 调度：新增 scalar issue 判定 + scalar pipeline。
  - 计时：`get_scalar_compute_cycles()` 映射到 `scalar_*_latency/div_latency`。
  - 观测：trace 新增 `CoreX_Scalar`，统计加入 scalar 维度。
  - 算子映射：在 `CholeskyInvNoBlockOp` 中将关键标量语义步骤映射为 scalar opcode。
- 关键产物：
  - trace：`results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv`
  - 图：`results/CHOL/falsification/cholesky_noblock_64x16_timeline_iso_scalar.png`
  - 报告：`DOCS/SCALAR_MINIMAL_IMPLEMENTATION_REPORT.md`

### 2.4 可视化解释修正（Cube/Wait）

- 做法：
  - 调整 lane 顺序使 `Cube` 与 `Wait` 相邻，降低误判。
  - 增加自动判定逻辑：仅在“无时间重叠”时才同排绘制。
- 结果：
  - 当前数据判定 `layout = separate-cube-wait`。
  - 解释闭环：不是绘图错误，而是多链路流水交错造成真实时间交叠。

---

## 3. 量化结果（可直接口播）

### 3.1 三路对比关键数字（来自 CSV）

来源：`results/LDL/falsification/chol_noblock_stdldl_blockldl_timeline_compare_moderate3_20260327.csv`

| Case | global_max_end | core0_max_end | Cube% | Vector% | MTE2% | Wait% | MTE3% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cholesky | 6284 | 6276 | 1.88 | 23.20 | 65.83 | 1.77 | 7.31 |
| LDL | 4290 | 4282 | 1.10 | 16.87 | 72.85 | 1.01 | 8.17 |
| LDL_Opt | 3032 | 2962 | 11.53 | 8.00 | 62.53 | 9.68 | 8.25 |

解读：

- 本周目标（`~3000`）由 `LDL_Opt=3032` 达成。
- 优化后 `Cube` 与 `Wait` 占比提高，反映并行发射更积极，同时依赖等待窗口被显式暴露。

### 3.2 Scalar 验证数据（Core0）

来源：`results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv`

- `core0_max_end = 5635`
- 事件计数：
  - `Vector=2544`
  - `MTE2=160`
  - `Wait=8`
  - `Cube=8`
  - `Scalar=192`
  - `MTE3=32`
- duration-sum（cycle）：
  - `Vector=2544`
  - `MTE2=10660`
  - `Wait=276`
  - `Cube=268`
  - `Scalar=768`
  - `MTE3=1789`

结论：

- `Core0_Scalar` 事件与时长均可观测，证明 Scalar 执行通道已生效。

### 3.3 Cube/Wait 重叠证据

在 Core0 中检测到 `cube_wait_overlap_pairs = 4`。示例：

- `Cube: [118,153)` 与 `Wait: [98,134)` 重叠区间为 `[118,134)`。

解释：

- `Wait` 与 `Cube` 来自不同指令链条的时段可以并存；一个 tile 在等待时，另一个 tile 可进入 Cube 计算，形成交错流水。

---

## 4. 代码改动面（按模块）

### 4.1 核心执行模型

- `src/Common.h`：新增 Scalar opcode。
- `src/Core.h` / `src/Core.cc`：新增 scalar pipeline、完成回调、统计与 trace 接入。
- `src/SystolicWS.h` / `src/SystolicWS.cc`：
  - scalar issue 路径；
  - `get_scalar_compute_cycles()`；
  - cycle 阶段推进 scalar pipeline。

### 4.2 算子侧

- `src/operations/CholeskyInvNoBlockOp.cc`：关键标量步骤切换到 scalar opcode。
- `src/operations/LDLDecompOp.cc`：温和优化参数与路径调节。

### 4.3 脚本与文档

- 脚本：
  - `scripts/compare_chol_stdldl_blockldl_timeline.py`
  - `scripts/plot_cholesky_core0_timeline_with_scalar.py`
- 文档：
  - `DOCS/SCALAR_MINIMAL_IMPLEMENTATION_REPORT.md`
  - `results/LDL/falsification/LDL_BLOCK_MODERATE3_COMPARE_20260327.md`
  - 及 LDL/CHOL 对比系列报告（`results/LDL/falsification/*.md`）

---

## 5. 计时模型说明（组会讲解版）

Scalar 指令周期模型：

- `SCALAR_ADD -> scalar_add_latency`
- `SCALAR_MUL -> scalar_mul_latency`
- `SCALAR_SQRT -> scalar_sqrt_latency`
- `SCALAR_DIV -> div_latency`

统一公式：

- `start_cycle = issue 时刻`
- `finish_cycle = start_cycle + latency(opcode)`

串行示例（单 Scalar pipeline）：

- 设 `t=100`，`scalar_add_latency=2`，`scalar_mul_latency=3`，`div_latency=10`。
- 执行区间：
  - `ADD: [100,102)`
  - `MUL: [102,105)`
  - `DIV: [105,115)`
- 总耗时：`15 cycles`。

---

## 6. 验证与复现实操（附命令）

> 以下命令为本周常用核查路径，可直接复用。

```bash
cd /project/Asim
python3 scripts/plot_cholesky_core0_timeline_with_scalar.py \
  --trace results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv \
  --png results/CHOL/falsification/cholesky_noblock_64x16_timeline_iso_scalar.png \
  --title "Cholesky - Core0"
```

```bash
cd /project/Asim
python3 - <<'PY'
import csv
from pathlib import Path
p=Path('results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv')
cube=[]; wait=[]
with p.open() as f:
    r=csv.DictReader(f)
    for row in r:
        u=row['unit']
        if not u.startswith('Core0_'): continue
        lane=u.split('_',1)[1]
        s=int(row['start_cycle']); e=int(row['end_cycle'])
        if lane=='Cube': cube.append((s,e))
        if lane=='Wait': wait.append((s,e))
ov=0
for c in cube:
    for w in wait:
        if max(c[0],w[0]) < min(c[1],w[1]):
            ov+=1
print('cube_wait_overlap_pairs=', ov)
PY
```

---

## 7. 风险、边界与待办

### 7.1 风险/边界

1. 当前 Scalar opcode 映射主要覆盖 no-block Cholesky 关键路径，尚未全算子统一。
2. 目前工作区存在大量未提交文件，若组内需要可审计 PR，需要先做变更分组与清理。
3. 本地 `git log --since=7 days` 无提交记录，周报主要依据“工作区产物+文档时间戳+实验文件”组织。

