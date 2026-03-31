# Cholesky vs LDL 详细对比报告（2026-03-27）

## 0. 结论先答复（对应你的问题）

1. **现在实验还要不要改？**
   - 若目标是“证明 Cholesky 算子正确性 + 解释与 LDL 性能差异”，当前结果已经足够。
   - 若目标是“做到与 LDL 周期几乎完全一致”，仍可继续做更深层同构（例如进一步压缩 Cholesky 的因子阶段依赖链）。

2. **Cholesky 与 LDL 是否已经对齐？**
   - **lowering 级别已显著对齐**：已从 `old -> invmul -> strict-iso` 连续改造，并将大量细粒度更新聚合。
   - **但并非完全同构到逐条指令等价**：在当前实现下，Cholesky 仍存在额外 `TRSM_NUM_UPD` / `RK_UPDATE` 结构，导致关键路径仍长于 LDL。

3. **Cholesky 算子正确性是否被验证？**
   - **是**。BER/SE 与 Exact、LDL 一致；逆矩阵相对误差在 $10^{-16}$ 量级。
   - 证据文件：`results/CHOL/falsification/cholesky_iso_ber_validation_20260327.txt`

4. **是否能说明 LDL 优越性？**
   - **能**。差异主要来自 factor 阶段依赖链和事件数，不是 `sqrt` 本身。
   - 在 strict-iso 对齐后，Cholesky 仍比 LDL 慢约 `1.46x`（Core0 与全局均一致量级）。

---

## 1. 公式推导与算法路径

### 1.1 Cholesky 路径（Hermitian SPD）

给定
$$
A = LL^H,
$$
其中 $L$ 为下三角。

逐列更新（标量表达）可写为：
$$
L_{jj} = \sqrt{A_{jj} - \sum_{k=0}^{j-1} |L_{jk}|^2}
$$
$$
L_{ij} = \frac{A_{ij} - \sum_{k=0}^{j-1} L_{ik}L_{jk}^*}{L_{jj}}, \quad i>j
$$

逆矩阵：
$$
A^{-1}=L^{-H}L^{-1}
$$

对应到当前算子分组：
- 因子阶段：`POTRF_DIAG_UPD/SQRT`, `TRSM_NUM_UPD/TRSM_DIAG_INV/TRSM_MUL`, `RK_UPDATE`
- 求逆拼装阶段：`FWD_DIAG_INV`, `FWD_OFF_MAC/FWD_OFF_MUL`, `BWD_MAC_FULL`

### 1.2 LDL 路径（Hermitian）

给定
$$
A = LDL^H,
$$
其中 $L$ 为单位下三角，$D$ 为对角（或块对角）。

标量表达：
$$
d_j = A_{jj} - \sum_{k=0}^{j-1} L_{jk}d_kL_{jk}^*
$$
$$
L_{ij} = \frac{A_{ij} - \sum_{k=0}^{j-1} L_{ik}d_kL_{jk}^*}{d_j}, \quad i>j
$$

逆矩阵：
$$
A^{-1}=L^{-H}D^{-1}L^{-1}
$$

对应算子分组：
- 因子阶段：`D_UPDATE`, `D_DIAG_INV`, `D_INV_MUL`, `L_UPDATE`
- 回代/拼装：`BWD_DIAG_MUL/ACC`, `BWD_OFF_MUL/ACC`

---

## 2. 正确性验证（Cholesky strict-iso）

验证脚本：`scripts/validate_cholesky_iso_ber_vs_ldl.py`

核心结果（`Exact / Cholesky-ISO / LDL`）：
- `SNR=0/5/10/15/20 dB` 上 BER 三者一致。
- SE 三者一致。
- `inv_relerr(iso)` 约 $2.1\sim2.2\times 10^{-16}$。

示例（20 dB）：
- BER: `0 / 0 / 0`
- SE: `132.5290 / 132.5290 / 132.5290`
- `inv_relerr iso/ldl = 2.157e-16 / 2.924e-16`

结论：**数值正确性和通信级正确性均通过**。

---

## 3. 演进对比（old / invmul / iso + LDL）

### 3.1 图与数据文件
- 演进图：`results/LDL/falsification/cholesky_ldl_evolution_compare_20260327.png`
- 汇总表：`results/LDL/falsification/cholesky_ldl_evolution_summary_20260327.csv`

### 3.2 关键数据（全局）

| Case | total_cnt | total_dur | max_end |
|---|---:|---:|---:|
| CHOL-old | 307796 | 29467524 | 97792 |
| CHOL-invmul | 178173 | 414276 | 15244 |
| CHOL-iso | 70653 | 306756 | 6284 |
| LDL-aligned | 51933 | 273400 | 4290 |

### 3.3 改造收益（以 max_end 为主）
- `old -> invmul`: `97792 -> 15244`（约 **6.42x** 提升）
- `invmul -> iso`: `15244 -> 6284`（约 **2.43x** 提升）
- `old -> iso`: `97792 -> 6284`（约 **15.56x** 提升）
- `iso vs LDL`: `6284 vs 4290`（CHOL 仍约 **1.46x**）

---

## 4. 单核（Core0）并排对比与分操作耗时

- 单核合并时序图：`results/LDL/falsification/ldl_cholesky_iso_core0_timeline.png`
- 单核分操作报告：`results/LDL/falsification/LDL_CHOLESKY_ISO_CORE0_COMPARE_20260327.md`

Core0 总周期：
- LDL: `4282`
- CHOL-ISO: `6276`
- 比值：`1.4657x`

阶段汇总（Core0）：
- Factor 阶段：LDL `864` vs CHOL `1952`（`2.259x`）
- Solve 阶段：LDL `1080` vs CHOL `1344`（`1.244x`）

关键解释：
- Cholesky 即使 strict-iso 后，仍保留 `TRSM_NUM_UPD` 与 `RK_UPDATE` 两类额外更新路径；
- 这些路径增加了事件数与依赖深度；
- `SQRT` 在总时长中占比小，不是主导项。

---

## 5. “是否已经对齐”的准确定义

- **已对齐（实现策略层）**：
  - 标量路径不走 Cube（Vector 映射已满足）
  - old/invmul/iso 已做连续 lowering 收敛
  - 调度与每核负载分布一致性良好

- **未完全等价（算法 DAG 层）**：
  - Cholesky 与 LDL 因子阶段依赖结构不同，导致同规模下仍有可观关键路径差异
  - 因此目前结果应解释为：
    - “已公平很多、可做可信比较”
    - 但不是“逐条指令同构到只剩 sqrt 差异”

---

## 6. 最终结论

1. **Cholesky strict-iso 算子正确**：BER/SE/逆矩阵误差验证通过。  
2. **LDL 仍更快且可解释**：优势主要在 factor 阶段依赖链更短、事件更少。  
3. **sqrt 不是主因**：即使对齐后，主差异仍来自更新路径结构。  
4. **演进收益清晰**：old→invmul→iso 的 lowering 改造已经显著缩小差距（从 97k 周期降到 6.3k）。

---

## 7. 复现实验命令

```bash
cd /project/Asim
python3 scripts/validate_cholesky_iso_ber_vs_ldl.py \
  --nr 64 --nt 16 --n-sc 8 --batch 4 --trials 2 \
  --snr-db 0,5,10,15,20 --pilot-len 16 --block-size 2 --seed 42

python3 scripts/compare_ldl_chol_iso_core0.py \
  --ldl-trace results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv \
  --chol-trace results/CHOL/falsification/cholesky_noblock_64x16_trace_iso.csv \
  --png results/LDL/falsification/ldl_cholesky_iso_core0_timeline.png \
  --report results/LDL/falsification/LDL_CHOLESKY_ISO_CORE0_COMPARE_20260327.md

python3 scripts/plot_cholesky_ldl_evolution.py \
  --png results/LDL/falsification/cholesky_ldl_evolution_compare_20260327.png \
  --csv results/LDL/falsification/cholesky_ldl_evolution_summary_20260327.csv
```
