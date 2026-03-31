# Block Cholesky(opt) vs Block LDL(opt2)：逐事件公式-单元-周期统计表

仅使用以下两份 trace：

- `results/CHOL/falsification/cholesky_block_64x16_trace_opt.csv`
- `results/LDL/falsification/ldl_block_64x16_trace_opt2.csv`

统计口径：

- `cnt`：事件条数；`dur`：事件总时长；`avg=dur/cnt`
- 旧口径（已弃用）：`share=dur/total_dur`
- 新口径（分域）：
	- **计算域占比**：`compute_share = dur / Σdur(计算域事件)`
	- **搬运域占比**：`transfer_share = dur / Σdur(Load+Store)`
	- **等待域占比**：`wait_share = dur / Σdur(Wait事件)`
- Cholesky(opt) 分母：`compute=54816, transfer=214909, wait=5927`
- LDL(opt2) 分母：`compute=36000, transfer=206408, wait=4991`

---

## A) Block Cholesky(opt) 逐事件表

| 事件步骤 | 对应公式/含义 | 计算单元 | 域 | cnt | dur | avg | 分域占比 |
|---|---|---|---|---|---|---|---|
| `CHOL_GRAM` | $G=H^HH$ | Cube | 计算 | 96 | 3360 | 35.00 | 6.13% (compute) |
| `CHOL_REG` | $A=G+\lambda I$ | Vector | 计算 | 96 | 96 | 1.00 | 0.18% (compute) |
| `CHOL_POTRF_DIAG_UPD` | $L_{jj}=\sqrt{A_{jj}-\sum_{k<j}|L_{jk}|^2}$ 中的对角累加更新项 | Vector | 计算 | 2688 | 2688 | 1.00 | 4.90% (compute) |
| `CHOL_POTRF_DIAG_SQRT` | $L_{jj}=\sqrt{\cdot}$ | Vector | 计算 | 768 | 3072 | 4.00 | 5.60% (compute) |
| `CHOL_TRSM_NUM_UPD` | $L_{ij}=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}^*}{L_{jj}}$ 中分子更新项 | Vector | 计算 | 5376 | 5376 | 1.00 | 9.81% (compute) |
| `CHOL_TRSM_DIV` | 上式中的除法（乘倒数） | Vector | 计算 | 2688 | 10752 | 4.00 | 19.61% (compute) |
| `CHOL_RK_UPDATE` | Schur 补更新（秩更新） | Vector | 计算 | 8064 | 8064 | 1.00 | 14.71% (compute) |
| `CHOL_FWD_DIAG_INV` | 前代中对角逆/归一化步骤 | Vector | 计算 | 768 | 3072 | 4.00 | 5.60% (compute) |
| `CHOL_FWD_OFF_MAC` | 前代非对角乘加 | Vector | 计算 | 2688 | 2688 | 1.00 | 4.90% (compute) |
| `CHOL_FWD_OFF_UPD` | 前代非对角更新（含缩放） | Vector | 计算 | 2688 | 10752 | 4.00 | 19.61% (compute) |
| `CHOL_BWD_MAC_FULL` | $A^{-1}=L^{-H}L^{-1}$ 回代矩阵乘（整块） | Cube | 计算 | 96 | 3072 | 32.00 | 5.60% (compute) |
| `CHOL_BARRIER` | 阶段同步屏障 | Vector | 计算 | 1824 | 1824 | 1.00 | 3.33% (compute) |
| `Load` | 输入搬运（DRAM->SPAD） | MTE2 | 搬运 | 3840 | 191813 | 49.95 | 89.25% (transfer) |
| `Store` | 结果回写（SPAD->DRAM） | MTE3 | 搬运 | 768 | 23096 | 30.07 | 10.75% (transfer) |
| `CubeWait` | Cube 等待（依赖/资源） | Wait | 等待 | 189 | 5927 | 31.36 | 100.00% (wait) |

---

## B) Block LDL(opt2) 逐事件表

| 事件步骤 | 对应公式/含义 | 计算单元 | 域 | cnt | dur | avg | 分域占比 |
|---|---|---|---|---|---|---|---|
| `LDL_GRAM` | $G=H^HH$ | Cube | 计算 | 96 | 3360 | 35.00 | 9.33% (compute) |
| `LDL_REG` | $A=G+\lambda I$ | Vector | 计算 | 96 | 96 | 1.00 | 0.27% (compute) |
| `LDL_D_UPDATE` | $D_j=A_{jj}-\sum_{k<j}L_{jk}D_kL_{jk}^H$ | Vector | 计算 | 768 | 768 | 1.00 | 2.13% (compute) |
| `LDL_D_DIAG_INV` | $D_j^{-1}$（对角块逆中的倒数步骤） | Vector | 计算 | 768 | 3072 | 4.00 | 8.53% (compute) |
| `LDL_D_INV_MUL` | $D_j^{-1}$ 后处理乘法 | Vector | 计算 | 768 | 768 | 1.00 | 2.13% (compute) |
| `LDL_L_UPDATE_CUBE` | $L_{ij}=\left(A_{ij}-\sum_{k<j}L_{ik}D_kL_{jk}^H\right)D_j^{-1}$ 中的 pack 主乘加（批量块更新） | Cube | 计算 | 576 | 18432 | 32.00 | 51.20% (compute) |
| `LDL_L_UPDATE_VECTOR` | 同一公式中的标量/小块尾项与归一化辅助步骤 | Vector | 计算 | 96 | 96 | 1.00 | 0.27% (compute) |
| `LDL_BWD_DIAG_MUL` | $X_{jj}=D_j^{-1}-\sum_{k>j}L_{kj}^HX_{jk}^H$ 中乘法项 | Vector | 计算 | 672 | 672 | 1.00 | 1.87% (compute) |
| `LDL_BWD_DIAG_ACC` | 上式累加项 | Vector | 计算 | 672 | 672 | 1.00 | 1.87% (compute) |
| `LDL_BWD_OFF_MUL` | $X_{ij}=-\sum_{k>i}L_{ki}^HX_{kj}$ 中乘法项 | Vector | 计算 | 2688 | 2688 | 1.00 | 7.47% (compute) |
| `LDL_BWD_OFF_ACC` | 上式累加项 | Vector | 计算 | 2688 | 2688 | 1.00 | 7.47% (compute) |
| `LDL_BARRIER` | 阶段同步屏障 | Vector | 计算 | 2688 | 2688 | 1.00 | 7.47% (compute) |
| `Load` | 输入搬运（DRAM->SPAD） | MTE2 | 搬运 | 3840 | 186103 | 48.46 | 90.16% (transfer) |
| `Store` | 结果回写（SPAD->DRAM） | MTE3 | 搬运 | 768 | 20305 | 26.44 | 9.84% (transfer) |
| `CubeWait` | Cube 等待（依赖/资源） | Wait | 等待 | 198 | 4991 | 25.21 | 100.00% (wait) |

---

## C) 直接可用结论（仅基于这两份结果）

1. 按“搬运域”口径看，`Load` 都占主导：CHOL `89.25%`、LDL `90.16%`（`Store` 分别 `10.75%`、`9.84%`）。
2. 按“计算域”口径看，CHOL 主要由 `TRSM_DIV/FWD_OFF_UPD/RK_UPDATE` 等步骤占比高；LDL 主要集中在 `LDL_L_UPDATE`，且可拆为 `Cube=51.20%` + `Vector=0.27%`。
3. `CubeWait` 单列为等待域（分母独立），避免与计算/搬运混算导致误解。
4. 因此新口径可同时回答“搬运瓶颈”和“算子内部计算热点”，比旧的单一 `dur/total_dur` 更可解释。

---

## D) 公式含义与在 LDL 中的作用（对应 `LDL_L_UPDATE`）

目标公式：

$$
L_{ij}=\left(A_{ij}-\sum_{k<j}L_{ik}D_kL_{jk}^H\right)D_j^{-1}
$$

逐项解释：

- $A_{ij}$：当前待更新的块（来自 Gram+正则后的系统矩阵分块）。
- $\sum_{k<j}L_{ik}D_kL_{jk}^H$：历史列 $k<j$ 对当前块的“已解释贡献”（Schur 补项），需要先减掉，避免重复计入。
- 括号内残差：当前列在已知前序因子后剩余的“净信息”。
- 右乘 $D_j^{-1}$：把残差投影到第 $j$ 个对角块尺度，得到最终的 $L_{ij}$。

在 LDL 分解中的作用：

1. **核心更新步骤**：它决定每一列（或块列）下三角因子 $L$ 的非对角块，直接影响后续所有列的更新质量。
2. **数值解耦**：通过显式的 $D_j$（块对角）把尺度从 $L$ 中分离出来，相比纯 Cholesky 形式更易处理某些缩放/条件数变化。
3. **对逆与检测链路的影响**：$L$ 与 $D$ 是后续前代/回代构造 $A^{-1}$（或等效求解）的基础，`L_UPDATE` 精度与性能都会传导到 BER/SE。

为何在实现上会出现 `Cube + Vector`（pack）：

- **Cube 主体**：pack 后的大块乘加（GEMM-like）主要映射到 Cube，因此 `LDL_L_UPDATE_CUBE` 占绝大多数时长（`51.20%` 计算域）。
- **Vector 辅助**：小块尾项、标量归一化或边界修正由 Vector 执行，时长很小（`0.27%` 计算域）。
- **工程含义**：这说明该公式的热点已经被很好地压到 Cube 主路径；若继续优化，应优先看 Cube 数据供给/并发，而不是该步骤的 Vector 算力。
