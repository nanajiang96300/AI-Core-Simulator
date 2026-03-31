# Cholesky 分解求逆算子全流程导出（`CholeskyInvOp`）

## 1. 文档目标

本文对 `src/operations/CholeskyInvOp.cc` 做完整流程导出，覆盖：

1. 算法数学流程；
2. 指令阶段映射；
3. 关键循环的事件计数公式；
4. `GEMM_PRELOAD`（Cube）高占比来源；
5. 默认测试配置下的代入结果。

---

## 2. 输入输出与默认参数

代码来源：

- 算子：`src/operations/CholeskyInvOp.cc`
- 模型装配：`src/models/CholeskyModel.cc`
- 默认样例：`example/cholesky_test.json`

张量语义：

- 输入 `H`：`[B, M, U]`
- 输入 `RegI`：`[U, U]`
- 输出 `Ainv`：`[B, U, U]`

默认参数（`example/cholesky_test.json`）：

$$
B_{batch}=96,\quad M=64,\quad U=16,\quad blk=2,\quad solve\_steps=1
$$

块数记为：

$$
N = U/blk
$$

默认下 $N=8$。

---

## 3. 算法流程（数学层）

### 3.1 正定矩阵构造

先构造：

$$
G = H^H H,
$$

再正则化：

$$
A = G + \lambda I
$$

### 3.2 块 Cholesky 分解

对块矩阵 $A$ 做下三角分解：

$$
A = L L^H
$$

其中每个块大小为 `blk x blk`（默认 `2x2`）。

### 3.3 前代求解

求 $Y=L^{-1}$（列方式前代）。

### 3.4 回代组装逆

由

$$
A^{-1} = L^{-H}L^{-1} = Y^H Y
$$

得到最终逆矩阵。

---

## 4. 指令级阶段映射（实现层）

`initialize_instructions()` 的阶段顺序如下。

### 4.0 分块的具体流程（按 `j/i/k` 展开）

设块大小 `blk=2`，块数 `N=U/blk`。把矩阵按块表示为 $A_{ij}$、$L_{ij}$，每块都是 `2x2`。

对每个列块 `j=0..N-1`，分解阶段执行以下 3 步：

1. **对角块更新（先减历史项）**

$$
S_{jj}=\sum_{k=0}^{j-1}L_{jk}L_{jk}^H,\quad
	ilde{A}_{jj}=A_{jj}-S_{jj}
$$

对应指令：`CHOL_POTRF_DIAG_UPD_j_k`（循环变量 `k`）。

2. **对角块开方（得到 $L_{jj}$）**

$$
L_{jj}=\text{chol}(\tilde{A}_{jj})
$$

对应指令：`CHOL_POTRF_DIAG_SQRT_j`。

3. **列下方块更新（`i>j`）**

先做分子更新：

$$
S_{ij}=\sum_{k=0}^{j-1}L_{ik}L_{jk}^H,\quad
	ilde{A}_{ij}=A_{ij}-S_{ij}
$$

对应指令：`CHOL_TRSM_NUM_UPD_i_j_k`。

再做三角求解（除以对角块）：

$$
L_{ij}=\tilde{A}_{ij}L_{jj}^{-H}
$$

对应指令：`CHOL_TRSM_DIV_i_j`。

最后做 trailing 子块更新（Rank-k 形式）：

$$
A_{ik}\leftarrow A_{ik}-L_{ij}L_{kj}^H,
\quad i\in[j+1,N-1],\ k\in[i,N-1]
$$

对应指令：`CHOL_RK_UPDATE_i_k_j`。

每列 `j` 结束后用 `CHOL_BARRIER_FACTOR_STEP_j` 封住列依赖，保证下一列读取到完整的 `L_{*j}`。

#### `2x2` 分块下的直观例子（`U=16 => N=8`）

- 当 `j=0`：
   - 无历史项（`k` 循环为空）；
   - 执行 `SQRT(0)`；
   - 对 `i=1..7` 执行 `TRSM_DIV(i,0)`；
   - 对全部下三角尾块执行 `RK_UPDATE(*,*,0)`。

- 当 `j=1`：
   - `k=0`，先执行一次 `POTRF_DIAG_UPD(1,0)`；
   - `SQRT(1)`；
   - 对 `i=2..7`，每个都先 `TRSM_NUM_UPD(i,1,0)` 再 `TRSM_DIV(i,1)`；
   - 然后做 `RK_UPDATE(*,*,1)`。

- 当 `j` 增大：
   - `k` 的长度随 `j` 增长（历史项更多）；
   - `i` 的长度随 `j` 减小（待更新行更少）；
   - 但三重循环总体仍给出 $O(N^3)$ 的更新数量。

### 阶段 S0：Load

1. `MOVIN(H)`
2. `MOVIN(RegI)`
3. `PIPE_BARRIER(CHOL_BARRIER_LOAD2GRAM)`

### 阶段 S1：Gram + Regularization

1. `GEMM_PRELOAD(CHOL_GRAM)` 对应 $H^H H$
2. `ADD(CHOL_REG)` 对应 $G+\lambda I$
3. `PIPE_BARRIER(CHOL_BARRIER_REG2FACTOR)`

### 阶段 S2：Factor（按列块 `j=0..N-1`）

每列 `j`：

1. `POTRF_DIAG_UPD(j,k)`：`k=0..j-1`（对角更新累积）
2. `SQRT(POTRF_DIAG_SQRT_j)`：对角开方
3. 对每个 `i=j+1..N-1`：
   - `TRSM_NUM_UPD(i,j,k)`：`k=0..j-1`
   - `TRSM_DIV(i,j)`
4. 对每个 `i=j+1..N-1`、`k=i..N-1`：
   - `RK_UPDATE(i,k,j)`
5. `PIPE_BARRIER(CHOL_BARRIER_FACTOR_STEP_j)`

### 阶段 S3：Forward Solve（按列 `c=0..N-1`）

每列 `c`：

1. `FWD_DIAG_INV(c)`
2. 对每个 `i=c+1..N-1`：
   - `FWD_OFF_MAC(i,c)`
   - `FWD_OFF_UPD(i,c)`
3. `PIPE_BARRIER(CHOL_BARRIER_FWD_COL_c)`

### 阶段 S4：Assemble + Store

1. `GEMM_PRELOAD(CHOL_BWD_MAC_FULL)` 对应 $Y^H Y$
2. `PIPE_BARRIER(CHOL_BARRIER_SOLVE2STORE)`
3. `MOVOUT(CHOL_OUT)`

---

## 5. 事件计数公式（每 batch）

以下只按循环结构计数，与具体周期模型解耦。

### 5.1 Cube 指令（`GEMM_PRELOAD`）

1. `CHOL_GRAM`：

$$
1
$$

2. `CHOL_POTRF_DIAG_UPD`：

$$
\sum_{j=0}^{N-1} j = \frac{N(N-1)}{2}
$$

3. `CHOL_TRSM_NUM_UPD`：

$$
\sum_{j=0}^{N-1}(N-j-1)j = \frac{N(N-1)(N-2)}{6}
$$

4. `CHOL_RK_UPDATE`：

$$
\sum_{j=0}^{N-1}\sum_{i=j+1}^{N-1}(N-i)=\frac{(N-1)N(N+1)}{6}
$$

5. `CHOL_FWD_OFF_MAC`：

$$
\sum_{c=0}^{N-1}(N-c-1)=\frac{N(N-1)}{2}
$$

6. `CHOL_BWD_MAC_FULL`：

$$
1
$$

总 Cube 数：

$$
N_{cube} = 2 + \frac{N(N-1)(2N+5)}{6}
$$

### 5.2 Vector 指令

1. `ADD`：`1`
2. `SQRT`：`N`
3. `DIV`（TRSM + FWD）：

$$
\frac{N(N-1)}{2} + N
$$

### 5.3 Barrier 指令

- `LOAD2GRAM`：1
- `REG2FACTOR`：1
- `FACTOR_STEP_j`：$N$
- `FWD_COL_c`：$N$
- `SOLVE2STORE`：1

总计：

$$
N_{barrier}=2N+3
$$

---

## 6. 默认配置代入（`U=16, blk=2 => N=8`）

### 6.1 每 batch 事件数

Cube（`GEMM_PRELOAD`）分项：

- `CHOL_GRAM = 1`
- `CHOL_POTRF_DIAG_UPD = 28`
- `CHOL_TRSM_NUM_UPD = 56`
- `CHOL_RK_UPDATE = 84`
- `CHOL_FWD_OFF_MAC = 28`
- `CHOL_BWD_MAC_FULL = 1`

总计：

$$
N_{cube}=198
$$

其它：

- `ADD = 1`
- `SQRT = 8`
- `DIV = 36`
- `BARRIER = 19`

### 6.2 全 batch（96）

$$
N_{cube,total}=198\times96=19008
$$

### 6.3 Cube 热点解释

`CHOL_RK_UPDATE` 与 `CHOL_TRSM_NUM_UPD` 都来自嵌套循环，主导项是 $O(N^3)$，因此当 `U/blk` 变大时，Cube 数增长最快。

---

## 7. 为什么 Cholesky 会用 `GEMM_PRELOAD`

在该工程里，`GEMM_PRELOAD` 表示“走 Cube 乘法管线 + preload 时序模型”的**实现指令类型**，并不要求是 DNN 中“固定权重矩阵”。

Cholesky 中大量步骤本质是小块矩阵乘/Rank-k 更新（例如 `RK_UPDATE`、`GRAM`、`Y^H Y`），因此映射到 `GEMM_PRELOAD` 是符合算法语义的。

---

## 8. 并行与依赖

1. **Batch 级并行**：`initialize_tiles()` 中 `batch -> core` 轮转分配；
2. **阶段内串并混合**：同一列内多条更新可排队，但列间依赖受 barrier 约束；
3. **关键依赖点**：
   - 分解列 `j` 必须在 `FACTOR_STEP_j` 后进入下一列；
   - 前代列 `c` 在 `FWD_COL_c` 后进入下一列；
   - `Y^H Y` 必须在前代全部完成后执行。

---

## 9. 与 LDL 路径的结构差异（简述）

1. Cholesky 分解阶段显式包含更多 rank-k / TRSM 更新，因此 Cube 项更“立方化”；
2. LDL 路径在本实现中通过 `pack_blocks` 与长 `tile_k` 折叠了部分循环，事件结构不同；
3. 两者都在最终逆组装处使用矩阵乘，但 Cholesky 在前段分解就已经产生大量 Cube。

---

## 10. 代码锚点（便于回查）

- `parse_attributes()`：读取 `batch_size/block_size/solve_steps`
- `initialize_tiles()`：batch 到 core 的分配
- `initialize_instructions()`：全流程指令生成
  - `CHOL_GRAM`
  - `CHOL_REG`
  - `CHOL_POTRF_DIAG_UPD_*`
  - `CHOL_TRSM_NUM_UPD_*`, `CHOL_TRSM_DIV_*`
  - `CHOL_RK_UPDATE_*`
  - `CHOL_FWD_DIAG_INV_*`, `CHOL_FWD_OFF_MAC_*`, `CHOL_FWD_OFF_UPD_*`
  - `CHOL_BWD_MAC_FULL`
  - `CHOL_OUT`

本报告可直接作为 Cholesky 求逆算子“流程导出文档”使用。
