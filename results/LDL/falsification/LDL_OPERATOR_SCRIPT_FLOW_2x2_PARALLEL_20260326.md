# LDL 详细流程与公式提取报告（算子 + 脚本，含 2x2 分块与并行）

## 1. 目标与范围

本报告将两个实现视角统一起来：

- 仿真算子实现：`src/operations/LDLDecompOp.cc`
- 算法质量脚本：`scripts/evaluate_ldl_quality.py`

并聚焦以下问题：

1. `2x2` 分块 LDL 的数学流程；
2. 该流程在算子指令级如何展开；
3. 并行是如何发生的（batch 级、块级 pack、流水级）；
4. 如何从循环直接推导出事件数量与关键复杂度。

---

## 2. 记号与参数

记：

$$
A = H^H H + \lambda I \in \mathbb{C}^{U\times U},\quad blk=2,
$$

$$
B = U/blk \quad (\text{块数}),\quad s=\texttt{bwd\_steps},\quad P=\texttt{cube\_pack\_blocks}.
$$

在 `example/ldl_test.json` 常用设置中：

$$
U=16,\ blk=2,\ B=8,\ s=1.
$$

---

## 3. 脚本侧算法流程（`evaluate_ldl_quality.py`）

### 3.1 2x2 分块 LDL 分解

对应函数：`block_ldl_decompose(a, cfg, block_size=2)`。

把 $A$ 按 `2x2` 划分为 $B\times B$ 个块，记块为 $A_{ij}\in\mathbb{C}^{2\times2}$。目标是：

$$
A = L D L^H,
$$

其中 $L$ 是单位下三角块矩阵，$D$ 是块对角矩阵。

按列块 $j=0\dots B-1$：

1. 对角块更新

$$
D_{jj}=A_{jj}-\sum_{k=0}^{j-1}L_{jk}D_{kk}L_{jk}^H
$$

2. 对角块求逆（2x2）

$$
D_{jj}^{-1}=\texttt{inv\_2x2\_complex}(D_{jj})
$$

3. 下三角块更新（$i>j$）

$$
L_{ij}=\left(A_{ij}-\sum_{k=0}^{j-1}L_{ik}D_{kk}L_{jk}^H\right)D_{jj}^{-1}
$$

实现细节：所有乘加都经 `qmatmul + quantize_complex`，因此每一步均带数值量化路径。

### 3.2 2x2 逆核公式

对应函数：`inv_2x2_complex(block, cfg)`，块

$$
\begin{bmatrix}
a & b \\
b^* & d
\end{bmatrix}
$$

先算行列式（按脚本是实部通道）：

$$
\det = \Re(a d - b b^*)
$$

再取倒数（`exact` 或近似倒数），最后

$$
\begin{bmatrix}
a & b \\
b^* & d
\end{bmatrix}^{-1}
=\frac{1}{\det}
\begin{bmatrix}
d & -b \\
-b^* & a
\end{bmatrix}.
$$

### 3.3 逆矩阵装配（后向块回代）

对应函数：`ldl_inverse(a, cfg, block_size=2)`。

核心是“从右到左”列回代，不显式构造 $L^{-1}$：

1. 对角块 $X_{jj}$：

$$
X_{jj}=D_{jj}^{-1}-\sum_{k=j+1}^{B-1}L_{kj}^H X_{jk}^H
$$

2. 非对角块 $X_{ij},\ i<j$：

$$
X_{ij}=-\sum_{k=i+1}^{B-1}L_{ki}^H X_{kj},\quad X_{ji}=X_{ij}^H
$$

最终 $A^{-1}=X$。

---

## 4. 算子侧指令流程（`LDLDecompOp.cc`）

### 4.1 批次并行（跨 core）

对应 `initialize_tiles`：

$$
\texttt{assigned\_core} = b \bmod N_{core}
$$

即 batch 轮转分发到各核，天然并行维度是 batch 级。

### 4.2 阶段划分

`initialize_instructions` 的阶段顺序：

1. `MOVIN`：加载 `H` 与 `RegI`；
2. `GEMM_PRELOAD(LDL_GRAM)`：$G=H^H H$；
3. `ADD(LDL_REG)`：$A=G+\lambda I$；
4. 块 LDL 主循环（按 $j$）：
   - `LDL_D_UPDATE_j`（对角更新建模）
   - `LDL_D_INV_j`（2x2 逆路径建模为 `DIV`）
   - `LDL_L_UPDATE_i_j_PACKp`（下方多块更新，支持 pack）
5. backward 主循环（按列 `j` 反向）：
   - `LDL_BWD_DIAG_MUL/ACC`
   - `LDL_BWD_OFF_MUL/ACC`
6. `MOVOUT(LDL_OUT)` 输出 `A_inv`。

阶段间通过 `PIPE_BARRIER` 保证依赖。

### 4.3 2x2 分块到指令参数的映射

当 `blk=2` 时：

- 每个对角块逆对应一次 `DIV`，`compute_size=blk*blk=4`；
- 对角/非对角回代中的单块输出都以 `tile_m=blk, tile_n=blk` 组织；
- 若把多个块 pack，`packed_dim = blk * packed_blocks`，对应更大的 GEMM 形状（同一条指令处理多个连续块）。

### 4.4 pack 并行（块级）

`cube_pack_blocks` 计算：

$$
P =
\begin{cases}
\texttt{pack\_blocks}, & \texttt{pack\_blocks}>0 \\
\max(1,\lfloor cube\_dim\_target/blk \rfloor), & \text{否则}
\end{cases}
$$

然后在 `for (i=j+1; i<n_blocks; i+=P)` 中按组发 `L_UPDATE`。

这表示：同一列 `j` 下方多个 `L_{ij}` 更新块被合并建模，减少指令数并提升吞吐。

---

## 5. 循环级公式提取（算子精确计数）

令 $B=U/blk$。

### 5.1 BLDL 阶段

每个 batch：

1. `D_UPDATE` 数量：

$$
N_{D\_UPDATE}=B
$$

2. `D_INV` 数量：

$$
N_{D\_INV}=B
$$

3. `L_UPDATE` 数量（pack 后）：

$$
N_{L\_UPDATE}=\sum_{j=0}^{B-1}\left\lceil\frac{B-j-1}{P}\right\rceil
$$

### 5.2 Backward 阶段

1. 对角乘加（仅当 `j < B-1` 有效）：

$$
N_{BWD\_DIAG\_MUL}=s(B-1),\quad
N_{BWD\_DIAG\_ACC}=s(B-1)
$$

2. 非对角乘加：

$$
N_{BWD\_OFF\_MUL}=s\sum_{j=0}^{B-1}j=s\frac{B(B-1)}{2},
$$

$$
N_{BWD\_OFF\_ACC}=s\frac{B(B-1)}{2}
$$

### 5.3 屏障数量

每 batch：

$$
N_{barrier}=3 + B + B + B + 1 = 3B+4
$$

分别对应：`LOAD2GRAM`,`GRAM2REG`,`REG2BLDL`，每列 BLDL step、每列 DIAG2OFF、每列 BWD_COL，以及尾部 `BWD2STORE`。

### 5.4 代入 `U=16, blk=2, B=8, s=1, P=8`

$$
N_{D\_UPDATE}=8,\ N_{D\_INV}=8,
$$

$$
N_{L\_UPDATE}=\sum_{j=0}^{7}\left\lceil\frac{7-j}{8}\right\rceil=7,
$$

$$
N_{BWD\_DIAG\_MUL}=7,\ N_{BWD\_DIAG\_ACC}=7,
$$

$$
N_{BWD\_OFF\_MUL}=28,\ N_{BWD\_OFF\_ACC}=28,
$$

$$
N_{barrier}=28.
$$

与已生成 falsification 报告中的实测计数口径一致。

---

## 6. 关键代码行含义说明（你关注的 `off_k_blocks`）

代码：

```cpp
const uint32_t off_k_blocks = (n_blocks > (iu + 1)) ? (n_blocks - (iu + 1)) : 0;
```

含义：对固定 $(i,j)$（且 $i<j$），非对角回代求和项的 $k$ 从 $i+1$ 到 $B-1$，因此项数是：

$$
\#k = B-(i+1).
$$

这与脚本中的：

```python
for k in range(i + 1, n_blocks):
```

完全同构。随后算子将该求和“折叠”为一个长 `tile_k` 的 GEMM（`off_k_len = off_k_blocks * blk`），这是“循环等价、实现重排”。

---

## 7. 并行性分解总结

### 7.1 已实现并行

1. **Batch 级并行**：`batch -> core` 轮转分配；
2. **块组并行（pack）**：同一 `j` 的多个 `i` 块合并；
3. **流水并行**：Load / Compute / Store 经队列与 barrier 管控。

### 7.2 依赖限制（不可随意并行）

1. `j` 方向的 LDL 主循环存在前后依赖；
2. backward 中每列 `j` 必须先完成 `x_jj`，再做该列 `x_ij`；
3. barrier 明确了 RAW 次序，因此并行粒度主要体现在“同阶段可独立批次、同列可 pack 子块”。

---

## 8. 脚本与算子的一致性映射

1. **分块粒度一致**：均以 `block_size=2` 为核心块粒度；
2. **对角逆一致**：脚本 `inv_2x2_complex`，算子 `DIV` 表达同一语义路径；
3. **回代求和区间一致**：脚本 `k in [i+1, B)`，算子 `off_k_blocks=B-(i+1)`；
4. **实现形态差异**：脚本是直接数学循环；算子是指令化、pack 化、屏障化后的等价重排。

---

## 9. 结论

从 `LDLDecompOp.cc` 与 `evaluate_ldl_quality.py` 提取可得：

1. 当前 LDL 路径确实是 `2x2` 分块 LDL + 后向块回代；
2. 关键循环边界（尤其 `off_k_blocks`）与脚本数学定义一一对应；
3. 并行主要来自 batch 级分核与 `pack_blocks` 块组合并；
4. 计数公式可直接由循环给出，并与现有 falsification 实测闭合。
