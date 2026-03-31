# LDL 分块前后：算法-事件-单元-周期对应报告（2026-03-27）

## 1. 结论摘要

- 你要的“分块前后对应表”已按 **公式 -> 事件 -> 计算单元 -> 周期模型 -> 实测数据** 给出。  
- 关键提速发生在 `LDL_D_UPDATE / LDL_BWD_DIAG_MUL / LDL_BWD_OFF_MUL`：`block_old -> block_opt2` 分别约 `32.0x / 27.57x / 30.89x`。  
- “公式里矩阵乘看起来不多，但仿真占比很大”的主因是：**Cube 单事件成本高（~28~35 cycles）+ 依赖造成 Wait**，而不是事件数量本身。

---

## 2. 周期模型（与代码一致）

### 2.1 Cube（`GEMM_PRELOAD`）

来自 `src/SystolicWS.cc::get_inst_compute_cycles`：

$$
C_{cube}=base\_latency+(cube_m+cube_n-2)+\max(1, B_mB_nB_k)
$$

其中

$$
B_m=\left\lceil\frac{tile_m}{cube_m}\right\rceil,
B_n=\left\lceil\frac{tile_n}{cube_n}\right\rceil,
B_k=\left\lceil\frac{tile_k}{cube_k}\right\rceil
$$

在当前配置（`cube_m=n=k=16`, `base=1`）下，若 `tile=(2,2,16)`：

$$
C_{cube}=1+(16+16-2)+1=32
$$

### 2.2 Vector（`MAC/ADD/MUL/DIV`）

来自 `src/SystolicWS.cc::get_vector_compute_cycles`：

$$
C_{vec}(op)=\left\lceil\frac{compute\_size}{V}\right\rceil\cdot latency(op),\quad V=\frac{vector\_process\_bit}{8}
$$

当前 `vector_process_bit=2048`，故 `V=256`；本实验里相关 `compute_size` 很小，通常只需 1 次迭代，
所以 `MAC/ADD/MUL` 常见为 1 cycle，`DIV` 常见为 4 cycles。

---

## 3. 算法步骤 -> 事件 -> 单元 对照表

| 算法公式步骤 | 事件前缀 | 主要 opcode | 主要计算单元 | 周期模型 |
|---|---|---|---|---|
| $G=H^HH$ | `LDL_GRAM` | `GEMM_PRELOAD` | Cube | $C_{cube}$ |
| $A=G+\lambda I$ | `LDL_REG` | `ADD` | Vector | $C_{vec}(ADD)$ |
| $D_j=A_{jj}-\sum_{k<j}L_{jk}D_kL_{jk}^H$ | `LDL_D_UPDATE` | old: `GEMM_PRELOAD`/opt2: `MAC` | old: Cube / opt2: Vector | old: $C_{cube}$ / opt2: $C_{vec}(MAC)$ |
| $D_j^{-1}$ | `LDL_D_DIAG_INV` | `DIV` | Vector | $C_{vec}(DIV)$ |
| 对角逆后处理 | `LDL_D_INV_MUL` | `MUL` | Vector | $C_{vec}(MUL)$ |
| $L_{ij}=\left(A_{ij}-\sum_{k<j}L_{ik}D_kL_{jk}^H\right)D_j^{-1}$（pack） | `LDL_L_UPDATE` | `GEMM_PRELOAD`/`MAC` | 以 Cube 为主 | $C_{cube}$ 或 $C_{vec}(MAC)$ |
| $X_{jj}=D_j^{-1}-\sum_{k>j}L_{kj}^HX_{jk}^H$ | `LDL_BWD_DIAG_MUL` + `LDL_BWD_DIAG_ACC` | old: `GEMM_PRELOAD`/opt2:`MAC` + `ADD` | old: Cube / opt2: Vector | old: $C_{cube}$ / opt2: $C_{vec}(MAC)$ |
| $X_{ij}=-\sum_{k>i}L_{ki}^HX_{kj}$ | `LDL_BWD_OFF_MUL` + `LDL_BWD_OFF_ACC` | old: `GEMM_PRELOAD`/opt2:`MAC` + `ADD` | old: Cube / opt2: Vector | old: $C_{cube}$ / opt2: $C_{vec}(MAC)$ |
| 阶段同步 | `LDL_BARRIER_*` | `PIPE_BARRIER` | Vector | 1 cycle |

---

## 4. 分块前后（含优化前后）总览

数据来源：

- no-block：`results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv`
- block-old：`results/LDL/falsification/ldl_block_64x16_trace.csv`
- block-opt2：`results/LDL/falsification/ldl_block_64x16_trace_opt2.csv`

| Case | events | max_end | total_dur | Cube dur | Vector dur | Wait dur |
|---|---:|---:|---:|---:|---:|---:|
| no-block | 51933 | 4290 | 273400 | 3360 | 51744 | 2944 |
| block-old | 18497 | 3704 | 424972 | 147744 | 10272 | 20903 |
| block-opt2 | 17382 | 1568 | 247399 | 21792 | 14208 | 4991 |

关键点：

- `block-old` 虽然总事件少于 no-block，但 `Cube dur` 和 `Wait dur` 显著上升，拖慢整体。  
- `block-opt2` 把小块乘法迁到 Vector 后，`Cube dur` 与 `Wait dur` 大幅下降，`max_end` 降到 `1568`。

---

## 5. 哪些步骤在分块后（优化后）得到了提升

对比 `block-old -> block-opt2`：

| 步骤 | old cnt/dur | opt2 cnt/dur | 平均每事件周期(old->opt2) | 提升倍数 |
|---|---:|---:|---:|---:|
| `LDL_D_UPDATE` | `768 / 24576` | `768 / 768` | `32.0 -> 1.0` | **32.00x** |
| `LDL_BWD_DIAG_MUL` | `672 / 18528` | `672 / 672` | `27.57 -> 1.0` | **27.57x** |
| `LDL_BWD_OFF_MUL` | `2688 / 83040` | `2688 / 2688` | `30.89 -> 1.0` | **30.89x** |
| `CubeWait` | `1313 / 20903` | `198 / 4991` | `15.92 -> 25.21` | **4.19x（总时长）** |
| `LDL_L_UPDATE` | `672 / 18528` | `672 / 18528` | `27.57 -> 27.57` | 1.00x |

解释：

- 本次优化**故意保留** `L_UPDATE` 的 pack+Cube 路径（大颗粒仍适合 Cube）；
- 优化重点是把“`2x2` 小块长K微更新”从 Cube 改到 Vector。

---

## 6. 为什么“矩阵乘不多，但仿真占比很大”

这是你提问的核心，结论如下：

1. **看事件数会低估代价**  
   - 例如 `block-old` 的 `LDL_D_UPDATE` 只有 `768` 个事件，但每个约 `32` 周期（Cube），总计 `24576` 周期。

2. **Cube 单事件成本由公式固定项主导**  
   - 在小 tile 下，`base + fill/drain`（约 `31`）几乎是刚性成本，
   - 即使公式里的“乘法块数”不大，单事件仍不便宜。

3. **依赖链引入额外等待（Wait）**  
   - 小块任务上 Cube 时，队列和依赖更容易产生 `CubeWait`，
   - `block-old` 的 `Wait dur=20903`，对关键路径影响明显。

4. **分块后若映射不当，会出现“事件少但更慢”**  
   - `no-block` 下很多乘法是 Vector 1-cycle；
   - `block-old` 把关键微更新放到 Cube，导致“条目少但每条更重 + 等待更长”。

5. **opt2 证明了这个判断**  
   - 把 `D_UPDATE/BWD_*_MUL` 改回 Vector 后，
   - 这些步骤平均周期从 `~28~32` 直接回到 `1`，`max_end` 从 `3704` 降到 `1568`。

---

## 7. 你可直接引用的结论语句

> 在 LDL 的 block 场景中，性能瓶颈不在“矩阵乘事件数量”本身，而在“微小矩阵乘被映射到 Cube 后的单事件固定成本与等待开销”。
> 当 `2x2` 微更新采用 Vector 路径后，关键步骤平均周期从 `~30` 降至 `1`，从而显著缩短关键路径。

---

## 8. 把 Cholesky 新结果放入同口径对照

数据来源：

- Cholesky old：`results/CHOL/falsification/cholesky_block_64x16_trace.csv`
- Cholesky opt：`results/CHOL/falsification/cholesky_block_64x16_trace_opt.csv`

### 8.1 Cholesky 自身前后对比（block）

| Case | events | max_end | total_dur | Cube dur | Vector dur | Wait dur |
|---|---:|---:|---:|---:|---:|---:|
| CHOL block old | 48614 | 13697 | 3185924 | 608544 | 29568 | 2289266 |
| CHOL block opt | 32637 | 3404 | 275652 | 6432 | 48384 | 5927 |

结论：

- Cholesky 经过同类“小块不走 Cube”优化后，`max_end` 从 `13697` 降至 `3404`（约 `4.02x`）。
- `Cube dur` 与 `Wait dur` 同步大幅下降，说明主要收益来自“微更新路由修正”。

### 8.2 与 LDL block-opt2 的并排结果

| Case | max_end | total_dur | Cube dur | Wait dur |
|---|---:|---:|---:|---:|
| LDL block opt2 | 1568 | 247399 | 21792 | 4991 |
| CHOL block opt | 3404 | 275652 | 6432 | 5927 |

结论：

- 当前 `64x16, block=2` 配置下，`LDL block opt2` 的关键路径更短（`1568 < 3404`）。
- CHOL 的 Cube 占比更低，但它在 Vector 路径上的步骤链更长（事件更多），整体 `max_end` 仍高于 LDL-opt2。

---

## 9. 你问的公式到底对应 LDL 哪些操作

你指出的公式：

$$
L_{ij}=\left(A_{ij}-\sum_{k<j}L_{ik}D_kL_{jk}^{H}\right)D_j^{-1}
$$

在当前 LDL 事件映射中，主要对应如下：

1. **乘法累加项（pack 版）**：`LDL_L_UPDATE_*`  
   - 对应上式中从 $A_{ij}-\sum_{k<j}(\cdot)$ 到右乘 $D_j^{-1}$ 的打包实现；
   - 在 block 模式里，这一步是 `L_UPDATE` 的主事件前缀。

2. **与对角块相关的前置步骤**：`LDL_D_UPDATE_*` + `LDL_D_DIAG_INV_*` + `LDL_D_INV_MUL_*`  
   - 这些步骤先构造并得到 $D_j^{-1}$；
   - 之后 `LDL_L_UPDATE_*` 才能完成上式最后的右乘项。

3. **为什么看起来“主要在这个公式上”**  
   - 在 `block_old` 里，`L_UPDATE` + `BWD_*_MUL` + `D_UPDATE` 三类都存在大量小块乘法；
   - 其中只有 `L_UPDATE` 保持 pack+Cube（设计上保留），其余已被优化到 Vector。

所以：

- **Cube 占比并不只来自这个公式**，还包括 `D_UPDATE`、`BWD_DIAG_MUL`、`BWD_OFF_MUL` 的旧路由；
- 优化后，Cube 主要集中在 `L_UPDATE`（和 `GRAM`），与你看到的数据一致。

---

## 10. 为什么 `LDL_D_UPDATE` 能提升 32 倍

对应表项：

`LDL_D_UPDATE 768/24576 -> 768/768 (32.0 -> 1.0, 32.00x)`

### 10.1 直接原因（主因）

`LDL_D_UPDATE` 的 opcode 从 **Cube** 改成了 **Vector MAC**：

- old：`GEMM_PRELOAD`，在当前 tile 下单事件约 `32` cycles；
- opt2：`MAC`，`compute_size` 很小，向量迭代通常 1 次，单事件约 `1` cycle。

因此每个事件从 `32 -> 1`，恰好对应 `32x`。

### 10.2 代码上具体优化位置

文件：`src/operations/LDLDecompOp.cc`

1. 新增 `pick_ldl_micro_mul_opcode(...)`：
   - 条件 `blk<=2 && tile_m<=2 && tile_n<=2` 时强制 `MAC`。

2. `LDL_D_UPDATE` 改用 `pick_ldl_micro_mul_opcode(...)` 选路。

3. 同时把 `d_update_k_len` 从 old 的固定 `U` 改成：
   - `d_update_k_len = max(blk, j*blk)`。

### 10.3 次要原因（辅助）

`d_update_k_len` 的修正在本配置下对单事件仍多为 1 次向量迭代，
所以对“32x”本身贡献较小；但它对更大规模/不同参数下的建模一致性和可扩展性更重要。

