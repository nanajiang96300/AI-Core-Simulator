# LDL 不分块（标量）求逆分析与事件校验（2026-03-26）

## 1. 范围与结论

本报告只分析**不分块 LDL**（`block_size=1`），不使用 `2x2` 分块推导。

基于：

- 算子：`src/operations/LDLDecompNoBlockOp.cc`（内部强制 `block_size=1`, `pack_blocks=1`）
- 核心实现：`src/operations/LDLDecompOp.cc`
- 配置：`example/ldl_noblock_test.json`（`U=16, bwd_steps=1`）
- Trace：`results/LDL/falsification/ldl_noblock_64x16_trace.csv`

结论先行：

1. 标量 LDL 的步骤公式与当前算子循环一一对应；
2. 理论事件计数与仿真计数完全一致（主事件差异为 0）；
3. 基线实现中，乘法类步骤并非全部走 Vector：部分长 `k` 累加仍走 Cube。
4. 本次已新增 `fix2`（`D_UPDATE` 在 no-block 下改为动态 `k_len=max(1,j)` + `MAC/GEMM` 选路），Cube 占用与总周期均下降。

---

## 2. 标量 LDL 求逆流程（不分块）

记：

$$
A = H^H H + \lambda I,\quad U=16.
$$

### 2.1 Gram + 正则

$$
G = H^H H,\qquad A = G + \lambda I
$$

### 2.2 LDL 分解（标量）

对每个 $j=0..U-1$：

1. 对角项：

$$
d_{jj}=a_{jj}-\sum_{k=0}^{j-1} l_{jk} d_{kk} \overline{l_{jk}}
$$

2. 对角倒数：

$$
d_{jj}^{-1}
$$

3. 下三角项（$i>j$）：

$$
l_{ij}=\left(a_{ij}-\sum_{k=0}^{j-1} l_{ik} d_{kk} \overline{l_{jk}}\right)d_{jj}^{-1}
$$

### 2.3 逆矩阵后向组装

1. 对角：

$$
x_{jj}=d_{jj}^{-1}-\sum_{k=j+1}^{U-1}\overline{l_{kj}}\,\overline{x_{jk}}
$$

2. 非对角（$i<j$）：

$$
x_{ij}=-\sum_{k=i+1}^{U-1}\overline{l_{ki}}\,x_{kj}
$$

---

## 3. 每一步对应的计算单元（Cube / Vector）

> 说明：当前实现有一个小算子门限：当 `tile_m<=2 && tile_k<=2 && tile_n<=2` 时，乘法指令从 `GEMM_PRELOAD` 改走 `MAC(Vector)`。

1. `LDL_GRAM`：**Cube**
2. `LDL_REG`：**Vector（ADD）**
3. `LDL_D_UPDATE_j`：
   - 基线：**Cube**（固定 GEMM 建模）
   - `fix2`：`k_len=max(1,j)` 动态化，`k_len<=2` 走 **Vector（MAC）**，其余走 **Cube**
4. `LDL_D_INV_j`：**Vector（DIV）**
5. `LDL_L_UPDATE_i_j`：在 no-block 下 `tile=1x1x1`，**全部 Vector（MAC）**
6. `LDL_BWD_DIAG_MUL_j`：
   - `k_len>2`：**Cube**
   - `k_len<=2`：**Vector（MAC）**
7. `LDL_BWD_DIAG_ACC_j`：**Vector（ADD）**
8. `LDL_BWD_OFF_MUL_i_j`：
   - `k_len>2`：**Cube**
   - `k_len<=2`：**Vector（MAC）**
9. `LDL_BWD_OFF_ACC_i_j`：**Vector（ADD）**
10. `LDL_BARRIER_*`：**Vector（barrier/NOP）**

---

## 4. 理论事件数量（`U=16`, `bwd_steps=1`）

设 $n=U=16$。

### 4.1 语义事件（不区分单元）

每 batch：

- `GRAM`: $1$
- `REG_ADD`: $1$
- `D_UPDATE`: $n=16$
- `D_INV`: $n=16$
- `L_UPDATE`: $\sum_{j=0}^{n-1}(n-j-1)=\frac{n(n-1)}{2}=120$
- `BWD_DIAG_MUL`: $n-1=15$
- `BWD_DIAG_ACC`: $15$
- `BWD_OFF_MUL`: $\frac{n(n-1)}{2}=120$
- `BWD_OFF_ACC`: $120$
- `BARRIER`: $3+n+n+n+1=52$

总计每 batch：

$$
1+1+16+16+120+15+15+120+120+52=476
$$

总 batch 为 96，因此：

- `GRAM=96`
- `REG_ADD=96`
- `D_UPDATE=1536`
- `D_INV=1536`
- `L_UPDATE=11520`
- `BWD_DIAG_MUL=1440`
- `BWD_DIAG_ACC=1440`
- `BWD_OFF_MUL=11520`
- `BWD_OFF_ACC=11520`
- `BARRIER=4992`

### 4.2 按单元拆分（理论）

1. `L_UPDATE`：全部 `1x1x1`，每 batch `120 Vector`。

2. `BWD_DIAG_MUL`：每 batch共 15 条；其中 `k_len<=2` 出现 2 条（Vector），其余 13 条（Cube）。

3. `BWD_OFF_MUL`：每 batch共 120 条；其中 `k_len<=2` 出现 3 条（Vector），其余 117 条（Cube）。

因此每 batch：

- Cube 乘法类：`GRAM(1) + D_UPDATE(16) + DIAG_MUL(13) + OFF_MUL(117) = 147`
- Vector 乘法类：`L_UPDATE(120) + DIAG_MUL(2) + OFF_MUL(3) = 125`

全 batch：

- `Cube = 147 * 96 = 14112`
- `Vector(乘法类) = 125 * 96 = 12000`

---

## 5. 仿真实测数量（`ldl_noblock_64x16_trace.csv`）

实测统计：

- `GRAM 96`
- `REG_ADD 96`
- `D_UPDATE 1536`
- `D_INV 1536`
- `L_UPDATE 11520`
- `BWD_DIAG_MUL 1440`
- `BWD_DIAG_ACC 1440`
- `BWD_OFF_MUL 11520`
- `BWD_OFF_ACC 11520`
- `BARRIER 4992`

按单元：

- `D_UPDATE`: `1536 Cube`
- `GRAM`: `96 Cube`
- `L_UPDATE`: `11520 Vector`
- `BWD_DIAG_MUL`: `1248 Cube + 192 Vector`
- `BWD_OFF_MUL`: `11232 Cube + 288 Vector`

并观测：

- `CubeWait = 1260`（调度等待，不属于主语义事件）。

---

## 6. 理论 vs 实测差异

| 事件类别 | 理论 | 实测 | 差异 |
|---|---:|---:|---:|
| GRAM | 96 | 96 | 0 |
| REG_ADD | 96 | 96 | 0 |
| D_UPDATE | 1536 | 1536 | 0 |
| D_INV | 1536 | 1536 | 0 |
| L_UPDATE | 11520 | 11520 | 0 |
| BWD_DIAG_MUL | 1440 | 1440 | 0 |
| BWD_DIAG_ACC | 1440 | 1440 | 0 |
| BWD_OFF_MUL | 11520 | 11520 | 0 |
| BWD_OFF_ACC | 11520 | 11520 | 0 |
| BARRIER | 4992 | 4992 | 0 |

按单元拆分也闭合：

- Cube 总计：理论 `14112`，实测 `14112`
- `BWD_DIAG_MUL`: 理论 `1248/192 (Cube/Vector)`，实测一致
- `BWD_OFF_MUL`: 理论 `11232/288 (Cube/Vector)`，实测一致

结论：**不分块 LDL 的事件数量模型与仿真计数完全一致（主事件差异为 0）。**

---

## 7. 回答“为什么标量公式里还会有 Cube”（基线）

虽然算法公式是标量求和形式，但当前实现把其中一部分长 `k` 累加仍映射为 `GEMM_PRELOAD`（Cube）来建模吞吐；仅在非常小的 `tile` 下转为 `MAC(Vector)`。

因此“标量公式”与“硬件仿真指令类型”不是一一同构：

- 公式描述的是数学运算结构；
- 指令类型描述的是模拟执行单元选择。

---

## 8. `ldl_noblock` 修正（fix2）结果

本次针对 no-block 路径修改 `LDL_D_UPDATE_j`：

- 在 `blk==1` 时将累加长度改为 `k_len=max(1,j)`（更贴近标量求和区间）；
- 按 `pick_mul_opcode(...)` 选择单元：`k_len<=2` 用 `MAC(Vector)`，否则 `GEMM_PRELOAD(Cube)`；
- 同步动态更新 `compute_size/tile_k`。

对比：

- 基线 trace：`results/LDL/falsification/ldl_noblock_64x16_trace.csv`
- fix2 trace：`results/LDL/falsification/ldl_noblock_64x16_trace_fix2.csv`

关键统计（`U=16`, batch=96）：

| 指标 | 基线 | fix2 | 变化 |
|---|---:|---:|---:|
| `LDL_*` Cube 事件数 | 14112 | 13824 | -288 |
| `LDL_*` Cube 总时长 | 451872 | 442656 | -9216 |
| `LDL_*` Vector 事件数 | 31584 | 31872 | +288 |
| `LDL_*` Vector 总时长 | 36192 | 36480 | +288 |
| `CubeWait` 事件数 | 1260 | 1185 | -75 |
| `CubeWait` 总时长 | 32741 | 28462 | -4279 |
| `max_end_cycle` | 11841 | 11666 | -175 |

`D_UPDATE` 单元拆分变化：

- 基线：`1536 Cube + 0 Vector`
- fix2：`1248 Cube + 288 Vector`

结论：`fix2` 使 no-block LDL 的 `D_UPDATE` 更接近标量求和建模，减少了不必要的 Cube 使用；同时保持事件闭合并带来总周期下降。

---

## 9. `ldl_noblock` 修正（fix3）结果与时序图

在继续全量 Vector 化 no-block 路径时，曾出现长时间空转（DRAM `BW utilization 0%` 持续打印）。根因是：

- 误跳过了 `blk==1, j==0` 的 `LDL_D_UPDATE_0`；
- 导致 `LDL_D_INV_0` 读取 `addr_Ainv` 时无生产者，前端依赖检查无法放行，仿真进入无进展循环。

fix3 修复点：

- 恢复 `LDL_D_UPDATE_0` 生产（保证依赖闭合）；
- 保留 no-block 的乘法累加单元选择为 Vector（`MAC`）策略；
- 使用 `ONNXIM_MAX_CORE_CYCLES` 作为安全上限，避免异常场景无限空转。

产物：

- trace：`results/LDL/falsification/ldl_noblock_64x16_trace_fix3.csv`
- 时序图：`results/LDL/falsification/ldl_noblock_64x16_timeline_fix3.png`

三版对比（`U=16`, batch=96）：

| 指标 | baseline | fix2 | fix3 |
|---|---:|---:|---:|
| `LDL_*` Cube 事件数 | 14112 | 13824 | 96 |
| `LDL_*` Cube 总时长 | 451872 | 442656 | 3360 |
| `LDL_*` Vector 事件数 | 31584 | 31872 | 45600 |
| `LDL_*` Vector 总时长 | 36192 | 36480 | 50208 |
| `CubeWait` 事件数 | 1260 | 1185 | 93 |
| `CubeWait` 总时长 | 32741 | 28462 | 2944 |
| `max_end_cycle` | 11841 | 11666 | 4162 |

`D_UPDATE` 单元拆分：

- baseline：`1536 Cube + 0 Vector`
- fix2：`1248 Cube + 288 Vector`
- fix3：`0 Cube + 1536 Vector`

结论：fix3 在保证依赖正确性的前提下，基本消除了 no-block 路径中的 Cube 误建模，时序图中长尾等待显著收敛，总周期进一步下降。
