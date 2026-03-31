# LDL 求逆：步骤公式、计算单元映射与事件数量校验（2026-03-26）

## 1. 目的

给出基于 `LDLDecompOp` 的 LDL 求逆流程中：

1. 每一步数学公式；
2. 每一步对应使用 `Cube` 还是 `Vector`；
3. 理论事件数量；
4. 与仿真 trace 实测数量的差异。

本报告基于当前修正后实现：

- 代码：`src/operations/LDLDecompOp.cc`
- 配置：`example/ldl_test.json`（`U=16, block_size=2, bwd_steps=1`）
- Trace：`results/LDL/falsification/ldl_64x16_trace_fix.csv`

---

## 2. 记号与参数

设：

$$
U=16,\quad blk=2,\quad B=U/blk=8,\quad s=bwd\_steps=1.
$$

打包参数（由代码自动推导）：

$$
P=cube\_pack\_blocks=8.
$$

---

## 3. 算法步骤与公式（含计算单元）

### 3.1 Gram + 正则

1. Gram：

$$
G = H^H H
$$

指令：`LDL_GRAM`，计算单元：**Cube**。

2. 正则：

$$
A = G + \lambda I
$$

指令：`LDL_REG`，计算单元：**Vector**（`ADD`）。

---

### 3.2 Block LDL 分解（按列块 $j=0..B-1$）

1. 对角更新：

$$
\tilde D_{jj}=A_{jj}-\sum_{k<j}L_{jk}D_{kk}L_{jk}^H
$$

指令：`LDL_D_UPDATE_j`，计算单元：**Cube**。

2. 对角逆：

$$
D_{jj}^{-1}
$$

指令：`LDL_D_INV_j`，计算单元：**Vector**（`DIV`）。

3. 下三角更新（带 pack）：

$$
L_{ij}=\left(A_{ij}-\sum_{k<j}L_{ik}D_{kk}L_{jk}^H\right)D_{jj}^{-1},\quad i>j
$$

指令：`LDL_L_UPDATE_*`。在当前实现中：

- 大包 `packed_dim>2`：**Cube**；
- 尾包 `packed_dim<=2`：**Vector**（`MAC`）。

---

### 3.3 Backward 逆组装（按列 $j=B-1..0$）

1. 对角块乘法项：

$$
X_{jj}=D_{jj}^{-1}-\sum_{k=j+1}^{B-1}L_{kj}^HX_{jk}^H
$$

乘法指令：`LDL_BWD_DIAG_MUL_*`，
累加指令：`LDL_BWD_DIAG_ACC_*`。

其中：

- `MUL`：长 `k` 走 **Cube**，最短 `k=2` 走 **Vector(MAC)**；
- `ACC`：**Vector(ADD)**。

2. 非对角块乘法项（$i<j$）：

$$
X_{ij}=-\sum_{k=i+1}^{B-1}L_{ki}^HX_{kj}
$$

乘法指令：`LDL_BWD_OFF_MUL_*`，
累加指令：`LDL_BWD_OFF_ACC_*`。

其中：

- `MUL`：长 `k` 走 **Cube**，最短 `k=2` 走 **Vector(MAC)**；
- `ACC`：**Vector(ADD)**。

---

### 3.4 Barrier / Memory

- `LDL_BARRIER_*`：**Vector**（1-cycle barrier）。
- `MOVIN` / `MOVOUT`：内存通路（非 Cube/Vector 计算指令）。

---

## 4. 理论事件数量推导

### 4.1 语义事件（不分计算单元）

每 batch：

- `GRAM`: $1$
- `REG_ADD`: $1$
- `D_UPDATE`: $B=8$
- `D_INV`: $B=8$
- `L_UPDATE`: $\sum_{j=0}^{B-1}\left\lceil\frac{B-j-1}{P}\right\rceil=7$
- `BWD_DIAG_MUL`: $s(B-1)=7$
- `BWD_DIAG_ACC`: $s(B-1)=7$
- `BWD_OFF_MUL`: $s\frac{B(B-1)}{2}=28$
- `BWD_OFF_ACC`: $s\frac{B(B-1)}{2}=28$
- `BARRIER`: $3+B+B+B+1=28$

乘以 batch=96 后：

- `GRAM=96`
- `REG_ADD=96`
- `D_UPDATE=768`
- `D_INV=768`
- `L_UPDATE=672`
- `BWD_DIAG_MUL=672`
- `BWD_DIAG_ACC=672`
- `BWD_OFF_MUL=2688`
- `BWD_OFF_ACC=2688`
- `BARRIER=2688`

### 4.2 Cube / Vector 拆分（考虑小算子 MAC 重定向）

当前阈值规则：`tile_m<=2 && tile_k<=2 && tile_n<=2` 走 `MAC(Vector)`。

据循环边界可得每 batch：

- `L_UPDATE`: 6 Cube + 1 Vector
- `BWD_DIAG_MUL`: 6 Cube + 1 Vector
- `BWD_OFF_MUL`: 27 Cube + 1 Vector

因此全 batch：

- `L_UPDATE`: 576 Cube + 96 Vector
- `BWD_DIAG_MUL`: 576 Cube + 96 Vector
- `BWD_OFF_MUL`: 2592 Cube + 96 Vector

再加上 `GRAM` 与 `D_UPDATE` 全部为 Cube：

$$
N_{cube}=96+768+576+576+2592=4608.
$$

---

## 5. 实测事件数量（trace 统计）

从 `ldl_64x16_trace_fix.csv` 统计得到：

- 语义事件总数：
  - `GRAM 96`
  - `REG_ADD 96`
  - `D_UPDATE 768`
  - `D_INV 768`
  - `L_UPDATE 672`
  - `BWD_DIAG_MUL 672`
  - `BWD_DIAG_ACC 672`
  - `BWD_OFF_MUL 2688`
  - `BWD_OFF_ACC 2688`
  - `BARRIER 2688`

- 按单元拆分：
  - `L_UPDATE`: `576 Cube + 96 Vector`
  - `BWD_DIAG_MUL`: `576 Cube + 96 Vector`
  - `BWD_OFF_MUL`: `2592 Cube + 96 Vector`
  - `D_UPDATE`: `768 Cube`
  - `GRAM`: `96 Cube`

并观测到调度等待事件：

- `CubeWait = 1316`（该项不属于 LDL 语义公式计数对象）。

---

## 6. 理论 vs 仿真差异表

| 类别 | 理论 | 实测 | 差异 |
|---|---:|---:|---:|
| GRAM | 96 | 96 | 0 |
| REG_ADD | 96 | 96 | 0 |
| D_UPDATE | 768 | 768 | 0 |
| D_INV | 768 | 768 | 0 |
| L_UPDATE | 672 | 672 | 0 |
| BWD_DIAG_MUL | 672 | 672 | 0 |
| BWD_DIAG_ACC | 672 | 672 | 0 |
| BWD_OFF_MUL | 2688 | 2688 | 0 |
| BWD_OFF_ACC | 2688 | 2688 | 0 |
| BARRIER | 2688 | 2688 | 0 |

按计算单元拆分也完全一致（关键项）：

- `L_UPDATE`: 理论 `576/96 (Cube/Vector)`，实测 `576/96`
- `BWD_DIAG_MUL`: 理论 `576/96`，实测 `576/96`
- `BWD_OFF_MUL`: 理论 `2592/96`，实测 `2592/96`
- `Cube总数`: 理论 `4608`，实测 `4608`

结论：**LDL 事件数量模型与当前实现、仿真 trace 完全闭合，差异为 0。**

---

## 7. 关于 `CubeWait` 的说明

`CubeWait` 由调度与发射时序引入，属于流水线等待事件，不是 LDL 算法语义步骤本身的“工作事件”。

因此：

- 语义事件（公式推导）与仿真事件可做一一对照并闭合；
- `CubeWait` 需单独作为“调度副产物”分析，不计入步骤公式主表。
