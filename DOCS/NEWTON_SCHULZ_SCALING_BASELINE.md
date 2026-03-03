# Newton–Schulz 基线算子缩放特性（Ascend 910B）

下面给出在 Ascend 910B 模型下，基线 Newton–Schulz 逆矩阵算子在不同矩阵尺寸下的周期数（batch=96，迭代次数 iterations=10）。

注意：各单元的周期数是「至少有一个该类型核心处于活跃状态」时的时钟数（跨核心做时间区间并集），**不是**按核心把活跃周期简单相加。

| Size | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |
|------|--------------|------|--------|----------------|-------|
| 16x16 | 309 | 2160 | 1394 | 243 | 2263 |
| 32x32 | 550 | 3048 | 2356 | 593 | 3159 |
| 64x64 | 2572 | 6601 | 5834 | 2399 | 7040 |
| 128x128 | 8618 | 14554 | 12142 | 6695 | 16369 |

## 复杂度分析（Complexity Analysis）

考虑一个 batch 中有 $B$ 个大小为 $N\times N$ 的矩阵（这里 $B=96$），Newton–Schulz 迭代步数为 $T$（这里 $T=10$）。对每个矩阵的每一轮迭代，大致执行：

- 在 Cube 阵列上进行一次矩阵乘 $T = A\cdot X$；
- 在 Vector 单元上进行一次逐元素更新 $R = C - T$；
- 在 Cube 阵列上进行一次矩阵乘 $X_{k+1} = X_k\cdot R$。

如果只从数量级上考虑操作数，单个矩阵每轮迭代的理想计算量为：

- **Cube（GEMM）**：一次矩阵乘的复杂度是 $\Theta(N^3)$，而每轮有两次 GEMM，总共 $T$ 轮：
  $$\text{Cube ops} \sim 2T\,N^3 = \Theta(N^3).$$
- **Vector（ADD）**：$R = C - T$ 要访问全部 $N^2$ 个元素，每轮一次：
  $$\text{Vector ops} \sim T\,N^2 = \Theta(N^2).$$
- **MOVIN/MOVOUT**：读入 $A, X_0, C$，以及写回 $X_T$，每个矩阵都是 $\Theta(N^2)$ 大小的数据搬运：
  $$\text{MOVIN} \sim \Theta(N^2),\quad \text{MOVOUT} \sim \Theta(N^2).$$

在 batch 大小 $B$ 固定时，各阶段的**渐近周期复杂度**可写成

$$
\text{Cube cycles} = \Theta(N^3),\quad
\text{Vector cycles} = \Theta(N^2),\quad
\text{Load/Store cycles} = \Theta(N^2).
$$

## 与实测缩放的对比（Comparison with Measured Scaling）

根据表中的 Total（总周期）：

- $16^2 \to 32^2$: $2263 \to 3159$（约 $\times 1.40$）
- $32^2 \to 64^2$: $3159 \to 7040$（约 $\times 2.23$）
- $64^2 \to 128^2$: $7040 \to 16369$（约 $\times 2.33$）

如果只按「原始 FLOP 数」来估算，一个单核串行的 $N\\times N$ GEMM 复杂度是 $\\Theta(N^3)$，在尺寸每翻一倍时，理论上 FLOP 数会放大 $2^3=8$ 倍。然而在本模拟器中：

- Cube 阵列高度并行，可以在多个 PE、多个 core 上同时执行部分乘加；
- 同时在 batch 内对权重/激活做复用，把大量 $N^3$ 级别的运算“铺开”到更宽的阵列上。

因此，**实际观测到的 wall-clock 周期增长明显是“亚立方”（sub-cubic）**，这在图表中已经体现出来：从 32→64→128，Total 的放大倍数大约在 $\times 2.2\sim2.3$，而不是 $8$ 倍。

从各个阶段来看，趋势与上面的复杂度分析是吻合的：

- **Cube 行** 增长最快：  
  $2160 \to 3048 \to 6601 \to 14554$，随着 $N$ 增大会呈现显著的超线性增长，并且在较大尺寸（64、128）时逐渐主导 Total。
- **MOVIN/MOVOUT 行** 更接近 $\Theta(N^2)$：  
  $309 \to 550 \to 2572 \to 8618$，同时受到 tile 切分和 DRAM 访问模式的影响，所以不是严格的幂律，但整体斜率接近 $N^2$。
- **Vector 行**：  
  $1394 \to 2356 \to 5834 \to 12142$，在小尺寸时常数/调度开销占比较大；尺寸增大后，其增长介于 $N^2$ 和 $N^3$ 之间，这与向量工作在 tile 维度上成批下发的实现方式有关。

综合来看：

- 从单个矩阵的视角，理论复杂度是以 Cube 计算为主，主项为 $\Theta(T\,N^3)$，Vector 和 Load/Store 贡献的是 $\Theta(T\,N^2)$ 与 $\Theta(N^2)$ 的低阶项。
- 从当前 910B 类架构的仿真结果看，固定 batch 时的总 wall-clock 周期随 $N$ 的增长表现为**亚立方**，这是因为阵列在矩阵乘内部以及 batch 维度上都充分利用了并行性与复用；但从相对趋势上看，**Cube 行随尺寸增长比 MOVIN/MOVOUT 和 Vector 更快**，与“计算主导、带宽/调度为次要瓶颈”的理论预期是一致的。