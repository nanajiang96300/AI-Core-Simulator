# LDL vs Cholesky 时序对比报告（2026-03-09）

## 1. 研究目标

在相同硬件配置与相同问题规模下，对比两类矩阵求逆实现路径的时序效率：

- 新算子：`LDL` 路径（含 `block_size=2` 的 2x2 批拼接优化）
- 基线算子：`Cholesky` 路径

目标是回答：在当前 NPU 风格指令级仿真模型中，`LDL` 相对 `Cholesky` 是否带来显著时延收益，以及收益来自哪些算法/实现层面的差异。

## 2. 对比口径

- 硬件配置：`configs/ascend_910b_quiet.json`（24 核）
- 维度配置：`M=64, U=16, batch=96`
- 新算子：`ldl_test`（含 2x2 -> 16x16 的批拼接优化）
- 基线算子：`cholesky_test`
- 数据来源：
  - `ldl_new/trace_c24.csv`
  - `cholesky_baseline/trace_c24.csv`

两条实验路径保持：

- 相同的 `M/U/batch`；
- 相同的核心数量与时钟配置；
- 相同 trace 导出方式（`ONNXIM_TRACE_CSV`）；
- 相同最大周期保护（`ONNXIM_MAX_CORE_CYCLES`）。

## 3. 算法推导与实现映射

### 3.1 问题定义

对正定（或经正则化后近似正定）矩阵

$$
A = H^H H + \lambda I
$$

求逆矩阵 $A^{-1}$，并比较不同分解路径在仿真器中的指令时序成本。

### 3.2 Cholesky 基线路径推导

设按块大小 $b$（本实验中 $b=2$）对矩阵分块，记第 $j$ 列块为：

$$
A_{ij}\in\mathbb{C}^{b\times b},\quad i,j=1,\dots,n_b,\quad n_b=U/b.
$$

Cholesky 分解写作：

$$
A = L L^H,
$$

其中 $L$ 为下三角。于是：

$$
A^{-1} = (L^H)^{-1}L^{-1}.
$$

分块 Cholesky 在第 $j$ 步的经典更新为：

1) 对角块（POTRF 前的 Schur 补）：

$$
  ilde A_{jj}=A_{jj}-\sum_{k=1}^{j-1}L_{jk}L_{jk}^H,
$$

再做 Cholesky：

$$
L_{jj}=\operatorname{chol}(\tilde A_{jj}).
$$

2) 列块三角解（TRSM）：

$$
L_{ij}=\Big(A_{ij}-\sum_{k=1}^{j-1}L_{ik}L_{jk}^H\Big)(L_{jj}^{-H}),\quad i>j.
$$

3) trailing 子矩阵更新（RK/SYRK）：

$$
A_{ik}\leftarrow A_{ik}-L_{ij}L_{kj}^H,\quad i,k>j.
$$

最后求逆通常通过两次三角求解实现：先求 $Y=L^{-1}$，再 $A^{-1}=Y^H Y$，或等价地逐列解线性系统。

计算流程在 NPU 上可对应为：

- **POTRF_GEMM**：Cube 执行块内乘累加，Vector 执行对角归一/开方近似与缩放；
- **TRSM**：Cube/Vector 组合，Cube 做块乘，Vector 做除法和系数更新；
- **RK_UPDATE**：大量小块 GEMM，主要消耗 Cube 发射条目与调度窗口；
- **SOLVE\_***：Vector 为主（`MUL/ADD/DIV`），并通过 barrier 保证列间依赖。

在本项目当前 Cholesky 基线实现中，`RK_UPDATE` 的小块更新链是事件数上升和关键路径拉长的主要来源（尤其当 $b=2$ 时微指令密度较高）。

### 3.3 LDL 新算子路径推导

LDL 分解写作：

$$
A = L D L^H,
$$

其中 $L$ 为单位下三角，$D$ 为块对角（本实现为小块）。于是：

$$
A^{-1} = (L^H)^{-1} D^{-1} L^{-1}.
$$

分块 LDL 在第 $j$ 步可写为：

1) 对角块更新：

$$
D_{jj}=A_{jj}-\sum_{k=1}^{j-1}L_{jk}D_{kk}L_{jk}^H.
$$

2) 非对角块更新（$i>j$）：

$$
L_{ij}=\Big(A_{ij}-\sum_{k=1}^{j-1}L_{ik}D_{kk}L_{jk}^H\Big)D_{jj}^{-1}.
$$

3) 逆矩阵组装：

$$
A^{-1}=L^{-H}D^{-1}L^{-1}.
$$

本实现采用“按列从右向左”的后向组装，等价于在分块层面解：

$$
L^H X = D^{-1}L^{-1},\quad X=A^{-1}.
$$

对比 Cholesky，LDL 路径避免显式平方根，核心开销集中在：

1. 块分解（`D_UPDATE`, `D_INV`, `L_UPDATE`）
2. 后向组装（`BWD_*`）

在 NPU 映射上：

- `D_UPDATE`：Cube 做块乘累加（对应式中求和项）；
- `D_INV`：Vector 执行小块逆/除法近似（`DIV`）；
- `L_UPDATE`：Cube 处理块乘；
- `BWD_*`：Vector 管线执行依赖敏感的 `MUL/ADD` 列回代，列间使用 barrier 保序。

#### 2x2 拼接优化的严格可行性证明

设有 $m$ 组互不耦合的子问题（本节以乘法为例，更新式同理）：

$$
C^{(t)} = A^{(t)}B^{(t)},\quad A^{(t)},B^{(t)},C^{(t)}\in\mathbb{C}^{2\times2},\ t=1,\dots,m.
$$

##### (1) 块对角构造与主命题

定义块对角拼接：

$$
\bar A = \operatorname{diag}(A^{(1)},\dots,A^{(m)}),\quad
\bar B = \operatorname{diag}(B^{(1)},\dots,B^{(m)}).
$$

命题：

$$
\bar C = \bar A\bar B = \operatorname{diag}(A^{(1)}B^{(1)},\dots,A^{(m)}B^{(m)}).
$$

证明：块矩阵乘法的 $(i,j)$ 块为

$$
(\bar A\bar B)_{ij}=\sum_{k=1}^{m}\bar A_{ik}\bar B_{kj}.
$$

由于 $\bar A,\bar B$ 仅在对角块非零：

- 当 $i\neq j$ 时，所有乘积项均为零，故 $(\bar A\bar B)_{ij}=0$；
- 当 $i=j$ 时，仅 $k=i$ 项非零，得 $(\bar A\bar B)_{ii}=A^{(i)}B^{(i)}$。

故命题成立。

##### (2) 与“逐块独立执行”的等价性

逐块执行得到集合 $\{C^{(t)}\}_{t=1}^m$；拼接执行得到 $\bar C$。二者关系是：

$$
C^{(t)} = \bar C_{tt},\quad t=1,\dots,m.
$$

即拼接结果按块切分后与逐块结果完全一致，属于严格数学等价，而非近似。

##### (3) 置换不变性（支持任意布局打包）

实际实现中，子块在内存中不一定天然按块对角排列。令 $P$ 为置换矩阵（$P^{-1}=P^T$），把“物理布局”重排到“逻辑块对角布局”：

$$
\hat A = P\bar A P^T,\quad \hat B = P\bar B P^T.
$$

则

$$
\hat C = \hat A\hat B = P\bar A P^T P\bar B P^T = P(\bar A\bar B)P^T.
$$

因此只要 pack/unpack 是双射重排（置换），计算语义保持不变。工程上等价于：

1. `pack`：按置换把多个 $2\times2$ 搬成大块布局；
2. `GEMM`：一次大矩阵乘；
3. `unpack`：逆置换写回各子块位置。

##### (4) 向量化视角（与硬件执行兼容）

利用恒等式 $\operatorname{vec}(AXB)=(B^T\otimes A)\operatorname{vec}(X)$，对块对角情形有：

$$
\operatorname{vec}(\bar C)=\Big(\operatorname{diag}(B^{(1)},\dots,B^{(m)})^T\otimes I\Big)\operatorname{vec}(\bar A),
$$

其系数矩阵依然呈块分离结构，不会引入不同子问题之间的交叉项，说明并行拼接在代数上可分离。

##### (5) 在 NPU 上的可实现性条件

拼接可行且收益成立的前提：

1. **子任务独立**：不同 $t$ 之间无数据依赖（本场景满足）；
2. **布局可逆**：pack/unpack 为可逆映射（置换）；
3. **数值规则一致**：拼接与非拼接使用同一精度/舍入模型；
4. **资源可容纳**：拼接后块尺寸不超过当前核可高效处理的 tile 范围。

在本项目配置下，取 $m=8$ 时：

$$
2m = 16,
$$

即将 `8` 个 `2x2` 子问题拼成一个 `16x16` 任务，和 `16x16x16` Cube 粒度匹配，从而减少微小 GEMM 发射条数与调度气泡。

### 3.4 从推导到仿真指令的对应

- Cholesky：`POTRF_GEMM/TRSM/RK_UPDATE/SOLVE_*`
- LDL：`D_UPDATE/D_INV/L_UPDATE/BWD_*`

虽然两者最终目标一致（构造 $A^{-1}$），但在当前 NPU 指令模型里：

- Cholesky 更依赖高频的 trailing 更新（`RK_UPDATE`）与小块 TRSM 链；
- LDL 更偏向 `D/L` 更新 + 回代组装，且 `L_UPDATE` 可批拼接。

这使得 LDL 在相同维度下通常表现为更低的 Cube 事件条数与更短的关键路径跨度。

## 4. 主要结论

- LDL 相比 Cholesky 在本配置下实现了**约 71.94%** 的总周期降低（可表述为“约 70%+ 提升”）。
- 提升核心来自：
  1) LDL 指令流减少了 Cholesky 中高成本的 `RK_UPDATE` 类更新路径；
  2) 对 `block_size=2` 的 `L_UPDATE` 做了批拼接，提升了 Cube 发射颗粒度与利用率；
  3) Cube 指令条数显著下降，调度与发射开销减少。

## 5. 量化结果（24 核全量）

| 指标 | LDL New | Cholesky Baseline | 变化 |
|---|---:|---:|---:|
| 总事件数 | 16,416 | 32,448 | -49.41% |
| 总周期跨度（span） | 3,844 | 13,697 | **-71.94%** |
| Cube 事件数 | 4,896 | 19,008 | **-74.24%** |
| Vector 事件数 | 6,912 | 8,832 | -21.74% |
| MTE2 事件数 | 3,840 | 3,840 | 0 |
| MTE3 事件数 | 768 | 768 | 0 |
| Cube 平均持续周期 | 32.06 | 32.02 | +0.04 |
| Vector 平均持续周期 | 1.33 | 3.35 | -2.01 |

> 计算方式：
> - 提升比例 = $(13697 - 3844) / 13697 \approx 71.94\%$

## 6. Core0/1 视图补充

- `ldl_new/trace_core01.csv`：`events=1368`, `span=3755`
- `cholesky_baseline/trace_core01.csv`：`events=2704`, `span=13632`

Core0/1 局部视图与全量趋势一致，LDL 在关键路径上明显更短。

## 7. 实验方法说明

### 7.1 执行步骤

1. 编译 `Simulator`。
2. 在同一 `24` 核配置下分别运行：
  - `ldl_test`
  - `cholesky_test`
3. 两次运行均设置：
  - `ONNXIM_TRACE_CSV=...` 导出 trace；
  - `ONNXIM_MAX_CORE_CYCLES=120000` 防止死锁导致无界等待。
4. 用 `visualizer_png.py` 分别绘制：
  - 24 核全图（`timeline_c24.png`）
  - Core0/1 子集图（`timeline_core01.png`）

### 7.2 指标定义

- **span**：`end_cycle.max - start_cycle.min`，表示总时序跨度（关键指标）。
- **events**：trace 事件总数，反映指令/访存事件规模。
- **cube_events/vector_events**：按 `Unit` 前缀统计各执行单元事件数量。
- **avg_dur**：同类事件平均持续周期。

### 7.3 公平性与边界

- 本报告比较的是**当前仿真建模下的执行时序效率**，不是数值误差/收敛精度对比。
- 结论适用于当前 `M=64, U=16, batch=96, 24核` 设定；更大维度需单独复测。
- `ONNXIM_MAX_CORE_CYCLES` 仅作为保护阈值，不影响正常完成时的最终 trace 数据。

## 8. LDL 优化点（对应当前实现）

1. **算法路径更轻量**
  - Cholesky 基线含 `POTRF + TRSM + RK_UPDATE + backward solve` 的组合，Cube 更新链较长。
  - LDL 新算子避免了等价高密度更新链中的部分开销，关键路径更短。

2. **2x2 批拼接提升 Cube 利用**
  - 对 `block_size=2` 的 `L_UPDATE`，按目标 Cube 维度（16）进行 block 打包，减少微小 GEMM 条数。
  - 结果体现为 Cube 事件数大幅下降，同时总周期下降。

3. **依赖与稳定性完善**
  - 已对关键后向阶段补充保守屏障策略，避免依赖顺序过于乐观。
  - 运行时建议保留 `ONNXIM_MAX_CORE_CYCLES` 作为死锁保护。

## 9. 结论与建议

### 9.1 结论

在当前实验口径下，`LDL New` 相比 `Cholesky Baseline` 达到约 **71.94%** 的时延改善，属于“约 70%+”的显著提升。该提升主要由：

- 更短的算法更新链；
- 更少的 Cube 微指令；
- `2x2 -> 16x16` 的拼接带来的发射效率提升。

### 9.2 后续建议

1. 增加多组维度（如 `U=32/64`）做趋势验证。
2. 补充精度侧对比（BER/SE/重构误差）形成“性能-精度”联合结论。
3. 将拼接参数做可配置化，做消融实验（拼接开/关、拼接粒度）。

## 10. 结果文件索引

- LDL：
  - `ldl_new/timeline_c24.png`
  - `ldl_new/timeline_core01.png`
- Cholesky：
  - `cholesky_baseline/timeline_c24.png`
  - `cholesky_baseline/timeline_core01.png`

## 11. 两核流水线与分公式周期统计（2026-03-12 更新）

### 11.1 两核流水线图

- Cholesky（Core0/Core1）：`cholesky_baseline/timeline_core01.png`
- BlockLDL（Core0/Core1）：`ldl_new/timeline_core01.png`

对应两核 trace：

- `cholesky_baseline/trace_core01.csv`
- `ldl_new/trace_core01.csv`

### 11.2 周期统计口径

- 周期定义：单事件周期 `dur = end_cycle - start_cycle`。
- 单元分类：
  - 搬运：`*_MTE2`、`*_MTE3`
  - CUBE：`*_Cube`
  - VECTOR：`*_Vector`
- “公式步骤”由算子事件名前缀映射（如 `CHOL_RK_UPDATE`、`LDL_D_UPDATE` 等）。

### 11.3 Cholesky 分解/求逆分步骤周期表

Trace：`cholesky_baseline/trace_c24.csv`，关键路径跨度（span）=`13697` cycles。

| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X$ 与外存交换 | 258546 | 0 | 0 | 258546 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| POTRF对角开方 | $L_{jj}=\operatorname{chol}(\tilde A_{jj})$ | 0 | 0 | 3072 | 3072 | 768 |
| TRSM列缩放 | $L_{ij}=(\cdots)L_{jj}^{-H}$ | 0 | 0 | 10752 | 10752 | 2688 |
| Schur/RK更新 | $A_{ik}\leftarrow A_{ik}-L_{ij}L_{kj}^H$ | 0 | 258048 | 0 | 258048 | 8064 |
| 前向对角求逆 | $Y_{jj}=1/L_{jj}$ | 0 | 0 | 3072 | 3072 | 768 |
| 前向累乘 | $Y_{ij}\mathrel{-}=L_{ik}Y_{kj}$ | 0 | 86016 | 0 | 86016 | 2688 |
| 前向缩放 | $Y_{ij}\leftarrow Y_{ij}/L_{ii}$ | 0 | 0 | 10752 | 10752 | 2688 |
| 后向组装 | $A^{-1}=Y^H Y$ | 0 | 3072 | 0 | 3072 | 96 |
| 同步屏障(Barrier) | 列/阶段依赖同步 | 0 | 0 | 1824 | 1824 | 1824 |

### 11.4 BlockLDL 分解/求逆分步骤周期表

Trace：`ldl_new/trace_c24.csv`，关键路径跨度（span）=`3844` cycles。

| 步骤 | 公式 | 搬运周期 | CUBE周期 | VECTOR周期 | 总周期 | 事件数 |
|---|---|---:|---:|---:|---:|---:|
| 搬运(Load/Store) | $A/B/X$ 与外存交换 | 246520 | 0 | 0 | 246520 | 4608 |
| Gram矩阵构建 | $A=H^H H$ | 0 | 3360 | 0 | 3360 | 96 |
| 对角正则 | $A\leftarrow A+\lambda I$ | 0 | 0 | 96 | 96 | 96 |
| D块更新 | $D_{jj}=A_{jj}-\sum L_{jk}D_{kk}L_{jk}^H$ | 0 | 24576 | 0 | 24576 | 768 |
| D块求逆 | $D_{jj}^{-1}$ | 0 | 0 | 3072 | 3072 | 768 |
| L块更新 | $L_{ij}=(A_{ij}-\sum L_{ik}D_{kk}L_{jk}^H)D_{jj}^{-1}$ | 0 | 21504 | 0 | 21504 | 672 |
| 回代对角初始化 | $X_{jj}=D_{jj}^{-1}$ | 0 | 21504 | 672 | 22176 | 1344 |
| 回代（乘+加） | $tmp\mathrel{+}=L^H X\;\text{并累加到}\;X$ | 0 | 86016 | 2688 | 88704 | 5376 |
| 同步屏障(Barrier) | 列/阶段依赖同步 | 0 | 0 | 2688 | 2688 | 2688 |

注：当前实现中无独立 `LDL_BWD_OFF_SCALE` 指令；回代路径采用 `GEMM_PRELOAD + ADD` 融合执行，因此以“回代（乘+加）”合并展示，避免将其误解为单独 scale 步骤。

补充导出文件：

- `cycle_breakdown_cholesky.csv`
- `cycle_breakdown_blockldl.csv`
- `cycle_breakdown_tables.md`

## 12. 仿真工具对比与选型依据（为何采用 Asim）

本节用于回答“为什么本项目使用 `Asim` 而非其他常见 NPU/AI 仿真工具”。
对比维度聚焦于本研究任务的可验证需求：

1. 算子级执行流可改造（如 `LDL/Cholesky` 指令重排）；
2. 可定位微架构瓶颈（碎片化、RAW 冒险、Barrier）；
3. 可输出可复盘证据（周期、单元利用、指令 trace、时序图）；
4. 可联动存储/互联建模（DRAM/NoC）以避免“只算核内算力”的偏差。

### 12.1 工具横向对比

| 工具 | 主要定位 | 核心优势 | 与本研究目标的适配边界 | 典型适用场景 |
|---|---|---|---|---|
| **Asim (ONNXim 扩展)** | 多核 NPU 周期级仿真（Cube/Vector/MTE + SRAM + NoC + DRAM） | 指令级 trace 与周期可观测；支持 ONNX 与 C++ 自定义算子；便于快速迭代执行流 | 在公开标准化 benchmark 套件上的现成对照案例少于 Timeloop/SCALE-Sim | 本研究这类“算子-微架构协同优化 + 时序归因” |
| **SCALE-Sim v3** | Systolic 阵列层/层级性能建模 | 层级吞吐/带宽分析成熟；支持 Ramulator/Accelergy；评估效率高 | 以层级模型为主，对细粒度指令依赖与 barrier 策略表达能力有限 | 阵列规模、带宽、拓扑维度 DSE |
| **Timeloop (+Accelergy)** | 映射搜索与分析模型（性能/能耗） | 映射空间搜索能力强；学术复现生态完善 | 主要为解析模型，不直接提供指令执行语义；对指令链级错误定位能力有限 | 数据流/tiling/映射联合优化 |
| **MAESTRO** | 数据流 cost model（复用/性能/能耗） | 分析速度快；数据复用解释性强 | 非完整周期执行仿真，难以覆盖流水相关的数据相关与同步细节 | 前期 dataflow 筛选与敏感性分析 |
| **STONNE** | 可重构 DNN 加速器周期级仿真 | 支持 dense/sparse 与多种可重构架构抽象 | 体系结构抽象与本研究 Ascend 风格单元模型同构性较弱，迁移与标定成本较高 | 可重构阵列研究与数据流实验 |
| **Astra-sim** | 分布式训练系统级仿真（集群通信/调度） | 大规模训练系统（计算-通信协同）建模能力强 | 关注层级在系统/网络侧，不面向单 NPU 核内算子执行流优化 | 多节点训练架构与通信算法评估 |
| **Ramulator2** | DRAM 周期级仿真 | 内存系统建模专业、可扩展、验证充分 | 不提供 NPU 计算核执行语义，需与上层计算模拟器耦合 | 内存控制器/DRAM 机制研究 |

### 12.2 结合本报告任务的选型结论

对于本报告的核心问题（`LDL vs Cholesky` 执行流优化、阶段级周期归因、碎片化与依赖冒险修复），`Asim` 在方法与证据链两个层面的适配性最高，理由如下：

1. **可直接表达并验证指令级设计变更**  
  本文优化过程涉及 `GEMM_PRELOAD` 粒度重构、`PIPE_BARRIER` 插入、`k` 维去循环化等微结构策略，均需要“可编程指令生成 + 周期级执行验证”能力；`Asim` 可在算子源码层完成改造并闭环复现。

2. **观测闭环完整，支持可复核证据链**  
  本文基于 `trace_c24.csv`、`trace_core01.csv`、`timeline_c24/core01.png` 与分步骤周期表完成“问题定位→策略修正→量化验证”的完整链路。该链路依赖 `Asim` 的指令级 trace 与可视化工作流。

3. **计算-存储-互联联合建模**  
  `Asim` 在同一框架内覆盖 Core 执行、片上存储、NoC、DRAM（含 Ramulator 后端），可降低仅核内建模带来的偏差风险，更适合通信相关算子与复杂依赖链分析。

4. **与当前工程组织高度一致**  
  当前仓库已形成 `mode` + `model` + `operation` 的迭代范式；新增/修改算子无需迁移到异构工具链，可显著降低实验迭代成本并提升可维护性。

### 12.3 边界与组合建议

为避免单一工具偏置，建议采用组合式评估框架：

- **Asim**：用于算子级微架构设计、依赖正确性与周期归因（本报告主工具）；
- **Timeloop/MAESTRO**：用于前置映射空间与数据流筛选；
- **SCALE-Sim**：用于阵列级吞吐/带宽趋势快扫；
- **Astra-sim**：用于后续扩展到分布式训练系统级评估。

该组合可在“可解释性、速度、覆盖面”之间取得工程上更稳健的平衡。

### 12.4 面向项目书目标的总结论证

结合项目书中“**AI 模型驱动的高效算法与算子设计框架**”与“在 AI 核上实现 **5X（挑战 10X）** 加速”的目标，`Asim` 的方法学价值可归纳为三点：

1. **支撑算法-算子协同优化，而非仅软件调优**  
  项目目标明确要求加速来源于算法与算子设计层面的结构性优化。`Asim` 提供可编辑的算子执行流与微架构映射能力，可直接评估“算法变形→算子实现→硬件执行”的一致性与收益。

2. **支持“生成-评估-反馈”闭环中的可量化评估环节**  
  项目路线强调候选算法自动生成与迭代优化。`Asim` 可输出周期、利用率、访存、指令级时序等中间特征指标，为候选方案筛选提供可度量、可比较、可复核的反馈信号。

3. **对 MIMO 接收机场景具有工程可迁移性**  
  项目关注无线物理层接收机算法（含矩阵分解/求逆类核心算子）。本报告中 `LDL/Cholesky` 的阶段级优化实践表明，`Asim` 能有效暴露并修复影响吞吐的关键瓶颈（碎片化、依赖冒险、同步策略），具备向后续 MIMO 算法簇扩展的可迁移性。

**结论**：在本项目“算法创新 + 算子映射 + 周期级验证”三位一体的研究目标下，`Asim` 不仅是可用工具，更是可支撑方法闭环的核心实验平台；其优势在于能够把“候选算法的理论收益”转化为“AI 核执行层面的可验证收益”。
