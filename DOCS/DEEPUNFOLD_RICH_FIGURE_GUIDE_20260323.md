# DeepUnfold 对比图集详解（2026-03-23）

图目录：`results/compare_aligned/rich_figures/`
统一实验口径：`M=256, K=32, batch=96, ascend_910b_quiet`
对比方法：`LDL / DeepUnfold / DeepUnfold-Opt / Cholesky`

---

## 图1：`01_wall_vs_work_dual_axis.png`
**图意**
- 同时展示两类关键指标：
  - 柱状：真实端到端总周期 `wall_cycles`
  - 折线：累计工作量 `work_cycles_sum = sum(dur)`

**坐标与读法**
- 横轴：方法名称。
- 左纵轴：真实总周期（越小越好）。
- 右纵轴：累计工作量（越小越好）。

**关键观察**
- DeepUnfold/DeepUnfold-Opt 的 `work` 明显高于 LDL，但 `wall` 显著低于 LDL。
- Cholesky 的 `work` 与 LDL接近，但 `wall` 极高，说明其时序重叠效率较低。

**结论**
- 该图直接证明“工作量小”与“实际延迟小”不是同一件事。
- DeepUnfold 当前优势主要来自并行重叠，而非工作量更少。

**注意事项**
- 双纵轴图容易产生视觉错觉，应结合图2/图3验证判断。

---

## 图2：`02_wall_speed_vs_ldl.png`
**图意**
- 以 LDL 为基准展示各方法真实总周期比值 `wall_speed_vs_ldl`。

**坐标与读法**
- 横轴：方法。
- 纵轴：相对 LDL 的速度比（<1 表示比 LDL 更快；>1 表示更慢）。
- 虚线 `y=1`：与 LDL 持平分界线。

**关键观察**
- DeepUnfold-Opt 约 `0.457`，DeepUnfold 约 `0.496`，均快于 LDL。
- Cholesky 约 `7.34`，显著慢于 LDL。

**结论**
- 在当前实现下，真实延迟排序为：`DeepUnfold-Opt < DeepUnfold < LDL << Cholesky`。

**注意事项**
- 该图只看端到端延迟，不反映总工作量与资源消耗。

---

## 图3：`03_overlap_factor.png`
**图意**
- 展示并行重叠强度指标：
  $$\text{overlap\_factor}=\frac{\sum dur}{wall\_cycles}$$

**坐标与读法**
- 横轴：方法。
- 纵轴：重叠系数（数值越高，说明并行重叠程度越强）。

**关键观察**
- DeepUnfold-Opt 与 DeepUnfold 的重叠系数远高于 LDL。
- Cholesky 的重叠系数最低，解释了其 `wall` 偏大现象。

**结论**
- DeepUnfold 的低延迟核心来自高并发重叠能力，而不是减少了算子总工作量。

**注意事项**
- overlap_factor 是代理指标，不是硬件真实并行度上限。

---

## 图4：`04_engine_duration_stacked.png`
**图意**
- 绝对周期堆叠图，分解 `MTE2/MTE3/Cube/Vector` 的累计周期贡献。

**坐标与读法**
- 横轴：方法。
- 纵轴：累计周期（绝对值）。
- 每个柱子内部颜色表示不同执行单元贡献。

**关键观察**
- DeepUnfold 系列柱高主要由 MTE 段组成。
- Cholesky 的 Cube 段明显更高，计算密集特征突出。

**结论**
- DeepUnfold 路径当前瓶颈仍是搬运链路，计算单元占比较低。

**注意事项**
- 该图是 `sum(dur)` 维度，不能直接代表总时延。

---

## 图5：`05_engine_duration_ratio_stacked.png`
**图意**
- 将图4归一化为百分比，比较不同方法的“结构占比”。

**坐标与读法**
- 横轴：方法。
- 纵轴：占比（0–100%）。

**关键观察**
- DeepUnfold 与 DeepUnfold-Opt 的 MTE 占比接近满格（约 99.57%）。
- LDL 也偏 MTE 主导，但低于 DeepUnfold。
- Cholesky 的 MTE 占比明显下降，Cube 占比大幅提升。

**结论**
- DeepUnfold 的后续优化重点依然应聚焦搬运削减与依赖链压缩。

**注意事项**
- 占比图掩盖了绝对量级，需结合图4联合使用。

---

## 图6：`06_engine_event_count_grouped.png`
**图意**
- 比较各执行单元的事件条目数（event count）。

**坐标与读法**
- 横轴：方法。
- 纵轴：事件数。
- 分组柱：MTE2/MTE3/Cube/Vector 事件数量。

**关键观察**
- DeepUnfold-Opt 相比 DeepUnfold 的 Vector 事件数下降，符合向量阶段合并策略。
- Cholesky 的 Cube 事件数量远高于其他方法。

**结论**
- 当前优化已在“指令条目层面”减少部分向量开销，但总瓶颈仍在搬运。

**注意事项**
- 事件数不等于事件耗时，需与图4配合分析。

---

## 图7：`07_mte_vs_compute_scatter.png`
**图意**
- 二维散点展示“搬运工作量 vs 计算工作量”的分布关系。

**坐标与读法**
- 横轴：`MTE total duration`。
- 纵轴：`Cube+Vector duration`。
- 点大小：与 `dur_total` 相关（总工作量越大点越大）。

**关键观察**
- DeepUnfold 两点在横轴方向明显靠右（搬运大）。
- Cholesky 在纵轴方向更高（计算更重）。

**结论**
- 各方法的“瓶颈类型”差异清晰：DeepUnfold 偏搬运，Cholesky 偏计算。

**注意事项**
- 点大小为相对缩放，不是额外物理单位。

---

## 图8：`08_load_store_other_share.png`
**图意**
- 从步骤热点视角展示 `Load/Store/Others` 三类占比。

**坐标与读法**
- 横轴：方法。
- 纵轴：占比（%）。
- 堆叠区块：Load、Store、Others。

**关键观察**
- DeepUnfold 系列 Load+Store 占比接近全部，且 Load 为绝对主导。
- Cholesky 的 Others 占比更高，说明分解计算步骤更有存在感。

**结论**
- DeepUnfold 继续优化时，应优先改进 Load/Store 路径和数据驻留策略。

**注意事项**
- 本图基于 step breakdown 口径，`Others` 含未单列步骤与长尾项。

---

## 图9：`09_top_steps_small_multiples.png`
**图意**
- 四宫格分别给出每种方法 Top-8 高耗时步骤，便于逐方法定位热点。

**坐标与读法**
- 每个子图对应一种方法。
- 横轴：步骤累计周期。
- 纵轴：步骤名。

**关键观察**
- 四种方法均可看到明显头部步骤；DeepUnfold 的头部被 Load/Store 强主导。
- DeepUnfold-Opt 相比 DeepUnfold 的部分向量相关步骤有所收缩。

**结论**
- 可直接据此做“下一轮优化优先级排序”。

**注意事项**
- Top-N 图展示的是“头部”，不是全部步骤全景。

---

## 图10：`10_step_heatmap_top12.png`
**图意**
- 跨方法热力图，展示全局 Top-12 高耗时步骤在不同方法中的分布差异。

**坐标与读法**
- 横轴：方法。
- 纵轴：步骤名。
- 颜色越深：该方法在该步骤上的耗时越高。

**关键观察**
- DeepUnfold 列在 Load 相关行通常更“深色”。
- Cholesky 在其分解相关步骤上出现独特深色带。

**结论**
- 热力图适合快速识别“某方法独有热点”与“共性热点”。

**注意事项**
- 热力图颜色是相对强弱，跨图比较时应注意色标范围一致性。

---

## 图11：`11_work_latency_efficiency_map.png`
**图意**
- 把方法放到“工作量-延迟”二维平面，形成效率象限图。

**坐标与读法**
- 横轴：`work_cycles_sum`（越左越好）。
- 纵轴：`wall_cycles`（越低越好）。
- 点越靠左下，综合效率越优。

**关键观察**
- LDL 在工作量上更有优势（更靠左），但 DeepUnfold-Opt 在延迟上更有优势（更靠下）。
- Cholesky 在纵轴位置偏高，体现出延迟短板。

**结论**
- 不同方法在“省工作量”与“低延迟”之间存在权衡，工程决策应按业务目标选型。

**注意事项**
- 该图用于策略评估，不等价于正确性指标（需另看 BER/SE）。

---

## 推荐引用方式（论文/汇报）
- 若强调“端到端体验”：优先引用图2、图3、图11。
- 若强调“瓶颈与优化路径”：优先引用图4、图5、图8、图9、图10。
- 若解释“为何 DeepUnfold 虽工作量更大却更快”：连用图1 + 图3。
