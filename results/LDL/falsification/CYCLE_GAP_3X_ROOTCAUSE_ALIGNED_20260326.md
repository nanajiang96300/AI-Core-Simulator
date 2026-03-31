# 完全对齐版：LDL vs Cholesky 周期差约 3x 的根因分析（Core0）

## 1. 范围与数据

- 对比对象：
  - LDL（对称改造版，`inv+mul`）：`results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv`
  - Cholesky（`inv*mul`）：`results/CHOL/falsification/cholesky_noblock_64x16_trace_invmul.csv`
- 统计范围：仅 `Core0_*` 事件
- 矩阵规模：`U=16`，Core0 对应 4 个 batch

## 2. 现象：为什么说“约 3 倍”

| 指标 | LDL-aligned | Cholesky-invmul | 比值 |
|---|---:|---:|---:|
| `max_end_cycle` | 4282 | 15236 | **3.56x** |
| 总事件时长和 | 12779 | 18753 | 1.47x |
| Vector 事件数 | 1964 | 7216 | **3.67x** |

关键观察：

1. `max_end_cycle` 差距是 `3.56x`；
2. 但“时长和”仅 `1.47x`；
3. 真正贴近 `max_end_cycle` 的是 **Vector 事件数 3.67x**。  
=> 瓶颈主要来自**单核 Vector 流水的串行深度**，而不是单个事件太慢。

## 3. 按步骤拆解：谁在拉开差距

以下是 `Cholesky - LDL` 的主要增量（按周期差从大到小）：

| 操作 | 周期增量 |
|---|---:|
| `CHOL_NB_RK_UPDATE` | +2720 |
| `CHOL_NB_TRSM_NUM_UPD` | +2240 |
| `CHOL_NB_TRSM_MUL` | +480 |
| `CHOL_NB_POTRF_DIAG_UPD` | +480 |
| `CHOL_NB_FWD_OFF_MUL` | +480 |
| `CHOL_NB_FWD_OFF_MAC` | +480 |
| `CHOL_NB_TRSM_DIAG_INV` | +256 |
| `CHOL_NB_POTRF_DIAG_SQRT` | +256 |
| `CHOL_NB_FWD_DIAG_INV` | +256 |

而 LDL 侧对应有一些“减少项”（即 Cholesky 没有这批事件），例如：

- `LDL_L_UPDATE` `-480`
- `LDL_BWD_OFF_MUL` `-480`
- `LDL_BWD_OFF_ACC` `-480`
- `LDL_D_DIAG_INV` `-256`
- `LDL_D_UPDATE / LDL_D_INV_MUL / LDL_BWD_DIAG_*` 等小项

净效果仍是 Cholesky 显著更长。

## 4. 每一步的机制解释（为什么会这样）

### 4.1 Gram / Reg / 内存不是主因

- `Load` 差异仅 `+86` 周期（`9396 vs 9310`）；
- `Store` 基本相同；
- `Gram`、`Wait` 也只贡献小头。  
=> 内存与 Cube 不是 3x 差距主导项。

### 4.2 主因 A：Cholesky 的分解阶段更新体量更大

在 no-block 下，Cholesky 分解有两类高体量更新：

- `TRSM_NUM_UPD`
- `RK_UPDATE`

对应 `Core0` 统计：

- `TRSM_NUM_UPD = 2240`
- `RK_UPDATE = 2720`

这两项合计 `4960` 周期，单独就超过 LDL 全部 Vector 执行时长的大头。

### 4.3 主因 B：前向阶段仍有双重循环更新

即使把 `DIV` 改成 `inv*mul`，Cholesky 仍保留：

- `FWD_OFF_MAC = 480`
- `FWD_OFF_MUL = 480`

这些都是逐对 `(i,c)` 的更新，数量级固定在三角展开规模，导致 Vector 队列更深。

### 4.4 主因 C：单核 Vector 资源串行执行，事件数直接映射到尾延迟

`SystolicWS` 的 Vector 路径单发（前一条未完成不再发下一条），因此：

- Cholesky Vector 事件 `7216`
- LDL Vector 事件 `1964`
- 比值 `3.67x` 与 `max_end_cycle` 比值 `3.56x` 非常接近

这说明：**3x 级周期差主要是“串行深度差”而非“单条慢很多”**。

## 5. 回答“是不是 sqrt 导致”

不是主因。`SQRT` 的确是 Cholesky 独有，但贡献是：

- `CHOL_NB_POTRF_DIAG_SQRT = 64 次 / 256 周期`（Core0）

它远小于 `RK_UPDATE + TRSM_NUM_UPD` 的 `4960` 周期量级。

## 6. 结论

在“完全对齐（双方都用 inv+mul）”后，周期仍约 `3.56x` 的根因是：

1. Cholesky 在分解+前向阶段的三角更新项更多（尤其 `RK_UPDATE`、`TRSM_NUM_UPD`）；
2. 这些更新大多落在 Vector 路径，形成更长的单核串行链；
3. 内存、Cube、sqrt 只占次要部分。

换句话说，当前差距主要反映**算法/循环结构导致的更新体量差**，而不再是“除法建模不公平”带来的假象。
