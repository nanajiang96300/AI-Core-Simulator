# LDL 分块并行优化说明（2026-03-27）

## 1. 先回答你的问题

你说的方向是对的：**LDL 在依赖上确实比 Cholesky 更容易并行化**，而且代码里已有可复用的“历史实现方法”。

本次我先做了“搜索旧实现 -> 套用到当前 block 路径 -> 实测验证”的闭环。

---

## 2. 搜到的“之前实现方法”

### 2.1 `L_UPDATE` 的批拼接（已有）

文件：`src/operations/LDLDecompOp.cc`

- 使用 `cube_m/n/k` 推导 `cube_pack_blocks`（`auto_pack_blocks`）
- 在 `for (i = j+1; i < n_blocks; i += cube_pack_blocks)` 中把多个 `2x2` 块拼成更大 `packed_dim`
- 通过 `LDL_L_UPDATE_*_PACK*` 降低碎片化发射

这套就是你说的“可并行批处理窗口”的核心旧方法。

### 2.2 no-block 的“动态K + 向量化”经验（已有）

来源：`results/LDL/falsification/LDL_NO_BLOCK_SCALAR_EVENT_VALIDATION_20260326.md`

- 之前已经验证过：把 `D_UPDATE` 从固定大 K 改成随阶段增长的 K，并优先走 Vector，可减少不必要 Cube 与等待。

### 2.3 文档结论（已有）

`DOCS/ASIM_FINAL_REPORT.md` 已明确：

- LDL 的 `D_UPDATE/D_INV/L_UPDATE` 更容易形成 ready 窗口；
- 关键收益是“指令组织与并行暴露”，不是单条指令神奇加速。

---

## 3. 本次针对 block LDL 的具体优化

修改文件：`src/operations/LDLDecompOp.cc`

### 3.1 小块长K更新优先走 Vector

新增 `pick_ldl_micro_mul_opcode(...)`：

- 条件：`blk <= 2 && tile_m <= 2 && tile_n <= 2`
- 动作：优先 `Opcode::MAC`（Vector）

应用到：

- `LDL_D_UPDATE_*`
- `LDL_BWD_DIAG_MUL_*`
- `LDL_BWD_OFF_MUL_*`

### 3.2 `D_UPDATE` 的 K 长度改为依赖感知

- 旧：`blk>1` 时固定 `d_update_k_len = U`
- 新：`d_update_k_len = max(blk, j * blk)`

这样更贴近“阶段性累加”的依赖规模，避免早期阶段用过大的 K。

---

## 4. 实测结果（同配置：`M=64,U=16,batch=96,block=2`）

### 4.1 全局结果

| Case | max_end | events | total_dur |
|---|---:|---:|---:|
| LDL block (old) | 3704 | 18497 | 424972 |
| LDL block (opt2) | 1568 | 17382 | 247399 |
| CHOL block (opt) | 3404 | 32637 | 275652 |

关键结论：

- LDL old -> opt2：`3704 -> 1568`（约 **2.36x** 提升）
- LDL opt2 相对 CHOL opt：`1568 vs 3404`（CHOL/LDL = **2.17x**）

### 4.2 单元占用变化（LDL old -> opt2）

| 指标 | old | opt2 |
|---|---:|---:|
| Cube events | 4608 | 672 |
| Cube dur | 147744 | 21792 |
| Wait events | 1313 | 198 |
| Wait dur | 20903 | 4991 |
| Vector events | 7968 | 11904 |

关键结论：

- 小块更新从 Cube 明显迁移到 Vector；
- Cube/Wait 大幅下降，关键路径显著收敛。

### 4.3 关键算子映射变化（LDL）

| Operation | old unit | opt2 unit |
|---|---|---|
| `LDL_D_UPDATE` | Cube | Vector |
| `LDL_BWD_DIAG_MUL` | Cube+Vector | Vector |
| `LDL_BWD_OFF_MUL` | Cube+Vector | Vector |
| `LDL_L_UPDATE` | 以 Cube 为主 | 保持不变（Cube主） |

解释：

- `L_UPDATE` 保留 pack + Cube 作为“大颗粒”路径；
- `D/BWD` 微更新切到 Vector，减少小任务上 Cube 的排队与空泡。

---

## 5. Core0 对比（新产物）

产物：

- 图：`results/LDL/falsification/ldl_cholesky_block_core0_timeline_ldl_opt2.png`
- 表：`results/LDL/falsification/LDL_CHOLESKY_BLOCK_CORE0_COMPARE_LDL_OPT2_20260327.md`

Core0 结果：

- LDL(opt2) `1495`
- CHOL(opt) `3396`
- 比值（CHOL/LDL）`2.2716`

---

## 6. 下一步可继续并行化的点

1. **BWD_OFF 分组打包**：按同 `off_k_len` 的 `(i,j)` 分组，尝试批量发射，继续压缩指令数。  
2. **可配置策略开关**：把“micro->Vector”做成属性（如 `ldl_micro_vector=true`），便于不同硬件参数下 A/B 测试。  
3. **barrier 细化**：在保证 RAW 安全前提下，减少列间过保守同步点，进一步提高重叠。
