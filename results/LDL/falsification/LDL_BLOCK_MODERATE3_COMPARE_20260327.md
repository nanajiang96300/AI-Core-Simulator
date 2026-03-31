# LDL 分块温和优化（~3000 cycles）与 CHOL/标准LDL 对比

## 1) 采用配置

- 代码：`src/operations/LDLDecompOp.cc`
  - 自动 pack 上限：`auto_pack_blocks <= 2`
  - `D_UPDATE`：`d_update_k_len = (blk > 1) ? U : blk`
- 用例：`example/ldl_test_moderate3.json`
  - `block_size=2`
  - `bwd_steps=3`
  - `pack_blocks=2`

## 2) 关键结果

- LDL 分块（moderate3）全局周期：`3032`
- Core0 结束周期：`2962`
- Core0 单元占比（dur-sum 口径）：
  - `MTE2=62.53%`
  - `Cube=11.53%`
  - `Vector=8.00%`
  - `Wait=9.68%`
  - `MTE3=8.25%`

## 3) 三者对比（同一画图脚本）

- Cholesky block(opt)：`global_max_end=3404`
- 标准 LDL(no-block)：`global_max_end=4290`
- LDL block(moderate3)：`global_max_end=3032`

## 4) 产出文件

- 三者对比时序图：`results/LDL/falsification/chol_stdldl_blockldl_timeline_compare_moderate3_20260327.png`
- 对比统计 CSV：`results/LDL/falsification/chol_stdldl_blockldl_timeline_compare_moderate3_20260327.csv`
- 绘图脚本：`scripts/compare_chol_stdldl_blockldl_timeline.py`

## 5) 备注

如果希望进一步降低 Cube 占比，可使用：
- `example/ldl_test_moderate3_nopack.json`（`pack_blocks=1`）
- 该配置周期约 `2594`，更偏“保守”但不再接近 `3000`。
