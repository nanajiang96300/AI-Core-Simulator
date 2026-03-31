# LDL 实验结果说明

本目录存放 `LDLDecompOp` 在 Asim 上的一次基线仿真实验结果。

## 实验配置

- 模式：`ldl_test`
- 配置文件：`configs/ascend_910b_quiet.json`
- 模型列表：`example/ldl_test.json`
- 矩阵维度：`H = [M, U] = [64, 16]`
- Gram/目标矩阵维度：`A = [U, U] = [16, 16]`
- Batch：`96`
- 分块参数：`block_size = 2`
- 回代参数：`bwd_steps = 1`

## 结果文件

- 时序 CSV：`profiling_log_ldl_64x16_b96.csv`
- 时序图 PNG：`profiling_log_ldl_64x16_b96.png`

## 复现实验命令

```bash
cd /project/Asim
ONNXIM_TRACE_CSV=results/LDL/profiling_log_ldl_64x16_b96.csv \
  ./build_asim/bin/Simulator \
  --config configs/ascend_910b_quiet.json \
  --models_list example/ldl_test.json \
  --mode ldl_test \
  --log_level info

python3 visualizer_png.py \
  -i results/LDL/profiling_log_ldl_64x16_b96.csv \
  -o results/LDL/profiling_log_ldl_64x16_b96.png
```

> 说明：当前 LDL 算子为执行流建模版本，重点用于评估 NPU 指令级时序与资源利用率。
