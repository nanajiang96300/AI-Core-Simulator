# Newton–Schulz 910B 实验结果说明

本目录存放的是在 `configs/ascend_910b_quiet.json` 配置下，对 `example/newton_schulz_test.json` C++ 测试模型运行所得的 profile 结果（batch=96，matrix_m/matrix_k 见 JSON）。

## 文件说明

- `profiling_log_newton_910b.csv` / `.png`  
  - 早期 Newton–Schulz 单步版本（实现中虽然 `iterations`>1，但内部始终使用初始 X，不做真实多步迭代），32×32 配置。
- `profiling_log_newton_910b_v2.csv` / `.png`  
  - 单步版本修复后、排查伪并行与 batch-tiling 配置（每核约 4 个 batch）的 910B 实验；主要用于与 LS 算子 pipeline 对比。
- `profiling_log_newton_910b_batch96.csv` / `.png`  
  - 32×32、`batch_size=96`，单 op batched 结构（消除 model 层双重 batching）之后的结果，验证 24 核并行度与 DRAM 带宽利用。
- `profiling_log_newton_910b_256x32.csv`  
  - 256×32 配置（`matrix_m=256, matrix_k=32`），batch=96；用于评估在 16×16×16 阵列上的波次分解与阵列/带宽利用率，无对应 PNG（如需可用同名生成）。
- `profiling_log_newton_910b_multiiter.csv` / `.png`  
  - **当前版本**：NewtonSchulzOp 改为真正多步迭代（后续迭代从 ACCUM 中的上一次 X 读取），在 910B 上的 32×32、batch=96 结果，用于对比单步 vs 多步迭代在 pipeline 上的差异。
- `debug_trace.csv`  
  - 调试用细粒度 trace，通常对应某次问题排查时打开的调试开关。

> 提示：同一条实验线的 CSV 与 PNG 现在已采用相同 basename（如 `profiling_log_newton_910b_multiiter.*`），便于快速定位。