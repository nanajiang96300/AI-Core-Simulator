# Newton–Schulz 8-Core 实验结果说明

本目录为在 8 核 systolic_ws 配置下运行 Newton–Schulz C++ 测试模型的结果，用于对比多核规模变化对 pipeline 的影响。

## 文件说明

- `profiling_log_newton_8core.csv` / `.png`  
  - 8 核配置下的 Newton–Schulz 实验结果，主要用于与 24 核 910B 配置的 pipeline PNG 做对比分析（核数缩小时的负载均衡与 DRAM 利用变化）。

> 具体 core 配置和 batch/matrix 大小请参考对应的 config JSON。