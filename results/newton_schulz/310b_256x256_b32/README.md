# Newton–Schulz 310B 256x256 batch32 实验说明

本目录用于 310B 配置下，矩阵尺寸约为 256x256、batch≈32 的 Newton–Schulz 实验（用于探索更大矩阵时的阵列利用率与带宽瓶颈）。

## 文件说明

- `newton_pipeline_310b_256x256_b32.png`  
  - 仅包含早期生成的 pipeline PNG，对应的 CSV profiling 文件暂未统一命名或可能位于其他目录；如需后续分析，建议重新运行实验并生成 `profiling_log_*.csv` 与同名 PNG。

> 若之后在本目录下增加对应 CSV，可按 910B 目录的风格重命名为同名 basename（例如 `profiling_log_newton_310b_256x256_b32.*`），并在本 README 中补充说明。