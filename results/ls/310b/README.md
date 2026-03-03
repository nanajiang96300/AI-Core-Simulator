# LS Channel Estimation 310B 实验结果说明

本目录存放的是 LS 信道估计算子（ChannelModel）在 310B 配置下运行得到的 profile 结果，主要用于对比 Newton–Schulz 算子的 pipeline 形态。

## 文件说明

- `profiling_log_ls_310b.csv` / `.png`  
  - LS 信道估计算子在 310B 配置下的 pipeline/profile 结果，用作通信算子与矩阵求逆算子的对照基线。

> 更详细的模型与映射配置，请参考相应的 config 与 model_list JSON。