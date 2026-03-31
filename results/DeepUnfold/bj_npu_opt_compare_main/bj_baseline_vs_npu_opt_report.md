# BJ Baseline vs NPU-Optimized 对比

- 说明：优化版为新增文件，不覆盖基线实现。
- 精度目标：BER/SE 与基线一致，且不劣于 Cholesky/LDL 趋势。
- 性能目标：NPU形态（分块GEMM+Vector更新）便于映射与统计。
- Overlap 版本：按算子做 H/Y 分块（逻辑等价）用于正确性校验。

- metrics: `bj_baseline_vs_npu_opt_metrics.csv`
- ber fig: `ber_vs_snr_bj_baseline_npu_opt.png`
- se fig: `se_vs_snr_bj_baseline_npu_opt.png`
- time fig: `time_per_sample_ms_bj_baseline_npu_opt.png`
