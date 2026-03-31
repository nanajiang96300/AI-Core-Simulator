# BJ-DeepUnfold 与 Cholesky/LDL 对比报告

## 方法
- Cholesky-MMSE：对 `A=H^H H + (sigma^2/E_s)I` 做 Cholesky 分解求逆。
- Block-LDL：沿用项目中块 LDL 近似求逆流程。
- BJ-DeepUnfold：Block-Jacobi 预条件 + Chebyshev 深度展开迭代求逆。

## DeepUnfold 参数
- layers: `12`
- block size: `4`
- adaptive bounds: `True`

## 统一环境
- nr=64, nt=16, n_sc=8, batch=4, trials=1, snr=[0.0, 10.0, 20.0]
- modulation=16qam, pilot_len=16, num_format=fp16, seed=42

## 文件
- metrics: `bj_deepunfold_vs_cholesky_ldl_metrics.csv`
- BER: `ber_vs_snr_cholesky_ldl_bj_deepunfold.png`
- SE: `se_vs_snr_cholesky_ldl_bj_deepunfold.png`
- overlay: `se_ber_overlay_cholesky_ldl_bj_deepunfold.png`
