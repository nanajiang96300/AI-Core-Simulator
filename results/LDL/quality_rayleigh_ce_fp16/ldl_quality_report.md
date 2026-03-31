# LDL 数值合理性评估报告

## 评估目标
通过 BER 与频谱效率（SE）对比 `Block-LDL`、`Exact MMSE(估计信道)` 与 `Oracle MMSE(真实信道)`，判断 LDL 算子结果是否在合理范围。

## 实验配置
- N_r: `64`
- N_t: `16`
- Subcarriers (N_sc): `168`
- Batch: `16`
- Trials: `2`
- Block size: `2`
- Channel model: `rayleigh`
- Channel estimation: `LS` (pilot_len=16)
- Numeric format: `fp16`
- Reciprocal mode: `approx`
- Truncation mantissa bits: `8`
- Modulation: `16qam`
- MAC accumulation chunk: `4`
- Pilot SNR: `same as data SNR`
- SNR(dB): `[0.0, 5.0, 10.0, 15.0, 20.0]`
- Seed: `42`

## 结果文件
- 指标表: `ldl_quality_metrics.csv`
- BER 图: `ldl_ber_vs_snr.png`
- SE 图: `ldl_se_vs_snr.png`
- 重构误差图: `ldl_recon_error_vs_snr.png`

## 自动判定
- 最大 BER 差值: `0.001369`
- 最大 SE 差值: `0.074445`
- 判定: **PASS**

> 注：该判定用于工程 sanity-check，不等价于严格通信算法精度证明。
