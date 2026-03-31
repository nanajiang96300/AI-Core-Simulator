# LDL old vs opt2 BER 正确性验证（2026-03-27）

## 1) 验证目标

针对你提出的要求，本次验证只看**数值正确性（BER/SE/逆矩阵误差）**，不看周期。

比较对象：

- `Exact MMSE`（`A^{-1}` 用 `numpy.linalg.inv`）
- `LDL-old`（算子优化前）
- `LDL-opt2`（算子优化后）

脚本：`scripts/validate_ldl_opt2_ber_correctness.py`（实现口径参考 `scripts/validate_cholesky_iso_ber_vs_ldl.py`）

---

## 2) 运行参数

- `nr=64, nt=16, n_sc=8, batch=4, trials=2`
- `pilot_len=16, block_size=2, seed=42`
- `snr_db = [0, 5, 10, 15, 20]`

结果 CSV：`results/LDL/falsification/ldl_old_opt2_ber_validation_20260327.csv`

---

## 3) 关键结果

在全部 SNR 点上：

1. `BER exact == BER old == BER opt2`  
2. `SE exact == SE old == SE opt2`  
3. 逆矩阵相对误差 `old/exact` 与 `opt2/exact` 均约 `2.9e-16` 量级  
4. `inv_relerr(old,opt2) = 0`

示例（10 dB）：

- BER：`8.789062e-03 / 8.789062e-03 / 8.789062e-03`
- SE：`79.9475 / 79.9475 / 79.9475`
- `inv_relerr old/exact = 2.858e-16`
- `inv_relerr opt2/exact = 2.858e-16`
- `inv_relerr old/opt2 = 0`

---

## 4) 结论

- 这次 `LDL opt2` 的算子修改在当前验证口径下**不改变 BER 正确性**。
- 优化属于调度/路由层面（micro-step 从 Cube 到 Vector、`D_UPDATE` 的 K 长度建模修正），
  数值输出与 `LDL-old`、`Exact` 一致。
