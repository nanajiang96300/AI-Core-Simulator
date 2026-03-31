#!/usr/bin/env python3

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvalConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    block_size: int
    snr_db_list: List[float]
    channel_model: str
    pilot_len: int
    pilot_snr_db: Optional[float]
    num_format: str
    reciprocal_mode: str
    trunc_mantissa_bits: int
    modulation: str
    mac_chunk: int
    seed: int
    out_dir: str


def _truncate_real_mantissa(values: np.ndarray, keep_bits: int) -> np.ndarray:
    if keep_bits >= 10:
        return values

    arr = np.array(values, dtype=np.float64, copy=True)
    finite_mask = np.isfinite(arr)
    nz_mask = finite_mask & (arr != 0.0)
    if not np.any(nz_mask):
        return arr

    abs_vals = np.abs(arr[nz_mask])
    exponents = np.floor(np.log2(abs_vals))
    step = np.power(2.0, exponents - keep_bits)
    truncated = np.floor(abs_vals / step) * step
    arr[nz_mask] = np.sign(arr[nz_mask]) * truncated
    return arr


def quantize_complex(values: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    if cfg.num_format == "fp16":
        real = arr.real.astype(np.float16).astype(np.float64)
        imag = arr.imag.astype(np.float16).astype(np.float64)
        real = _truncate_real_mantissa(real, cfg.trunc_mantissa_bits)
        imag = _truncate_real_mantissa(imag, cfg.trunc_mantissa_bits)
        return real + 1j * imag
    return arr.astype(np.complex128, copy=True)


def qmatmul(a: np.ndarray, b: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"qmatmul shape mismatch: {a.shape} x {b.shape}")

    chunk = max(1, cfg.mac_chunk)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.complex128)
    for start in range(0, a.shape[1], chunk):
        end = min(start + chunk, a.shape[1])
        partial = a[:, start:end] @ b[start:end, :]
        partial = quantize_complex(partial, cfg)
        out = quantize_complex(out + partial, cfg)
    return out


def approx_reciprocal_scalar(x: complex, cfg: EvalConfig) -> complex:
    if cfg.reciprocal_mode == "exact":
        return 1.0 / x

    real_x = np.float64(np.real(x))
    imag_x = np.float64(np.imag(x))
    if abs(imag_x) > 1e-9:
        return 1.0 / (real_x + 1j * imag_x)

    val = real_x
    if abs(val) < 1e-12:
        val = np.copysign(1e-12, val if val != 0 else 1.0)

    y0 = np.float16(1.0 / val).astype(np.float64)
    y1 = y0 * (2.0 - val * y0)
    y1 = _truncate_real_mantissa(np.array([y1]), cfg.trunc_mantissa_bits)[0]
    return complex(y1, 0.0)


def inv_2x2_complex(block: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    block = quantize_complex(block, cfg)
    a = block[0, 0]
    b = block[0, 1]
    b_conj = np.conj(b)
    d = block[1, 1]

    det_real = np.real(a * d - b * b_conj)
    det = complex(det_real, 0.0)
    eps = 1e-12
    if np.abs(det) < eps:
        det = det + eps

    inv_det = approx_reciprocal_scalar(det, cfg)
    numer = np.array([[d, -b], [-b_conj, a]], dtype=np.complex128)
    inv_block = quantize_complex(numer * inv_det, cfg)
    return inv_block


def _get_block(mat: np.ndarray, idx_r: int, idx_c: int, blk: int) -> np.ndarray:
    r0, r1 = idx_r * blk, (idx_r + 1) * blk
    c0, c1 = idx_c * blk, (idx_c + 1) * blk
    return mat[r0:r1, c0:c1]


def _set_block(mat: np.ndarray, idx_r: int, idx_c: int, blk: int, value: np.ndarray):
    r0, r1 = idx_r * blk, (idx_r + 1) * blk
    c0, c1 = idx_c * blk, (idx_c + 1) * blk
    mat[r0:r1, c0:c1] = value


def block_ldl_decompose(a: np.ndarray, cfg: EvalConfig, block_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    a = quantize_complex(a, cfg)
    n = a.shape[0]
    if n % block_size != 0:
        raise ValueError(f"nt={n} must be divisible by block_size={block_size}")

    n_blocks = n // block_size
    l_blocks = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]
    d_blocks = [None for _ in range(n_blocks)]

    identity_block = np.eye(block_size, dtype=np.complex128)
    for idx in range(n_blocks):
        l_blocks[idx][idx] = quantize_complex(identity_block.copy(), cfg)

    for j in range(n_blocks):
        a_jj = a[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
        summation = np.zeros((block_size, block_size), dtype=np.complex128)
        for k in range(j):
            ljk = l_blocks[j][k]
            dkk = d_blocks[k]
            summation += qmatmul(qmatmul(ljk, dkk, cfg), ljk.conj().T, cfg)

        d_jj = quantize_complex(a_jj - summation, cfg)
        d_blocks[j] = quantize_complex(d_jj, cfg)
        d_jj_inv = inv_2x2_complex(d_jj, cfg)

        for i in range(j + 1, n_blocks):
            a_ij = a[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            summation_ij = np.zeros((block_size, block_size), dtype=np.complex128)
            for k in range(j):
                lik = l_blocks[i][k]
                ljk = l_blocks[j][k]
                dkk = d_blocks[k]
                summation_ij += qmatmul(qmatmul(lik, dkk, cfg), ljk.conj().T, cfg)
            l_blocks[i][j] = qmatmul(quantize_complex(a_ij - summation_ij, cfg), d_jj_inv, cfg)

    l = np.zeros((n, n), dtype=np.complex128)
    d = np.zeros((n, n), dtype=np.complex128)
    for i in range(n_blocks):
        for j in range(i + 1):
            block = l_blocks[i][j]
            if block is not None:
                l[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = quantize_complex(block, cfg)
        d[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = quantize_complex(d_blocks[i], cfg)

    return quantize_complex(l, cfg), quantize_complex(d, cfg)


def ldl_inverse(a: np.ndarray, cfg: EvalConfig, block_size: int = 2) -> Tuple[np.ndarray, float]:
    l, d = block_ldl_decompose(a, cfg, block_size=block_size)

    n = a.shape[0]
    n_blocks = n // block_size

    d_inv_blocks = []
    for j in range(n_blocks):
        d_jj = _get_block(d, j, j, block_size)
        d_inv_blocks.append(inv_2x2_complex(d_jj, cfg))

    x = np.zeros_like(a, dtype=np.complex128)

    # Pure backward-style block solve (paper-inspired):
    # compute columns from right to left without explicit L^{-1} matrix construction.
    for j in range(n_blocks - 1, -1, -1):
        # diagonal block
        x_jj = d_inv_blocks[j].copy()
        for k in range(j + 1, n_blocks):
            l_kj_h = _get_block(l, k, j, block_size).conj().T
            x_jk_h = _get_block(x, j, k, block_size).conj().T
            x_jj = quantize_complex(x_jj - qmatmul(l_kj_h, x_jk_h, cfg), cfg)
        _set_block(x, j, j, block_size, x_jj)

        # off-diagonal blocks in column j
        for i in range(j - 1, -1, -1):
            accum = np.zeros((block_size, block_size), dtype=np.complex128)
            for k in range(i + 1, n_blocks):
                l_ki_h = _get_block(l, k, i, block_size).conj().T
                x_kj = _get_block(x, k, j, block_size)
                accum = quantize_complex(accum + qmatmul(l_ki_h, x_kj, cfg), cfg)

            x_ij = quantize_complex(-accum, cfg)
            _set_block(x, i, j, block_size, x_ij)
            _set_block(x, j, i, block_size, x_ij.conj().T)

    a_inv = quantize_complex(x, cfg)

    recon = quantize_complex(l @ d @ l.conj().T, cfg)
    recon_err = np.linalg.norm(a - recon, ord="fro") / (np.linalg.norm(a, ord="fro") + 1e-12)
    return a_inv, float(recon_err)


def bits_to_16qam(bits: np.ndarray) -> np.ndarray:
    # Gray-coded 2-bit PAM mapping: 00->-3, 01->-1, 11->+1, 10->+3
    mapping = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64)
    i_idx = bits[:, 0] * 2 + bits[:, 1]
    q_idx = bits[:, 2] * 2 + bits[:, 3]
    i_val = mapping[i_idx]
    q_val = mapping[q_idx]
    return (i_val + 1j * q_val) / np.sqrt(10.0)


def demod_16qam(symbols: np.ndarray) -> np.ndarray:
    # bits_to_16qam() normalizes constellation by sqrt(10), so recover
    # the original PAM scale (-3, -1, +1, +3) before threshold slicing.
    scaled = symbols * np.sqrt(10.0)

    def slicer(vals: np.ndarray):
        out = np.zeros((vals.shape[0], 2), dtype=np.int32)
        out[vals < -2.0] = [0, 0]
        mask = (vals >= -2.0) & (vals < 0.0)
        out[mask] = [0, 1]
        mask = (vals >= 0.0) & (vals < 2.0)
        out[mask] = [1, 1]
        out[vals >= 2.0] = [1, 0]
        return out

    i_bits = slicer(np.real(scaled))
    q_bits = slicer(np.imag(scaled))
    return np.concatenate([i_bits, q_bits], axis=1)


def estimate_se(w: np.ndarray, h: np.ndarray, noise_var: float) -> float:
    g = w @ h
    ww_h = w @ w.conj().T
    se = 0.0
    nt = g.shape[0]
    for u in range(nt):
        signal = np.abs(g[u, u]) ** 2
        inter = np.sum(np.abs(g[u, :]) ** 2) - signal
        noise = noise_var * np.real(ww_h[u, u])
        sinr = signal / (inter + noise + 1e-12)
        se += np.log2(1.0 + sinr)
    return float(np.real(se))


def generate_channel(rng: np.random.Generator, nr: int, nt: int, channel_model: str) -> np.ndarray:
    if channel_model.lower() == "rayleigh":
        return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)
    raise ValueError(f"Unsupported channel model: {channel_model}")


def ls_channel_estimate(
    rng: np.random.Generator,
    h_true: np.ndarray,
    pilot_len: int,
    pilot_noise_var: float,
) -> np.ndarray:
    nt = h_true.shape[1]
    if pilot_len < nt:
        raise ValueError(f"pilot_len ({pilot_len}) must be >= nt ({nt}) for LS estimation")

    pilot_power = 1.0
    x_p = np.zeros((nt, pilot_len), dtype=np.complex128)
    x_p[:, :nt] = np.sqrt(pilot_power) * np.eye(nt, dtype=np.complex128)

    n_p = np.sqrt(pilot_noise_var / 2.0) * (
        rng.standard_normal((h_true.shape[0], pilot_len))
        + 1j * rng.standard_normal((h_true.shape[0], pilot_len))
    )
    y_p = h_true @ x_p + n_p

    gram = x_p @ x_p.conj().T
    h_hat = y_p @ x_p.conj().T @ np.linalg.inv(gram + 1e-12 * np.eye(nt, dtype=np.complex128))
    return h_hat


def run_eval(cfg: EvalConfig):
    rng = np.random.default_rng(cfg.seed)

    metrics = []
    t_global = time.time()

    for snr_index, snr_db in enumerate(cfg.snr_db_list, start=1):
        t_snr = time.time()
        print(
            f"[progress] SNR {snr_db} dB ({snr_index}/{len(cfg.snr_db_list)}) started...",
            flush=True,
        )
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin

        bit_err_exact = 0
        bit_err_ldl = 0
        bit_err_oracle = 0
        total_bits = 0

        se_exact_sum = 0.0
        se_ldl_sum = 0.0
        se_oracle_sum = 0.0
        recon_err_sum = 0.0

        total_samples = cfg.trials * cfg.batch * cfg.n_sc
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        for _ in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt, cfg.channel_model)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            if cfg.modulation == "16qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
                x = bits_to_16qam(tx_bits)
            else:
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 1), dtype=np.int32)
                x = (2 * tx_bits[:, 0] - 1).astype(np.float64).astype(np.complex128)

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y = h_true @ x + noise

            a_est = h_est.conj().T @ h_est + noise_var * np.eye(cfg.nt, dtype=np.complex128)
            a_oracle = h_true.conj().T @ h_true + noise_var * np.eye(cfg.nt, dtype=np.complex128)

            h_h_est = h_est.conj().T
            h_h_true = h_true.conj().T

            a_inv_exact = np.linalg.inv(a_est)
            a_inv_oracle = np.linalg.inv(a_oracle)
            a_inv_ldl, recon_err = ldl_inverse(a_est, cfg, block_size=cfg.block_size)

            w_exact = a_inv_exact @ h_h_est
            w_oracle = a_inv_oracle @ h_h_true
            w_ldl = a_inv_ldl @ h_h_est

            xhat_exact = w_exact @ y
            xhat_oracle = w_oracle @ y
            xhat_ldl = w_ldl @ y

            if cfg.modulation == "16qam":
                bits_hat_exact = demod_16qam(xhat_exact)
                bits_hat_oracle = demod_16qam(xhat_oracle)
                bits_hat_ldl = demod_16qam(xhat_ldl)
                bit_err_exact += int(np.sum(bits_hat_exact != tx_bits))
                bit_err_oracle += int(np.sum(bits_hat_oracle != tx_bits))
                bit_err_ldl += int(np.sum(bits_hat_ldl != tx_bits))
                total_bits += cfg.nt * 4
            else:
                bits_hat_exact = (np.real(xhat_exact) >= 0).astype(np.int32)[:, None]
                bits_hat_oracle = (np.real(xhat_oracle) >= 0).astype(np.int32)[:, None]
                bits_hat_ldl = (np.real(xhat_ldl) >= 0).astype(np.int32)[:, None]
                bit_err_exact += int(np.sum(bits_hat_exact != tx_bits))
                bit_err_oracle += int(np.sum(bits_hat_oracle != tx_bits))
                bit_err_ldl += int(np.sum(bits_hat_ldl != tx_bits))
                total_bits += cfg.nt

            se_exact_sum += estimate_se(w_exact, h_true, noise_var)
            se_oracle_sum += estimate_se(w_oracle, h_true, noise_var)
            se_ldl_sum += estimate_se(w_ldl, h_true, noise_var)
            recon_err_sum += recon_err

        ber_exact = bit_err_exact / max(total_bits, 1)
        ber_oracle = bit_err_oracle / max(total_bits, 1)
        ber_ldl = bit_err_ldl / max(total_bits, 1)
        se_exact = se_exact_sum / max(total_samples, 1)
        se_oracle = se_oracle_sum / max(total_samples, 1)
        se_ldl = se_ldl_sum / max(total_samples, 1)
        recon_err_avg = recon_err_sum / max(total_samples, 1)

        metrics.append(
            {
                "snr_db": snr_db,
                "ber_oracle_mmse": ber_oracle,
                "ber_exact_mmse": ber_exact,
                "ber_ldl": ber_ldl,
                "se_oracle_mmse": se_oracle,
                "se_exact_mmse": se_exact,
                "se_ldl": se_ldl,
                "recon_error": recon_err_avg,
            }
        )

        print(
            f"[progress] SNR {snr_db} dB done in {time.time() - t_snr:.2f}s, elapsed {time.time() - t_global:.2f}s",
            flush=True,
        )

    return metrics


def save_outputs(cfg: EvalConfig, metrics):
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_path = os.path.join(cfg.out_dir, "ldl_quality_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "ber_oracle_mmse",
                "ber_exact_mmse",
                "ber_ldl",
                "se_oracle_mmse",
                "se_exact_mmse",
                "se_ldl",
                "recon_error",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    snr = [row["snr_db"] for row in metrics]
    ber_oracle = [row["ber_oracle_mmse"] for row in metrics]
    ber_exact = [row["ber_exact_mmse"] for row in metrics]
    ber_ldl = [row["ber_ldl"] for row in metrics]
    se_oracle = [row["se_oracle_mmse"] for row in metrics]
    se_exact = [row["se_exact_mmse"] for row in metrics]
    se_ldl = [row["se_ldl"] for row in metrics]
    recon = [row["recon_error"] for row in metrics]

    ber_eps = 1e-8
    ber_oracle_plot = [max(v, ber_eps) for v in ber_oracle]
    ber_exact_plot = [max(v, ber_eps) for v in ber_exact]
    ber_ldl_plot = [max(v, ber_eps) for v in ber_ldl]

    plt.figure(figsize=(7, 5))
    plt.semilogy(snr, ber_oracle_plot, marker="^", label="Oracle MMSE (True H)")
    plt.semilogy(snr, ber_exact_plot, marker="o", label="Exact MMSE")
    plt.semilogy(snr, ber_ldl_plot, marker="s", linestyle="--", label="Block-LDL")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("LDL Reasonability Check: BER vs SNR")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    ber_png = os.path.join(cfg.out_dir, "ldl_ber_vs_snr.png")
    plt.savefig(ber_png, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(snr, se_oracle, marker="^", label="Oracle MMSE (True H)")
    plt.plot(snr, se_exact, marker="o", label="Exact MMSE")
    plt.plot(snr, se_ldl, marker="s", linestyle="--", label="Block-LDL")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Spectral Efficiency (bits/s/Hz)")
    plt.title("LDL Reasonability Check: SE vs SNR")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    se_png = os.path.join(cfg.out_dir, "ldl_se_vs_snr.png")
    plt.savefig(se_png, dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(snr, recon, marker="d", color="#8e44ad")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Relative Frobenius Error")
    plt.title("LDL Decomposition Reconstruction Error")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    recon_png = os.path.join(cfg.out_dir, "ldl_recon_error_vs_snr.png")
    plt.savefig(recon_png, dpi=220)
    plt.close()

    ber_gap_max = max(
        abs(row["ber_ldl"] - row["ber_exact_mmse"]) for row in metrics
    )
    se_gap_max = max(
        abs(row["se_ldl"] - row["se_exact_mmse"]) for row in metrics
    )

    status = "PASS" if ber_gap_max < 0.08 and se_gap_max < 2.0 else "CHECK"

    report_path = os.path.join(cfg.out_dir, "ldl_quality_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# LDL 数值合理性评估报告\n\n")
        handle.write("## 评估目标\n")
        handle.write("通过 BER 与频谱效率（SE）对比 `Block-LDL`、`Exact MMSE(估计信道)` 与 `Oracle MMSE(真实信道)`，判断 LDL 算子结果是否在合理范围。\n\n")
        handle.write("## 实验配置\n")
        handle.write(f"- N_r: `{cfg.nr}`\n")
        handle.write(f"- N_t: `{cfg.nt}`\n")
        handle.write(f"- Subcarriers (N_sc): `{cfg.n_sc}`\n")
        handle.write(f"- Batch: `{cfg.batch}`\n")
        handle.write(f"- Trials: `{cfg.trials}`\n")
        handle.write(f"- Block size: `{cfg.block_size}`\n")
        handle.write(f"- Channel model: `{cfg.channel_model}`\n")
        handle.write(f"- Channel estimation: `LS` (pilot_len={cfg.pilot_len})\n")
        handle.write(f"- Numeric format: `{cfg.num_format}`\n")
        handle.write(f"- Reciprocal mode: `{cfg.reciprocal_mode}`\n")
        handle.write(f"- Truncation mantissa bits: `{cfg.trunc_mantissa_bits}`\n")
        handle.write(f"- Modulation: `{cfg.modulation}`\n")
        handle.write(f"- MAC accumulation chunk: `{cfg.mac_chunk}`\n")
        if cfg.pilot_snr_db is None:
            handle.write("- Pilot SNR: `same as data SNR`\n")
        else:
            handle.write(f"- Pilot SNR: `{cfg.pilot_snr_db} dB`\n")
        handle.write(f"- SNR(dB): `{cfg.snr_db_list}`\n")
        handle.write(f"- Seed: `{cfg.seed}`\n\n")
        handle.write("## 结果文件\n")
        handle.write(f"- 指标表: `{os.path.basename(csv_path)}`\n")
        handle.write(f"- BER 图: `{os.path.basename(ber_png)}`\n")
        handle.write(f"- SE 图: `{os.path.basename(se_png)}`\n")
        handle.write(f"- 重构误差图: `{os.path.basename(recon_png)}`\n\n")
        handle.write("## 自动判定\n")
        handle.write(f"- 最大 BER 差值: `{ber_gap_max:.6f}`\n")
        handle.write(f"- 最大 SE 差值: `{se_gap_max:.6f}`\n")
        handle.write(f"- 判定: **{status}**\n\n")
        handle.write("> 注：该判定用于工程 sanity-check，不等价于严格通信算法精度证明。\n"
        )

    return csv_path, ber_png, se_png, recon_png, report_path


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate LDL reasonability using BER/SE.")
    parser.add_argument("--nr", type=int, default=64, help="Receive antennas (N_r)")
    parser.add_argument("--nt", type=int, default=16, help="Transmit antennas/users (N_t)")
    parser.add_argument("--n-sc", type=int, default=168, help="Number of subcarriers (N_sc)")
    parser.add_argument("--batch", type=int, default=96, help="Batch size for reporting")
    parser.add_argument("--trials", type=int, default=20, help="Monte-Carlo trials per batch item")
    parser.add_argument("--block-size", type=int, default=2, help="LDL block size")
    parser.add_argument("--channel-model", type=str, default="rayleigh", help="Channel model (rayleigh)")
    parser.add_argument("--pilot-len", type=int, default=16, help="Pilot length for LS channel estimation")
    parser.add_argument(
        "--pilot-snr-db",
        type=float,
        default=None,
        help="Pilot SNR in dB (default: same as data SNR)",
    )
    parser.add_argument(
        "--num-format",
        type=str,
        default="fp16",
        choices=["fp16", "fp64"],
        help="Arithmetic format emulation",
    )
    parser.add_argument(
        "--reciprocal-mode",
        type=str,
        default="approx",
        choices=["approx", "exact"],
        help="Reciprocal implementation for 2x2 inverse",
    )
    parser.add_argument(
        "--trunc-mantissa-bits",
        type=int,
        default=8,
        help="Mantissa bits kept after FP16 cast (<=10 introduces truncation error)",
    )
    parser.add_argument(
        "--modulation",
        type=str,
        default="16qam",
        choices=["16qam", "bpsk"],
        help="Modulation for BER evaluation",
    )
    parser.add_argument(
        "--mac-chunk",
        type=int,
        default=4,
        help="K-dimension chunk for quantized accumulation emulation",
    )
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20", help="Comma-separated SNR list")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/LDL/quality",
        help="Output directory under LDL results",
    )

    args = parser.parse_args()
    snr_list = [float(token.strip()) for token in args.snr_db.split(",") if token.strip()]

    return EvalConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        block_size=args.block_size,
        snr_db_list=snr_list,
        channel_model=args.channel_model,
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        num_format=args.num_format,
        reciprocal_mode=args.reciprocal_mode,
        trunc_mantissa_bits=args.trunc_mantissa_bits,
        modulation=args.modulation,
        mac_chunk=args.mac_chunk,
        seed=args.seed,
        out_dir=args.out_dir,
    )


def main():
    cfg = parse_args()
    if cfg.nt % cfg.block_size != 0:
        raise SystemExit(f"nt ({cfg.nt}) must be divisible by block_size ({cfg.block_size})")

    metrics = run_eval(cfg)
    outputs = save_outputs(cfg, metrics)

    print("LDL quality evaluation finished.")
    print("Generated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
