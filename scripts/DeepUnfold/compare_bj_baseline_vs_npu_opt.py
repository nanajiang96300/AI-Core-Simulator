#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.DeepUnfold.bj_deep_unfolding_onnxim import (  # noqa: E402
    bj_chebyshev_inverse,
    build_regularized_system,
    generate_system,
    make_square_qam_constellation,
)
from scripts.DeepUnfold.bj_deep_unfolding_npu_opt import (  # noqa: E402
    NPUOptConfig,
    bj_chebyshev_inverse_npu_opt,
)
from scripts.DeepUnfold.bj_deep_unfolding_npu_opt_overlap import (  # noqa: E402
    NPUOptOverlapConfig,
    bj_deep_unfolding_detect_npu_opt_overlap,
)
from scripts.evaluate_ldl_quality import (  # noqa: E402
    EvalConfig,
    bits_to_16qam,
    demod_16qam,
    estimate_se,
    ldl_inverse,
    ls_channel_estimate,
)


@dataclass
class CompareConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    block_size: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    modulation: str
    seed: int
    out_dir: str

    npu_layers: int
    npu_block: int
    npu_adaptive_bounds: bool
    npu_tile_m: int
    npu_tile_n: int
    npu_tile_k: int
    npu_overlap_h_chunks: int
    npu_overlap_y_chunks: int


def cholesky_inverse(a_mat: np.ndarray) -> np.ndarray:
    n = a_mat.shape[0]
    identity = np.eye(n, dtype=np.complex128)
    l_mat = np.linalg.cholesky(a_mat)
    y_mat = np.linalg.solve(l_mat, identity)
    x_mat = np.linalg.solve(l_mat.conj().T, y_mat)
    return x_mat


def run_compare(cfg: CompareConfig) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    constellation = make_square_qam_constellation(16)

    ldl_cfg = EvalConfig(
        nr=cfg.nr,
        nt=cfg.nt,
        n_sc=cfg.n_sc,
        batch=cfg.batch,
        trials=cfg.trials,
        block_size=cfg.block_size,
        snr_db_list=cfg.snr_db_list,
        channel_model="rayleigh",
        pilot_len=cfg.pilot_len,
        pilot_snr_db=cfg.pilot_snr_db,
        num_format="fp16",
        reciprocal_mode="approx",
        trunc_mantissa_bits=8,
        modulation=cfg.modulation,
        mac_chunk=4,
        seed=cfg.seed,
        out_dir=cfg.out_dir,
    )

    npu_cfg = NPUOptConfig(
        n_layers=cfg.npu_layers,
        blk=cfg.npu_block,
        adaptive_bounds=cfg.npu_adaptive_bounds,
        tile_m=cfg.npu_tile_m,
        tile_n=cfg.npu_tile_n,
        tile_k=cfg.npu_tile_k,
        symmetrize_each_layer=True,
    )

    npu_overlap_cfg = NPUOptOverlapConfig(
        n_layers=cfg.npu_layers,
        blk=cfg.npu_block,
        adaptive_bounds=cfg.npu_adaptive_bounds,
        tile_m=cfg.npu_tile_m,
        tile_n=cfg.npu_tile_n,
        tile_k=cfg.npu_tile_k,
        symmetrize_each_layer=True,
        h_chunks=cfg.npu_overlap_h_chunks,
        y_chunks=cfg.npu_overlap_y_chunks,
    )

    metrics: List[Dict[str, float]] = []
    t_global = time.time()

    for index, snr_db in enumerate(cfg.snr_db_list, start=1):
        t_snr = time.time()
        print(f"[progress] SNR {snr_db} dB ({index}/{len(cfg.snr_db_list)}) started...", flush=True)

        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        err_bits_chol = 0
        err_bits_ldl = 0
        err_bits_bj_base = 0
        err_bits_bj_npu = 0
        err_bits_bj_npu_overlap = 0
        total_bits = 0

        se_chol_sum = 0.0
        se_ldl_sum = 0.0
        se_bj_base_sum = 0.0
        se_bj_npu_sum = 0.0
        se_bj_npu_overlap_sum = 0.0

        t_bj_base = 0.0
        t_bj_npu = 0.0
        t_bj_npu_overlap = 0.0
        npu_gemm_calls_sum = 0
        npu_vector_calls_sum = 0
        npu_overlap_gemm_calls_sum = 0
        npu_overlap_vector_calls_sum = 0

        total_samples = cfg.trials * cfg.batch * cfg.n_sc

        for _ in range(total_samples):
            h_true, _s, _y_unused, _noise_unused = generate_system(
                rng,
                n_rx=cfg.nr,
                n_stream=cfg.nt,
                constellation=constellation,
                snr_db=snr_db,
            )
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            if cfg.modulation == "16qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
                x_tx = bits_to_16qam(tx_bits)
                bits_per_sym = 4
            else:
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 1), dtype=np.int32)
                x_tx = (2 * tx_bits[:, 0] - 1).astype(np.float64).astype(np.complex128)
                bits_per_sym = 1

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y_rx = h_true @ x_tx + noise

            _, a_est = build_regularized_system(h_est, noise_var, constellation)
            h_h_est = h_est.conj().T

            a_inv_chol = cholesky_inverse(a_est)
            a_inv_ldl, _ = ldl_inverse(a_est, ldl_cfg, block_size=cfg.block_size)

            t0 = time.perf_counter()
            a_inv_bj_base = bj_chebyshev_inverse(
                a_est,
                n_layers=cfg.npu_layers,
                adaptive_bounds=cfg.npu_adaptive_bounds,
            )
            t_bj_base += time.perf_counter() - t0

            t1 = time.perf_counter()
            a_inv_bj_npu, _stats = bj_chebyshev_inverse_npu_opt(a_est, npu_cfg)
            t_bj_npu += time.perf_counter() - t1
            npu_gemm_calls_sum += int(_stats.gemm_calls)
            npu_vector_calls_sum += int(_stats.vector_calls)

            t2 = time.perf_counter()
            overlap_out = bj_deep_unfolding_detect_npu_opt_overlap(
                h_est,
                y_rx,
                noise_var,
                constellation,
                npu_overlap_cfg,
            )
            t_bj_npu_overlap += time.perf_counter() - t2
            overlap_stats = overlap_out["stats"]
            npu_overlap_gemm_calls_sum += int(overlap_stats.gemm_calls)
            npu_overlap_vector_calls_sum += int(overlap_stats.vector_calls)
            a_inv_bj_npu_overlap = overlap_out["a_inv"]

            w_chol = a_inv_chol @ h_h_est
            w_ldl = a_inv_ldl @ h_h_est
            w_bj_base = a_inv_bj_base @ h_h_est
            w_bj_npu = a_inv_bj_npu @ h_h_est
            w_bj_npu_overlap = a_inv_bj_npu_overlap @ h_h_est

            xhat_chol = w_chol @ y_rx
            xhat_ldl = w_ldl @ y_rx
            xhat_bj_base = w_bj_base @ y_rx
            xhat_bj_npu = w_bj_npu @ y_rx
            xhat_bj_npu_overlap = w_bj_npu_overlap @ y_rx

            if cfg.modulation == "16qam":
                bits_hat_chol = demod_16qam(xhat_chol)
                bits_hat_ldl = demod_16qam(xhat_ldl)
                bits_hat_bj_base = demod_16qam(xhat_bj_base)
                bits_hat_bj_npu = demod_16qam(xhat_bj_npu)
                bits_hat_bj_npu_overlap = demod_16qam(xhat_bj_npu_overlap)
            else:
                bits_hat_chol = (np.real(xhat_chol) >= 0).astype(np.int32)[:, None]
                bits_hat_ldl = (np.real(xhat_ldl) >= 0).astype(np.int32)[:, None]
                bits_hat_bj_base = (np.real(xhat_bj_base) >= 0).astype(np.int32)[:, None]
                bits_hat_bj_npu = (np.real(xhat_bj_npu) >= 0).astype(np.int32)[:, None]
                bits_hat_bj_npu_overlap = (
                    (np.real(xhat_bj_npu_overlap) >= 0).astype(np.int32)[:, None]
                )

            err_bits_chol += int(np.sum(bits_hat_chol != tx_bits))
            err_bits_ldl += int(np.sum(bits_hat_ldl != tx_bits))
            err_bits_bj_base += int(np.sum(bits_hat_bj_base != tx_bits))
            err_bits_bj_npu += int(np.sum(bits_hat_bj_npu != tx_bits))
            err_bits_bj_npu_overlap += int(np.sum(bits_hat_bj_npu_overlap != tx_bits))
            total_bits += cfg.nt * bits_per_sym

            se_chol_sum += estimate_se(w_chol, h_true, noise_var)
            se_ldl_sum += estimate_se(w_ldl, h_true, noise_var)
            se_bj_base_sum += estimate_se(w_bj_base, h_true, noise_var)
            se_bj_npu_sum += estimate_se(w_bj_npu, h_true, noise_var)
            se_bj_npu_overlap_sum += estimate_se(w_bj_npu_overlap, h_true, noise_var)

        ber_chol = err_bits_chol / max(total_bits, 1)
        ber_ldl = err_bits_ldl / max(total_bits, 1)
        ber_bj_base = err_bits_bj_base / max(total_bits, 1)
        ber_bj_npu = err_bits_bj_npu / max(total_bits, 1)
        ber_bj_npu_overlap = err_bits_bj_npu_overlap / max(total_bits, 1)

        se_chol = se_chol_sum / max(total_samples, 1)
        se_ldl = se_ldl_sum / max(total_samples, 1)
        se_bj_base = se_bj_base_sum / max(total_samples, 1)
        se_bj_npu = se_bj_npu_sum / max(total_samples, 1)
        se_bj_npu_overlap = se_bj_npu_overlap_sum / max(total_samples, 1)

        metrics.append(
            {
                "snr_db": snr_db,
                "ber_cholesky": ber_chol,
                "ber_ldl": ber_ldl,
                "ber_bj_baseline": ber_bj_base,
                "ber_bj_npu_opt": ber_bj_npu,
                "ber_bj_npu_opt_overlap": ber_bj_npu_overlap,
                "se_cholesky": se_chol,
                "se_ldl": se_ldl,
                "se_bj_baseline": se_bj_base,
                "se_bj_npu_opt": se_bj_npu,
                "se_bj_npu_opt_overlap": se_bj_npu_overlap,
                "ber_gap_npu_vs_base": ber_bj_npu - ber_bj_base,
                "se_gap_npu_vs_base": se_bj_npu - se_bj_base,
                "ber_gap_npu_overlap_vs_npu": ber_bj_npu_overlap - ber_bj_npu,
                "se_gap_npu_overlap_vs_npu": se_bj_npu_overlap - se_bj_npu,
                "time_ms_bj_baseline": 1000.0 * t_bj_base / max(total_samples, 1),
                "time_ms_bj_npu_opt": 1000.0 * t_bj_npu / max(total_samples, 1),
                "time_ms_bj_npu_opt_overlap": 1000.0 * t_bj_npu_overlap / max(total_samples, 1),
                "npu_gemm_calls_per_sample": npu_gemm_calls_sum / max(total_samples, 1),
                "npu_vector_calls_per_sample": npu_vector_calls_sum / max(total_samples, 1),
                "npu_overlap_gemm_calls_per_sample": npu_overlap_gemm_calls_sum
                / max(total_samples, 1),
                "npu_overlap_vector_calls_per_sample": npu_overlap_vector_calls_sum
                / max(total_samples, 1),
            }
        )

        print(
            f"[progress] SNR {snr_db} dB done in {time.time() - t_snr:.2f}s, elapsed {time.time() - t_global:.2f}s",
            flush=True,
        )

    return metrics


def save_outputs(cfg: CompareConfig, metrics: List[Dict[str, float]]) -> tuple[str, str, str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_path = os.path.join(cfg.out_dir, "bj_baseline_vs_npu_opt_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    snr = [row["snr_db"] for row in metrics]
    ber_chol = [row["ber_cholesky"] for row in metrics]
    ber_ldl = [row["ber_ldl"] for row in metrics]
    ber_base = [row["ber_bj_baseline"] for row in metrics]
    ber_npu = [row["ber_bj_npu_opt"] for row in metrics]
    ber_npu_overlap = [row["ber_bj_npu_opt_overlap"] for row in metrics]

    se_chol = [row["se_cholesky"] for row in metrics]
    se_ldl = [row["se_ldl"] for row in metrics]
    se_base = [row["se_bj_baseline"] for row in metrics]
    se_npu = [row["se_bj_npu_opt"] for row in metrics]
    se_npu_overlap = [row["se_bj_npu_opt_overlap"] for row in metrics]

    eps = 1e-8
    ber_chol_plot = [max(v, eps) for v in ber_chol]
    ber_ldl_plot = [max(v, eps) for v in ber_ldl]
    ber_base_plot = [max(v, eps) for v in ber_base]
    ber_npu_plot = [max(v, eps) for v in ber_npu]
    ber_npu_overlap_plot = [max(v, eps) for v in ber_npu_overlap]

    ber_png = os.path.join(cfg.out_dir, "ber_vs_snr_bj_baseline_npu_opt.png")
    plt.figure(figsize=(8, 5.2))
    plt.semilogy(snr, ber_chol_plot, marker="o", label="Cholesky")
    plt.semilogy(snr, ber_ldl_plot, marker="s", linestyle="--", label="LDL")
    plt.semilogy(snr, ber_base_plot, marker="^", linestyle="-.", label="BJ Baseline")
    plt.semilogy(snr, ber_npu_plot, marker="d", linestyle="-", label="BJ NPU-Opt")
    plt.semilogy(
        snr,
        ber_npu_overlap_plot,
        marker="x",
        linestyle="-",
        label="BJ NPU-Opt Overlap",
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER Comparison")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_png, dpi=220)
    plt.close()

    se_png = os.path.join(cfg.out_dir, "se_vs_snr_bj_baseline_npu_opt.png")
    plt.figure(figsize=(8, 5.2))
    plt.plot(snr, se_chol, marker="o", label="Cholesky")
    plt.plot(snr, se_ldl, marker="s", linestyle="--", label="LDL")
    plt.plot(snr, se_base, marker="^", linestyle="-.", label="BJ Baseline")
    plt.plot(snr, se_npu, marker="d", linestyle="-", label="BJ NPU-Opt")
    plt.plot(snr, se_npu_overlap, marker="x", linestyle="-", label="BJ NPU-Opt Overlap")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE (bits/s/Hz)")
    plt.title("SE Comparison")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(se_png, dpi=220)
    plt.close()

    time_png = os.path.join(cfg.out_dir, "time_per_sample_ms_bj_baseline_npu_opt.png")
    time_base = [row["time_ms_bj_baseline"] for row in metrics]
    time_npu = [row["time_ms_bj_npu_opt"] for row in metrics]
    time_npu_overlap = [row["time_ms_bj_npu_opt_overlap"] for row in metrics]
    x_idx = np.arange(len(snr))
    width = 0.25

    plt.figure(figsize=(8, 5.2))
    plt.bar(x_idx - width, time_base, width=width, label="BJ Baseline")
    plt.bar(x_idx, time_npu, width=width, label="BJ NPU-Opt")
    plt.bar(x_idx + width, time_npu_overlap, width=width, label="BJ NPU-Opt Overlap")
    plt.xticks(x_idx, [str(int(v)) for v in snr])
    plt.xlabel("SNR (dB)")
    plt.ylabel("Time per sample (ms)")
    plt.title("Runtime Proxy (Python)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(time_png, dpi=220)
    plt.close()

    report_path = os.path.join(cfg.out_dir, "bj_baseline_vs_npu_opt_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# BJ Baseline vs NPU-Optimized 对比\n\n")
        handle.write("- 说明：优化版为新增文件，不覆盖基线实现。\n")
        handle.write("- 精度目标：BER/SE 与基线一致，且不劣于 Cholesky/LDL 趋势。\n")
        handle.write("- 性能目标：NPU形态（分块GEMM+Vector更新）便于映射与统计。\n")
        handle.write("- Overlap 版本：按算子做 H/Y 分块（逻辑等价）用于正确性校验。\n\n")
        handle.write(f"- metrics: `{os.path.basename(csv_path)}`\n")
        handle.write(f"- ber fig: `{os.path.basename(ber_png)}`\n")
        handle.write(f"- se fig: `{os.path.basename(se_png)}`\n")
        handle.write(f"- time fig: `{os.path.basename(time_png)}`\n")

    return csv_path, ber_png, se_png, time_png


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(description="Compare BJ baseline vs NPU-optimized version.")

    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--modulation", type=str, default="16qam", choices=["16qam", "bpsk"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/DeepUnfold/bj_npu_opt_compare")

    parser.add_argument("--npu-layers", type=int, default=12)
    parser.add_argument("--npu-block", type=int, default=4)
    parser.add_argument("--npu-adaptive-bounds", action="store_true")
    parser.add_argument("--npu-tile-m", type=int, default=16)
    parser.add_argument("--npu-tile-n", type=int, default=16)
    parser.add_argument("--npu-tile-k", type=int, default=16)
    parser.add_argument("--npu-overlap-h-chunks", type=int, default=2)
    parser.add_argument("--npu-overlap-y-chunks", type=int, default=2)

    args = parser.parse_args()
    snr_list = [float(token.strip()) for token in args.snr_db.split(",") if token.strip()]

    return CompareConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        block_size=args.block_size,
        snr_db_list=snr_list,
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        modulation=args.modulation,
        seed=args.seed,
        out_dir=args.out_dir,
        npu_layers=args.npu_layers,
        npu_block=args.npu_block,
        npu_adaptive_bounds=args.npu_adaptive_bounds,
        npu_tile_m=args.npu_tile_m,
        npu_tile_n=args.npu_tile_n,
        npu_tile_k=args.npu_tile_k,
        npu_overlap_h_chunks=args.npu_overlap_h_chunks,
        npu_overlap_y_chunks=args.npu_overlap_y_chunks,
    )


def main() -> None:
    cfg = parse_args()
    metrics = run_compare(cfg)
    outputs = save_outputs(cfg, metrics)

    print("BJ baseline vs NPU-opt comparison finished.")
    print("Generated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
