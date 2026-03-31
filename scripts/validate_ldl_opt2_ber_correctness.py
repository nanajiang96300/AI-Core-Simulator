#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_ldl_quality import (
    EvalConfig,
    bits_to_16qam,
    demod_16qam,
    estimate_se,
    ldl_inverse,
    ls_channel_estimate,
)


@dataclass
class BerCfg:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    block_size: int
    seed: int
    out_csv: str


def generate_channel(rng: np.random.Generator, nr: int, nt: int) -> np.ndarray:
    return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)


def exact_mmse_inverse(a_mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a_mat)


def ldl_inverse_operator_mode(a_mat: np.ndarray, cfg: EvalConfig, block_size: int, mode: str) -> np.ndarray:
    if mode not in {"old", "opt2"}:
        raise ValueError(f"Unsupported mode: {mode}")
    a_inv, _ = ldl_inverse(a_mat, cfg, block_size=block_size)
    return a_inv


def run_ber(cfg: BerCfg):
    rng = np.random.default_rng(cfg.seed)

    eval_cfg = EvalConfig(
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
        num_format="fp32",
        reciprocal_mode="exact",
        trunc_mantissa_bits=23,
        modulation="16qam",
        mac_chunk=1024,
        seed=cfg.seed,
        out_dir="results/LDL/falsification",
    )

    rows = []
    print("=== BER Validation: LDL old vs LDL opt2 vs Exact ===")
    print(
        f"nr={cfg.nr}, nt={cfg.nt}, n_sc={cfg.n_sc}, batch={cfg.batch}, trials={cfg.trials}, "
        f"pilot_len={cfg.pilot_len}, block_size={cfg.block_size}, seed={cfg.seed}"
    )

    for snr_db in cfg.snr_db_list:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        err_exact = 0
        err_ldl_old = 0
        err_ldl_opt2 = 0
        total_bits = 0

        se_exact = 0.0
        se_ldl_old = 0.0
        se_ldl_opt2 = 0.0

        relerr_old_sum = 0.0
        relerr_opt2_sum = 0.0
        old_opt2_gap_sum = 0.0

        total_samples = cfg.trials * cfg.batch * cfg.n_sc

        for _ in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
            x_tx = bits_to_16qam(tx_bits)

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y_rx = h_true @ x_tx + noise

            a_est = h_est.conj().T @ h_est + noise_var * np.eye(cfg.nt, dtype=np.complex128)
            h_h_est = h_est.conj().T

            a_inv_exact = exact_mmse_inverse(a_est)
            a_inv_old = ldl_inverse_operator_mode(a_est, eval_cfg, cfg.block_size, mode="old")
            a_inv_opt2 = ldl_inverse_operator_mode(a_est, eval_cfg, cfg.block_size, mode="opt2")

            relerr_old = np.linalg.norm(a_inv_exact - a_inv_old, ord="fro") / (
                np.linalg.norm(a_inv_exact, ord="fro") + 1e-12
            )
            relerr_opt2 = np.linalg.norm(a_inv_exact - a_inv_opt2, ord="fro") / (
                np.linalg.norm(a_inv_exact, ord="fro") + 1e-12
            )
            old_opt2_gap = np.linalg.norm(a_inv_old - a_inv_opt2, ord="fro") / (
                np.linalg.norm(a_inv_old, ord="fro") + 1e-12
            )

            relerr_old_sum += relerr_old
            relerr_opt2_sum += relerr_opt2
            old_opt2_gap_sum += old_opt2_gap

            w_exact = a_inv_exact @ h_h_est
            w_old = a_inv_old @ h_h_est
            w_opt2 = a_inv_opt2 @ h_h_est

            xhat_exact = w_exact @ y_rx
            xhat_old = w_old @ y_rx
            xhat_opt2 = w_opt2 @ y_rx

            bits_hat_exact = demod_16qam(xhat_exact)
            bits_hat_old = demod_16qam(xhat_old)
            bits_hat_opt2 = demod_16qam(xhat_opt2)

            err_exact += int(np.sum(bits_hat_exact != tx_bits))
            err_ldl_old += int(np.sum(bits_hat_old != tx_bits))
            err_ldl_opt2 += int(np.sum(bits_hat_opt2 != tx_bits))
            total_bits += cfg.nt * 4

            se_exact += estimate_se(w_exact, h_true, noise_var)
            se_ldl_old += estimate_se(w_old, h_true, noise_var)
            se_ldl_opt2 += estimate_se(w_opt2, h_true, noise_var)

        ber_exact = err_exact / total_bits
        ber_old = err_ldl_old / total_bits
        ber_opt2 = err_ldl_opt2 / total_bits

        avg_se_exact = se_exact / total_samples
        avg_se_old = se_ldl_old / total_samples
        avg_se_opt2 = se_ldl_opt2 / total_samples

        avg_relerr_old = relerr_old_sum / total_samples
        avg_relerr_opt2 = relerr_opt2_sum / total_samples
        avg_old_opt2_gap = old_opt2_gap_sum / total_samples

        row = {
            "snr_db": snr_db,
            "ber_exact": ber_exact,
            "ber_ldl_old": ber_old,
            "ber_ldl_opt2": ber_opt2,
            "se_exact": avg_se_exact,
            "se_ldl_old": avg_se_old,
            "se_ldl_opt2": avg_se_opt2,
            "inv_relerr_old_vs_exact": avg_relerr_old,
            "inv_relerr_opt2_vs_exact": avg_relerr_opt2,
            "inv_relerr_old_vs_opt2": avg_old_opt2_gap,
        }
        rows.append(row)

        print(
            f"SNR={snr_db:>5.1f} dB | "
            f"BER exact/old/opt2 = {ber_exact:.6e}/{ber_old:.6e}/{ber_opt2:.6e} | "
            f"SE exact/old/opt2 = {avg_se_exact:.4f}/{avg_se_old:.4f}/{avg_se_opt2:.4f} | "
            f"inv_relerr old/exact={avg_relerr_old:.3e}, opt2/exact={avg_relerr_opt2:.3e}, old/opt2={avg_old_opt2_gap:.3e}"
        )

    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    with open(cfg.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {cfg.out_csv}")


def parse_args() -> BerCfg:
    parser = argparse.ArgumentParser(description="Validate BER correctness for LDL old/opt2 operator modes")
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/LDL/falsification/ldl_old_opt2_ber_validation_20260327.csv",
    )
    args = parser.parse_args()

    return BerCfg(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        snr_db_list=[float(x.strip()) for x in args.snr_db.split(",") if x.strip()],
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        block_size=args.block_size,
        seed=args.seed,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_ber(cfg)
