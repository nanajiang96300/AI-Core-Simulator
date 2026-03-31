#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List
import os
import sys

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
class BerConfig:
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


def generate_channel(rng: np.random.Generator, nr: int, nt: int) -> np.ndarray:
    return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)


def cholesky_iso_factor(a_mat: np.ndarray) -> np.ndarray:
    n = a_mat.shape[0]
    a_work = a_mat.astype(np.complex128, copy=False)
    l_mat = np.zeros_like(a_work, dtype=np.complex128)

    for j in range(n):
        if j > 0:
            diag_acc = np.dot(l_mat[j, :j], l_mat[j, :j].conj())
        else:
            diag_acc = 0.0

        l_jj = np.sqrt(a_work[j, j] - diag_acc)
        l_mat[j, j] = l_jj
        inv_l_jj = 1.0 / l_jj

        for i in range(j + 1, n):
            if j > 0:
                off_acc = np.dot(l_mat[i, :j], l_mat[j, :j].conj())
            else:
                off_acc = 0.0
            l_mat[i, j] = (a_work[i, j] - off_acc) * inv_l_jj

    return l_mat


def cholesky_iso_inverse(a_mat: np.ndarray) -> np.ndarray:
    l_mat = cholesky_iso_factor(a_mat)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    l_inv = np.linalg.solve(l_mat, identity)
    return l_inv.conj().T @ l_inv


def cholesky_exact_inverse(a_mat: np.ndarray) -> np.ndarray:
    l_mat = np.linalg.cholesky(a_mat)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    y_mat = np.linalg.solve(l_mat, identity)
    return np.linalg.solve(l_mat.conj().T, y_mat)


def run_ber(cfg: BerConfig):
    rng = np.random.default_rng(cfg.seed)

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
        num_format="fp32",
        reciprocal_mode="exact",
        trunc_mantissa_bits=23,
        modulation="16qam",
        mac_chunk=1024,
        seed=cfg.seed,
        out_dir="results/CHOL/falsification",
    )

    print("=== BER Validation: Cholesky-ISO flow vs Exact vs LDL ===")
    print(
        f"nr={cfg.nr}, nt={cfg.nt}, n_sc={cfg.n_sc}, batch={cfg.batch}, trials={cfg.trials}, "
        f"pilot_len={cfg.pilot_len}, block_size={cfg.block_size}"
    )

    for snr_db in cfg.snr_db_list:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        ber_err_exact = 0
        ber_err_iso = 0
        ber_err_ldl = 0
        total_bits = 0

        se_exact = 0.0
        se_iso = 0.0
        se_ldl = 0.0

        inv_relerr_iso_sum = 0.0
        inv_relerr_ldl_sum = 0.0

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

            a_inv_exact = cholesky_exact_inverse(a_est)
            a_inv_iso = cholesky_iso_inverse(a_est)
            a_inv_ldl, _ = ldl_inverse(a_est, ldl_cfg, block_size=cfg.block_size)

            relerr_iso = np.linalg.norm(a_inv_exact - a_inv_iso, ord="fro") / (
                np.linalg.norm(a_inv_exact, ord="fro") + 1e-12
            )
            relerr_ldl = np.linalg.norm(a_inv_exact - a_inv_ldl, ord="fro") / (
                np.linalg.norm(a_inv_exact, ord="fro") + 1e-12
            )
            inv_relerr_iso_sum += relerr_iso
            inv_relerr_ldl_sum += relerr_ldl

            w_exact = a_inv_exact @ h_h_est
            w_iso = a_inv_iso @ h_h_est
            w_ldl = a_inv_ldl @ h_h_est

            xhat_exact = w_exact @ y_rx
            xhat_iso = w_iso @ y_rx
            xhat_ldl = w_ldl @ y_rx

            bits_hat_exact = demod_16qam(xhat_exact)
            bits_hat_iso = demod_16qam(xhat_iso)
            bits_hat_ldl = demod_16qam(xhat_ldl)

            ber_err_exact += int(np.sum(bits_hat_exact != tx_bits))
            ber_err_iso += int(np.sum(bits_hat_iso != tx_bits))
            ber_err_ldl += int(np.sum(bits_hat_ldl != tx_bits))
            total_bits += cfg.nt * 4

            se_exact += estimate_se(w_exact, h_true, noise_var)
            se_iso += estimate_se(w_iso, h_true, noise_var)
            se_ldl += estimate_se(w_ldl, h_true, noise_var)

        ber_exact = ber_err_exact / total_bits
        ber_iso = ber_err_iso / total_bits
        ber_ldl = ber_err_ldl / total_bits

        avg_relerr_iso = inv_relerr_iso_sum / total_samples
        avg_relerr_ldl = inv_relerr_ldl_sum / total_samples

        avg_se_exact = se_exact / total_samples
        avg_se_iso = se_iso / total_samples
        avg_se_ldl = se_ldl / total_samples

        print(
            f"SNR={snr_db:>5.1f} dB | "
            f"BER exact/iso/ldl = {ber_exact:.6e}/{ber_iso:.6e}/{ber_ldl:.6e} | "
            f"SE exact/iso/ldl = {avg_se_exact:.4f}/{avg_se_iso:.4f}/{avg_se_ldl:.4f} | "
            f"inv_relerr iso/ldl = {avg_relerr_iso:.3e}/{avg_relerr_ldl:.3e}"
        )


def parse_args() -> BerConfig:
    parser = argparse.ArgumentParser(description="Validate Cholesky ISO operator flow with BER")
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return BerConfig(
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
    )


if __name__ == "__main__":
    config = parse_args()
    run_ber(config)
