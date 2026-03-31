#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


Array = np.ndarray


def make_square_qam_constellation(order: int) -> Array:
    side = int(np.sqrt(order))
    if side * side != order:
        raise ValueError(f"Only square QAM is supported, got order={order}")
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    xv, yv = np.meshgrid(levels, levels)
    constellation = xv.reshape(-1) + 1j * yv.reshape(-1)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    return constellation.astype(np.complex128)


def average_symbol_energy(constellation: Array) -> float:
    return float(np.mean(np.abs(constellation) ** 2))


def matched_filter(H: Array, y: Array) -> Array:
    return H.conj().T @ y


def hard_demod(z: Array, constellation: Array) -> Array:
    dist2 = np.abs(z[:, None] - constellation[None, :]) ** 2
    idx = np.argmin(dist2, axis=1)
    return constellation[idx]


def generate_system(
    rng: np.random.Generator,
    n_rx: int,
    n_stream: int,
    constellation: Array,
    snr_db: float,
) -> Tuple[Array, Array, Array, float]:
    H = (
        rng.standard_normal((n_rx, n_stream)) + 1j * rng.standard_normal((n_rx, n_stream))
    ) / np.sqrt(2.0 * n_rx)
    s = constellation[rng.integers(0, len(constellation), size=n_stream)]
    noise_var = 10.0 ** (-snr_db / 10.0)
    n = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx)
    )
    y = H @ s + n
    return H.astype(np.complex128), s.astype(np.complex128), y.astype(np.complex128), noise_var


def chebyshev_omega(n_layers: int, bmin: float = 0.1, bmax: float = 1.2) -> List[float]:
    omegas: List[float] = []
    for t in range(n_layers):
        theta = np.pi * (2 * t + 1) / (2 * n_layers)
        dt = 0.5 * (bmax + bmin) + 0.5 * (bmax - bmin) * np.cos(theta)
        omegas.append(float(1.0 / dt))
    return omegas


def chebyshev_omega_adaptive(b_mat: Array, n_layers: int, floor: float = 1e-8) -> List[float]:
    eigvals = np.linalg.eigvalsh(b_mat)
    bmax = float(np.max(eigvals).real)
    bmin = max(float(np.min(eigvals).real), floor)
    return chebyshev_omega(n_layers=n_layers, bmin=bmin, bmax=bmax)


def build_regularized_system(H: Array, noise_var: float, constellation: Array) -> Tuple[Array, Array]:
    symbol_energy = average_symbol_energy(constellation)
    gram = H.conj().T @ H
    a_mat = gram + (noise_var / symbol_energy) * np.eye(H.shape[1], dtype=np.complex128)
    return gram, a_mat


def build_block_jacobi_preconditioner(a_mat: Array, blk: int = 4) -> Tuple[Array, Array]:
    ns = a_mat.shape[0]
    m_half_inv = np.zeros_like(a_mat, dtype=np.complex128)
    n_blk = ns // blk
    remainder = ns % blk

    for block_id in range(n_blk):
        start, stop = block_id * blk, (block_id + 1) * blk
        block = a_mat[start:stop, start:stop]
        eigvals, eigvecs = np.linalg.eigh(block)
        eigvals = np.clip(eigvals, 1e-12, None)
        m_half_inv[start:stop, start:stop] = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.conj().T

    if remainder > 0:
        start = n_blk * blk
        block = a_mat[start:, start:]
        eigvals, eigvecs = np.linalg.eigh(block)
        eigvals = np.clip(eigvals, 1e-12, None)
        m_half_inv[start:, start:] = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.conj().T

    b_mat = m_half_inv @ a_mat @ m_half_inv
    return b_mat, m_half_inv


def bj_chebyshev_inverse(a_mat: Array, n_layers: int = 12, adaptive_bounds: bool = False) -> Array:
    b_mat, m_half_inv = build_block_jacobi_preconditioner(a_mat, blk=4)
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    omegas = (
        chebyshev_omega_adaptive(b_mat, n_layers=n_layers)
        if adaptive_bounds
        else chebyshev_omega(n_layers=n_layers, bmin=0.1, bmax=1.2)
    )
    for omega in omegas:
        y_mat = y_mat + omega * (identity - b_mat @ y_mat)
    return m_half_inv @ y_mat @ m_half_inv


def bj_deep_unfolding_detect(
    H: Array,
    y: Array,
    noise_var: float,
    constellation: Array,
    n_layers: int = 12,
    adaptive_bounds: bool = False,
) -> Dict[str, Array]:
    gram, a_mat = build_regularized_system(H, noise_var, constellation)
    x_approx = bj_chebyshev_inverse(a_mat, n_layers=n_layers, adaptive_bounds=adaptive_bounds)
    y_mf = matched_filter(H, y)
    z = x_approx @ y_mf
    x_hat = hard_demod(z, constellation)
    return {
        "z": z,
        "x_hat": x_hat,
        "gram": gram,
        "a_mat": a_mat,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    constellation = make_square_qam_constellation(16)
    H, s, y, noise_var = generate_system(rng, n_rx=64, n_stream=8, constellation=constellation, snr_db=10.0)
    out_fixed = bj_deep_unfolding_detect(H, y, noise_var, constellation, n_layers=12, adaptive_bounds=False)
    out_adapt = bj_deep_unfolding_detect(H, y, noise_var, constellation, n_layers=12, adaptive_bounds=True)
    ser_fixed = float(np.mean(out_fixed["x_hat"] != s))
    ser_adapt = float(np.mean(out_adapt["x_hat"] != s))
    print(f"BJ-DetNet fixed-bound SER : {ser_fixed:.4f}")
    print(f"BJ-DetNet adaptive SER    : {ser_adapt:.4f}")


if __name__ == "__main__":
    main()