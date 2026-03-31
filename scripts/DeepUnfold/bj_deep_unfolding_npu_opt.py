#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class NPUOptConfig:
    n_layers: int = 12
    blk: int = 4
    adaptive_bounds: bool = True
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    symmetrize_each_layer: bool = True


@dataclass
class NPUPerfStats:
    gemm_calls: int = 0
    vector_calls: int = 0


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


def generate_system(
    rng: np.random.Generator,
    n_rx: int,
    n_stream: int,
    constellation: Array,
    snr_db: float,
) -> Tuple[Array, Array, Array, float]:
    h_mat = (
        rng.standard_normal((n_rx, n_stream)) + 1j * rng.standard_normal((n_rx, n_stream))
    ) / np.sqrt(2.0 * n_rx)
    s_vec = constellation[rng.integers(0, len(constellation), size=n_stream)]
    noise_var = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_var / 2.0) * (
        rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx)
    )
    y_vec = h_mat @ s_vec + noise
    return h_mat.astype(np.complex128), s_vec.astype(np.complex128), y_vec.astype(np.complex128), float(noise_var)


def matched_filter(h_mat: Array, y_vec: Array) -> Array:
    return h_mat.conj().T @ y_vec


def hard_demod(z_vec: Array, constellation: Array) -> Array:
    dist2 = np.abs(z_vec[:, None] - constellation[None, :]) ** 2
    indices = np.argmin(dist2, axis=1)
    return constellation[indices]


def build_regularized_system(h_mat: Array, noise_var: float, constellation: Array) -> Tuple[Array, Array]:
    symbol_energy = average_symbol_energy(constellation)
    gram = h_mat.conj().T @ h_mat
    a_mat = gram + (noise_var / symbol_energy) * np.eye(h_mat.shape[1], dtype=np.complex128)
    return gram, a_mat


def chebyshev_omega(n_layers: int, bmin: float = 0.1, bmax: float = 1.2) -> List[float]:
    omegas: List[float] = []
    for idx in range(n_layers):
        theta = np.pi * (2 * idx + 1) / (2 * n_layers)
        dt = 0.5 * (bmax + bmin) + 0.5 * (bmax - bmin) * np.cos(theta)
        omegas.append(float(1.0 / dt))
    return omegas


def chebyshev_omega_adaptive(b_mat: Array, n_layers: int, floor: float = 1e-8) -> List[float]:
    eigvals = np.linalg.eigvalsh(b_mat)
    bmax = float(np.max(eigvals).real)
    bmin = max(float(np.min(eigvals).real), floor)
    return chebyshev_omega(n_layers=n_layers, bmin=bmin, bmax=bmax)


def build_block_jacobi_preconditioner(a_mat: Array, blk: int = 4) -> Tuple[Array, Array]:
    n_stream = a_mat.shape[0]
    m_half_inv = np.zeros_like(a_mat, dtype=np.complex128)
    n_blk = n_stream // blk
    remainder = n_stream % blk

    for block_id in range(n_blk):
        start = block_id * blk
        stop = (block_id + 1) * blk
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


def tiled_gemm(a_mat: Array, b_mat: Array, cfg: NPUOptConfig, stats: NPUPerfStats) -> Array:
    m_dim, k_dim = a_mat.shape
    k_dim2, n_dim = b_mat.shape
    if k_dim != k_dim2:
        raise ValueError(f"shape mismatch for tiled_gemm: {a_mat.shape} x {b_mat.shape}")

    out = np.zeros((m_dim, n_dim), dtype=np.complex128)

    for m0 in range(0, m_dim, cfg.tile_m):
        m1 = min(m0 + cfg.tile_m, m_dim)
        for n0 in range(0, n_dim, cfg.tile_n):
            n1 = min(n0 + cfg.tile_n, n_dim)
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.complex128)
            for k0 in range(0, k_dim, cfg.tile_k):
                k1 = min(k0 + cfg.tile_k, k_dim)
                a_tile = a_mat[m0:m1, k0:k1]
                b_tile = b_mat[k0:k1, n0:n1]
                acc += a_tile @ b_tile
                stats.gemm_calls += 1
            out[m0:m1, n0:n1] = acc

    return out


def bj_chebyshev_inverse_npu_opt(
    a_mat: Array,
    cfg: NPUOptConfig,
    stats: NPUPerfStats | None = None,
) -> Tuple[Array, NPUPerfStats]:
    local_stats = stats if stats is not None else NPUPerfStats()

    b_mat, m_half_inv = build_block_jacobi_preconditioner(a_mat, blk=cfg.blk)
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)

    omegas = (
        chebyshev_omega_adaptive(b_mat, n_layers=cfg.n_layers)
        if cfg.adaptive_bounds
        else chebyshev_omega(n_layers=cfg.n_layers, bmin=0.1, bmax=1.2)
    )

    for omega in omegas:
        by = tiled_gemm(b_mat, y_mat, cfg, local_stats)
        residual = identity - by
        local_stats.vector_calls += 1
        y_mat = y_mat + omega * residual
        local_stats.vector_calls += 1

        if cfg.symmetrize_each_layer:
            y_mat = 0.5 * (y_mat + y_mat.conj().T)
            local_stats.vector_calls += 1

    left = tiled_gemm(m_half_inv, y_mat, cfg, local_stats)
    x_approx = tiled_gemm(left, m_half_inv, cfg, local_stats)
    return x_approx, local_stats


def bj_deep_unfolding_detect_npu_opt(
    h_mat: Array,
    y_vec: Array,
    noise_var: float,
    constellation: Array,
    cfg: NPUOptConfig | None = None,
) -> Dict[str, Array | NPUPerfStats]:
    opt_cfg = cfg if cfg is not None else NPUOptConfig()

    gram, a_mat = build_regularized_system(h_mat, noise_var, constellation)
    x_approx, stats = bj_chebyshev_inverse_npu_opt(a_mat, opt_cfg)
    y_mf = matched_filter(h_mat, y_vec)
    z_vec = x_approx @ y_mf
    x_hat = hard_demod(z_vec, constellation)

    return {
        "z": z_vec,
        "x_hat": x_hat,
        "gram": gram,
        "a_mat": a_mat,
        "stats": stats,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    constellation = make_square_qam_constellation(16)
    h_mat, s_vec, y_vec, noise_var = generate_system(rng, n_rx=64, n_stream=8, constellation=constellation, snr_db=10.0)

    cfg = NPUOptConfig(n_layers=12, blk=4, adaptive_bounds=True, tile_m=16, tile_n=16, tile_k=16)
    out = bj_deep_unfolding_detect_npu_opt(h_mat, y_vec, noise_var, constellation, cfg)

    ser = float(np.mean(out["x_hat"] != s_vec))
    stats: NPUPerfStats = out["stats"]  # type: ignore[assignment]
    print(f"BJ-DeepUnfold NPU-opt SER : {ser:.4f}")
    print(f"GEMM calls={stats.gemm_calls}, Vector calls={stats.vector_calls}")


if __name__ == "__main__":
    main()
