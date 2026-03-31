#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from scripts.DeepUnfold.bj_deep_unfolding_npu_opt import (
    NPUOptConfig,
    NPUPerfStats,
    average_symbol_energy,
    bj_chebyshev_inverse_npu_opt,
    hard_demod,
    make_square_qam_constellation,
    tiled_gemm,
)

Array = np.ndarray


@dataclass
class NPUOptOverlapConfig(NPUOptConfig):
    h_chunks: int = 2
    y_chunks: int = 2


def _split_ranges(length: int, chunks: int) -> List[Tuple[int, int]]:
    chunks = max(1, min(chunks, length))
    base = length // chunks
    rem = length % chunks
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(chunks):
        span = base + (1 if idx < rem else 0)
        end = start + span
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def build_regularized_system_chunked(
    h_mat: Array,
    noise_var: float,
    constellation: Array,
    cfg: NPUOptOverlapConfig,
    stats: NPUPerfStats | None = None,
) -> Tuple[Array, Array]:
    local_stats = stats if stats is not None else NPUPerfStats()
    row_ranges = _split_ranges(h_mat.shape[0], cfg.h_chunks)

    gram = np.zeros((h_mat.shape[1], h_mat.shape[1]), dtype=np.complex128)
    for r0, r1 in row_ranges:
        h_blk = h_mat[r0:r1, :]
        gram += tiled_gemm(h_blk.conj().T, h_blk, cfg, local_stats)

    symbol_energy = average_symbol_energy(constellation)
    a_mat = gram + (noise_var / symbol_energy) * np.eye(h_mat.shape[1], dtype=np.complex128)
    return gram, a_mat


def matched_filter_chunked(
    h_mat: Array,
    y_vec: Array,
    cfg: NPUOptOverlapConfig,
    stats: NPUPerfStats | None = None,
) -> Array:
    local_stats = stats if stats is not None else NPUPerfStats()
    row_ranges = _split_ranges(h_mat.shape[0], cfg.y_chunks)

    y_mf = np.zeros((h_mat.shape[1],), dtype=np.complex128)
    for r0, r1 in row_ranges:
        h_blk = h_mat[r0:r1, :]
        y_blk = y_vec[r0:r1]
        y_mf += h_blk.conj().T @ y_blk
        local_stats.vector_calls += 1

    return y_mf


def bj_chebyshev_inverse_npu_opt_overlap(
    a_mat: Array,
    cfg: NPUOptOverlapConfig,
    stats: NPUPerfStats | None = None,
) -> Tuple[Array, NPUPerfStats]:
    return bj_chebyshev_inverse_npu_opt(a_mat, cfg, stats)


def bj_deep_unfolding_detect_npu_opt_overlap(
    h_mat: Array,
    y_vec: Array,
    noise_var: float,
    constellation: Array,
    cfg: NPUOptOverlapConfig | None = None,
) -> Dict[str, Array | NPUPerfStats]:
    opt_cfg = cfg if cfg is not None else NPUOptOverlapConfig()
    stats = NPUPerfStats()

    gram, a_mat = build_regularized_system_chunked(h_mat, noise_var, constellation, opt_cfg, stats)
    a_inv, stats = bj_chebyshev_inverse_npu_opt_overlap(a_mat, opt_cfg, stats)
    y_mf = matched_filter_chunked(h_mat, y_vec, opt_cfg, stats)
    z_vec = a_inv @ y_mf
    x_hat = hard_demod(z_vec, constellation)

    return {
        "z": z_vec,
        "x_hat": x_hat,
        "gram": gram,
        "a_mat": a_mat,
        "a_inv": a_inv,
        "stats": stats,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    constellation = make_square_qam_constellation(16)

    nr, nt = 64, 8
    h_mat = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0 * nr)
    tx = constellation[rng.integers(0, len(constellation), size=nt)]
    snr_db = 10.0
    noise_var = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(nr) + 1j * rng.standard_normal(nr))
    y_vec = h_mat @ tx + noise

    cfg = NPUOptOverlapConfig(n_layers=12, blk=4, h_chunks=2, y_chunks=2)
    out = bj_deep_unfolding_detect_npu_opt_overlap(h_mat, y_vec, noise_var, constellation, cfg)

    ser = float(np.mean(out["x_hat"] != tx))
    stats: NPUPerfStats = out["stats"]  # type: ignore[assignment]
    print(f"BJ-DeepUnfold NPU-opt-overlap SER : {ser:.4f}")
    print(f"GEMM calls={stats.gemm_calls}, Vector calls={stats.vector_calls}")


if __name__ == "__main__":
    main()
