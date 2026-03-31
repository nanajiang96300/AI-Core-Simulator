#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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


def compute_gram(H: Array) -> Array:
    return H.conj().T @ H


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


def sort_users(G: Array, noise_var: float, symbol_energy: float, block_size: int = 2) -> List[Array]:
    diag_abs = np.abs(np.diag(G))
    lambda_u = np.sum(np.abs(G) ** 2, axis=1) - diag_abs ** 2
    sinr_inv = lambda_u / (diag_abs ** 2 + 1e-12) + noise_var / (symbol_energy * diag_abs + 1e-12)
    user_order = np.argsort(sinr_inv)
    return [user_order[i:i + block_size] for i in range(0, len(user_order), block_size)]


def invert_2x2_block(mat: Array) -> Array:
    a, b = mat[0, 0], mat[0, 1]
    c, d = mat[1, 0], mat[1, 1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        return np.linalg.pinv(mat)
    return (1.0 / det) * np.array([[d, -b], [-c, a]], dtype=np.complex128)


def build_block_diagonal_inverse(G_sorted: Array, block_size: int = 2) -> Array:
    n_stream = G_sorted.shape[0]
    if n_stream % block_size != 0:
        raise ValueError("n_stream must be divisible by block_size")
    out = np.zeros_like(G_sorted, dtype=np.complex128)
    for start in range(0, n_stream, block_size):
        stop = start + block_size
        out[start:stop, start:stop] = invert_2x2_block(G_sorted[start:stop, start:stop])
    return out


def soft_box_real(x: Array, beta: float, half_range: float) -> Array:
    expv = np.exp(-(beta / half_range) * x)
    return (-(expv - 1.0) / (expv + 1.0)) * half_range


def soft_box_complex(v: Array, beta: float, half_range: float) -> Array:
    return soft_box_real(v.real, beta, half_range) + 1j * soft_box_real(v.imag, beta, half_range)


def gbcd_box_par_detect(
    H: Array,
    y: Array,
    noise_var: float,
    constellation: Array,
    num_iters: int = 6,
    block_size: int = 2,
    soft_box_beta: float = 4.0,
    box_half_range: float = 1.5,
) -> Dict[str, Array]:
    if block_size != 2:
        raise ValueError("This ONNXim-flow reference currently assumes block_size=2")

    symbol_energy = average_symbol_energy(constellation)
    G = compute_gram(H)
    y_mf = matched_filter(H, y)
    order = sort_users(G, noise_var, symbol_energy, block_size=block_size)
    perm = np.concatenate(order, axis=0).astype(np.int64)
    inv_perm = np.argsort(perm)

    G_sorted = G[np.ix_(perm, perm)]
    y_mf_sorted = y_mf[perm]
    K_diag = build_block_diagonal_inverse(G_sorted, block_size=block_size)

    z = np.zeros(H.shape[1], dtype=np.complex128)
    r = y_mf_sorted.astype(np.complex128).copy()
    residual_norm_history = [float(np.linalg.norm(r))]

    for _ in range(num_iters):
        v = K_diag @ r + z
        z_new = soft_box_complex(v, beta=soft_box_beta, half_range=box_half_range)
        delta = z_new - z
        r = r - G_sorted @ delta
        z = z_new
        residual_norm_history.append(float(np.linalg.norm(r)))

    z_unsorted = z[inv_perm]
    x_hat = hard_demod(z_unsorted, constellation)
    return {
        "z": z_unsorted,
        "x_hat": x_hat,
        "gram": G,
        "y_mf": y_mf,
        "order": [np.asarray(block, dtype=np.int64) for block in order],
        "residual_norm_history": np.asarray(residual_norm_history, dtype=np.float64),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    constellation = make_square_qam_constellation(16)
    H, s, y, noise_var = generate_system(rng, n_rx=64, n_stream=8, constellation=constellation, snr_db=10.0)
    out = gbcd_box_par_detect(
        H,
        y,
        noise_var,
        constellation,
        num_iters=6,
        soft_box_beta=4.0,
        box_half_range=1.5,
    )
    ser = float(np.mean(out["x_hat"] != s))
    print(f"GBCD-BOX-Par ONNX-flow SER : {ser:.4f}")


if __name__ == "__main__":
    main()