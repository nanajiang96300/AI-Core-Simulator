import argparse
import numpy as np


def make_hpd_matrix(n: int, seed: int, dtype=np.complex128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, n))
    imag = rng.standard_normal((n, n))
    temp = real + 1j * imag
    mat = temp @ temp.conj().T
    mat = mat + n * np.eye(n, dtype=np.complex128)
    return mat.astype(dtype, copy=False)


def chain_cholesky_nonuniform(mat: np.ndarray, block_size: int = 2) -> np.ndarray:
    n = mat.shape[0]
    if n <= block_size:
        return np.linalg.cholesky(mat)

    b = block_size
    a = mat[:b, :b]
    b01 = mat[:b, b:]
    b10 = mat[b:, :b]
    d = mat[b:, b:]

    l11 = np.linalg.cholesky(a)
    l11_inv = np.linalg.inv(l11)
    l21 = b10 @ l11_inv.conj().T

    schur = d - l21 @ l21.conj().T
    l22 = chain_cholesky_nonuniform(schur, block_size=b)

    l = np.zeros_like(mat)
    l[:b, :b] = l11
    l[b:, :b] = l21
    l[b:, b:] = l22
    return l


def lower_inverse_nonuniform(l: np.ndarray, block_size: int = 2) -> np.ndarray:
    n = l.shape[0]
    if n <= block_size:
        return np.linalg.inv(l)

    b = block_size
    l11 = l[:b, :b]
    l21 = l[b:, :b]
    l22 = l[b:, b:]

    x = np.linalg.inv(l11)
    z = lower_inverse_nonuniform(l22, block_size=b)
    y = -z @ l21 @ x

    l_inv = np.zeros_like(l)
    l_inv[:b, :b] = x
    l_inv[b:, :b] = y
    l_inv[b:, b:] = z
    return l_inv


def cholesky_uniform(mat: np.ndarray, leaf_size: int = 2) -> np.ndarray:
    n = mat.shape[0]
    if n <= leaf_size:
        return np.linalg.cholesky(mat)

    split = n // 2
    if split == 0:
        return np.linalg.cholesky(mat)

    a = mat[:split, :split]
    b10 = mat[split:, :split]
    d = mat[split:, split:]

    l11 = cholesky_uniform(a, leaf_size=leaf_size)
    l11_inv = np.linalg.inv(l11)
    l21 = b10 @ l11_inv.conj().T
    schur = d - l21 @ l21.conj().T
    l22 = cholesky_uniform(schur, leaf_size=leaf_size)

    l = np.zeros_like(mat)
    l[:split, :split] = l11
    l[split:, :split] = l21
    l[split:, split:] = l22
    return l


def lower_inverse_uniform(l: np.ndarray, leaf_size: int = 2) -> np.ndarray:
    n = l.shape[0]
    if n <= leaf_size:
        return np.linalg.inv(l)

    split = n // 2
    if split == 0:
        return np.linalg.inv(l)

    l11 = l[:split, :split]
    l21 = l[split:, :split]
    l22 = l[split:, split:]

    x = lower_inverse_uniform(l11, leaf_size=leaf_size)
    z = lower_inverse_uniform(l22, leaf_size=leaf_size)
    y = -z @ l21 @ x

    l_inv = np.zeros_like(l)
    l_inv[:split, :split] = x
    l_inv[split:, :split] = y
    l_inv[split:, split:] = z
    return l_inv


def assemble_inverse_from_linv(l_inv: np.ndarray) -> np.ndarray:
    return l_inv.conj().T @ l_inv


def validate_one(mat: np.ndarray, mode: str, block_size: int, leaf_size: int):
    if mode == "nonuniform":
        l = chain_cholesky_nonuniform(mat, block_size=block_size)
        l_inv = lower_inverse_nonuniform(l, block_size=block_size)
    elif mode == "uniform":
        l = cholesky_uniform(mat, leaf_size=leaf_size)
        l_inv = lower_inverse_uniform(l, leaf_size=leaf_size)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    inv_custom = assemble_inverse_from_linv(l_inv)
    inv_ref = np.linalg.inv(mat)

    recon_err = np.linalg.norm(mat - l @ l.conj().T, ord="fro")
    inv_err = np.linalg.norm(inv_ref - inv_custom, ord="fro")
    residual = np.linalg.norm(np.eye(mat.shape[0], dtype=mat.dtype) - mat @ inv_custom, ord="fro")
    return recon_err, inv_err, residual


def run_suite(n: int, tests: int, seed: int, block_size: int, leaf_size: int, threshold: float):
    print("=== Cholesky Block Inversion Validation ===")
    print(f"matrix_size={n}, tests={tests}, block_size={block_size}, leaf_size={leaf_size}, threshold={threshold:.1e}")
    print("focus: nonuniform chain decomposition (fixed 2x2 peeling style)")

    nonuniform_pass = 0
    uniform_pass = 0

    nonuniform_metrics = []
    uniform_metrics = []

    for i in range(tests):
        case_seed = seed + i
        mat = make_hpd_matrix(n=n, seed=case_seed)

        nu_recon, nu_inv, nu_res = validate_one(mat, mode="nonuniform", block_size=block_size, leaf_size=leaf_size)
        uf_recon, uf_inv, uf_res = validate_one(mat, mode="uniform", block_size=block_size, leaf_size=leaf_size)

        nonuniform_metrics.append((nu_recon, nu_inv, nu_res))
        uniform_metrics.append((uf_recon, uf_inv, uf_res))

        if nu_inv < threshold:
            nonuniform_pass += 1
        if uf_inv < threshold:
            uniform_pass += 1

        print(
            f"[case {i:02d} seed={case_seed}] "
            f"NU(inv={nu_inv:.3e}, recon={nu_recon:.3e}, residual={nu_res:.3e}) | "
            f"U(inv={uf_inv:.3e}, recon={uf_recon:.3e}, residual={uf_res:.3e})"
        )

    def summarize(name: str, data, passed: int):
        arr = np.array(data)
        max_recon, max_inv, max_res = arr.max(axis=0)
        avg_recon, avg_inv, avg_res = arr.mean(axis=0)
        print(f"\n[{name}] pass={passed}/{tests}")
        print(f"  max: recon={max_recon:.3e}, inv={max_inv:.3e}, residual={max_res:.3e}")
        print(f"  avg: recon={avg_recon:.3e}, inv={avg_inv:.3e}, residual={avg_res:.3e}")

    summarize("nonuniform-chain", nonuniform_metrics, nonuniform_pass)
    summarize("uniform-halving", uniform_metrics, uniform_pass)

    print("\nResult:")
    if nonuniform_pass == tests:
        print("  Nonuniform chain validation PASSED.")
    else:
        print("  Nonuniform chain validation FAILED.")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate uniform vs nonuniform Cholesky block inversion")
    parser.add_argument("--n", type=int, default=8, help="matrix size")
    parser.add_argument("--tests", type=int, default=8, help="number of random test cases")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--block-size", type=int, default=2, help="fixed block size for nonuniform chain")
    parser.add_argument("--leaf-size", type=int, default=2, help="leaf size for uniform recursive split")
    parser.add_argument("--threshold", type=float, default=1e-10, help="inverse error pass threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size < 1 or args.leaf_size < 1:
        raise ValueError("block-size and leaf-size must be >= 1")
    if args.n < 2:
        raise ValueError("matrix size n must be >= 2")

    run_suite(
        n=args.n,
        tests=args.tests,
        seed=args.seed,
        block_size=args.block_size,
        leaf_size=args.leaf_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
