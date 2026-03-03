#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def newton_schulz_inverse(A, iters=10):
    """Classic Newton–Schulz matrix inverse iteration for a square matrix A.

    Uses scaling X0 = A^T / (||A||_1 * ||A||_inf) and update
        X_{k+1} = X_k (2I - A X_k).
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "A must be square"
    norm1 = np.linalg.norm(A, 1)
    norm_inf = np.linalg.norm(A, np.inf)
    alpha = 1.0 / (norm1 * norm_inf)
    X = alpha * A.T
    I = np.eye(n, dtype=A.dtype)
    for k in range(iters):
        X = X @ (2 * I - A @ X)
    return X


def run_random_tests(n=32, iters=10, num_cases=5, seed=0):
    rng = np.random.default_rng(seed)
    print(f"Testing Newton–Schulz inverse on {num_cases} well-conditioned {n}x{n} matrices (near identity)...")
    for i in range(num_cases):
        # Generate a well-conditioned matrix close to identity: A = I + 0.1 * R
        R = rng.standard_normal((n, n))
        A = np.eye(n) + 0.1 * R
        A_inv_ref = np.linalg.inv(A)
        X = newton_schulz_inverse(A, iters=iters)
        err = np.linalg.norm(X - A_inv_ref, ord='fro') / np.linalg.norm(A_inv_ref, ord='fro')
        res = np.linalg.norm(A @ X - np.eye(n), ord='fro') / np.linalg.norm(np.eye(n), ord='fro')
        print(f"Case {i}: rel_inv_err={err:.3e}, rel_residual=||AX-I||_F={res:.3e}")


def sweep_iters_and_plot(n=32,
                         max_iters=12,
                         num_cases=50,
                         seed=0,
                         out_path="img/newton_schulz_residual_vs_iters_fp16_fp32.png"):
    """Sweep iteration counts and plot residual vs iterations for FP32 / FP16.

    For each dtype in {float32, float16}, this function generates `num_cases`
    random well-conditioned matrices A (near identity), runs Newton–Schulz for
    iters=1..max_iters, and records the mean residual ||AX - I||_F / ||I||_F.
    Results are plotted on a log-scale y-axis and saved to `out_path`.
    """

    rng = np.random.default_rng(seed)
    iters_list = list(range(1, max_iters + 1))
    dtypes = [np.float32, np.float16]

    residual_means = {np.float32: [], np.float16: []}

    for it in iters_list:
        for dtype in dtypes:
            res_vals = []
            for _ in range(num_cases):
                # Generate a well-conditioned matrix close to identity.
                R = rng.standard_normal((n, n))
                A = (np.eye(n) + 0.1 * R).astype(dtype)

                X = newton_schulz_inverse(A, iters=it)

                # Compute residual in higher precision (float64) for stability.
                I = np.eye(n, dtype=np.float64)
                AX = (A @ X).astype(np.float64)
                res = np.linalg.norm(AX - I, ord="fro") / np.linalg.norm(I, ord="fro")
                res_vals.append(res)

            residual_means[dtype].append(float(np.mean(res_vals)))

    plt.figure(figsize=(6, 4))
    for dtype, label, marker in [
        (np.float32, "FP32 (sim full_precision)", "o"),
        (np.float16, "FP16 (sim precision=2)", "s"),
    ]:
        plt.semilogy(iters_list, residual_means[dtype], marker=marker, label=label)

    plt.xlabel("Newton Schulz iterations")
    plt.ylabel("Mean residual ||AX - I||_F")
    plt.title(f"Newton Schulz convergence on random {n}x{n} near-identity matrices")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    print(f"Saving convergence plot to {out_path}")
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Newton Schulz inverse numeric test.")
    parser.add_argument('--n', type=int, default=32, help='Matrix size (square).')
    parser.add_argument('--iters', type=int, default=10, help='Number of Newton Schulz iterations (for print mode).')
    parser.add_argument('--num_cases', type=int, default=5, help='Number of random test matrices.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--mode', type=str, default='print', choices=['print', 'sweep'],
                        help="'print' to show per-case errors, 'sweep' to plot residual vs iterations.")
    parser.add_argument('--max_iters', type=int, default=12,
                        help='Maximum iterations for sweep mode (iters from 1..max_iters).')
    parser.add_argument('--out', type=str,
                        default='img/newton_schulz_residual_vs_iters_fp16_fp32.png',
                        help='Output PNG path for sweep mode.')
    args = parser.parse_args()

    if args.mode == 'print':
        run_random_tests(n=args.n, iters=args.iters, num_cases=args.num_cases, seed=args.seed)
    else:
        sweep_iters_and_plot(n=args.n,
                             max_iters=args.max_iters,
                             num_cases=args.num_cases,
                             seed=args.seed,
                             out_path=args.out)
