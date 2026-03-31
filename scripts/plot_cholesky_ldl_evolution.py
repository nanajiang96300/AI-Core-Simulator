#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize(path: Path):
    total_cnt = 0
    total_dur = 0
    max_end = 0

    core0_cnt = 0
    core0_dur = 0
    core0_end = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row["start_cycle"])
            end = int(row["end_cycle"])
            dur = end - start
            total_cnt += 1
            total_dur += dur
            max_end = max(max_end, end)

            unit = row["unit"].strip('"')
            if unit.startswith("Core0_"):
                core0_cnt += 1
                core0_dur += dur
                core0_end = max(core0_end, end)

    return {
        "total_cnt": total_cnt,
        "total_dur": total_dur,
        "max_end": max_end,
        "core0_cnt": core0_cnt,
        "core0_dur": core0_dur,
        "core0_end": core0_end,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Cholesky lowering evolution against LDL")
    parser.add_argument("--old", default="results/CHOL/falsification/cholesky_noblock_64x16_trace.csv")
    parser.add_argument("--invmul", default="results/CHOL/falsification/cholesky_noblock_64x16_trace_invmul.csv")
    parser.add_argument("--iso", default="results/CHOL/falsification/cholesky_noblock_64x16_trace_iso.csv")
    parser.add_argument("--ldl", default="results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv")
    parser.add_argument("--png", default="results/LDL/falsification/cholesky_ldl_evolution_compare_20260327.png")
    parser.add_argument("--csv", default="results/LDL/falsification/cholesky_ldl_evolution_summary_20260327.csv")
    args = parser.parse_args()

    cases = [
        ("CHOL-old", Path(args.old)),
        ("CHOL-invmul", Path(args.invmul)),
        ("CHOL-iso", Path(args.iso)),
        ("LDL-aligned", Path(args.ldl)),
    ]

    rows = []
    for name, path in cases:
        stats = summarize(path)
        rows.append((name, stats))

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "total_cnt", "total_dur", "max_end", "core0_cnt", "core0_dur", "core0_end"])
        for name, s in rows:
            w.writerow([name, s["total_cnt"], s["total_dur"], s["max_end"], s["core0_cnt"], s["core0_dur"], s["core0_end"]])

    names = [x[0] for x in rows]
    max_end = np.array([x[1]["max_end"] for x in rows], dtype=float)
    core0_end = np.array([x[1]["core0_end"] for x in rows], dtype=float)
    total_cnt = np.array([x[1]["total_cnt"] for x in rows], dtype=float)
    core0_cnt = np.array([x[1]["core0_cnt"] for x in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    x = np.arange(len(names))
    color = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    axes[0, 0].bar(x, max_end, color=color)
    axes[0, 0].set_title("Global max_end_cycle")
    axes[0, 0].set_xticks(x, names, rotation=20)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 1].bar(x, core0_end, color=color)
    axes[0, 1].set_title("Core0 max_end_cycle")
    axes[0, 1].set_xticks(x, names, rotation=20)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1, 0].bar(x, total_cnt, color=color)
    axes[1, 0].set_title("Global event count")
    axes[1, 0].set_xticks(x, names, rotation=20)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1, 1].bar(x, core0_cnt, color=color)
    axes[1, 1].set_title("Core0 event count")
    axes[1, 1].set_xticks(x, names, rotation=20)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.35)

    ldl_global = max_end[-1]
    ldl_core0 = core0_end[-1]
    for i, value in enumerate(max_end):
        axes[0, 0].text(i, value, f"{value/ldl_global:.2f}x", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(core0_end):
        axes[0, 1].text(i, value, f"{value/ldl_core0:.2f}x", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cholesky Lowering Evolution vs LDL-aligned", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])

    out_png = Path(args.png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print("summary_csv", out_csv)
    print("plot_png", out_png)


if __name__ == "__main__":
    main()
