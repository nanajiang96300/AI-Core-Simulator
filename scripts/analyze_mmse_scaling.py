#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "mmse"

# Fixed K = 32, varying M (base-station antennas)
CASES_FIXED_K32 = [
    (64, 32),
    (128, 32),
    (256, 32),
    (512, 32),
    (1024, 32),
]

# Fixed M = 256, varying K (user antennas)
CASES_FIXED_M256 = [
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_to_orig = {c.lower(): c for c in df.columns}
    unit_col = lower_to_orig.get("unit")
    name_col = lower_to_orig.get("name")
    start_col = lower_to_orig.get("startcycle") or lower_to_orig.get("start_cycle")
    end_col = lower_to_orig.get("endcycle") or lower_to_orig.get("end_cycle")

    required = {"unit": unit_col, "name": name_col, "start": start_col, "end": end_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"CSV is missing required columns (or variants): {', '.join(missing)}"
        )

    df = df.rename(
        columns={
            unit_col: "Unit",
            name_col: "Name",
            start_col: "StartCycle",
            end_col: "EndCycle",
        }
    )
    return df


def _engine_category(unit: str) -> str:
    # Unit is like "Core0_Cube", "Core3_MTE2", etc.
    parts = str(unit).split("_", 1)
    engine = parts[1] if len(parts) == 2 else parts[0]
    if engine == "Cube":
        return "Cube"
    if engine == "Vector":
        return "Vector"
    if engine == "MTE2":
        return "MOVIN"  # load
    if engine == "MTE3":
        return "MOVOUT"  # store
    return "Other"


def _union_length(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    # sort by start
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_end:
            total += max(0, cur_end - cur_start)
            cur_start, cur_end = s, e
        else:
            if e > cur_end:
                cur_end = e
    total += max(0, cur_end - cur_start)
    return total


def analyze_case(m: int, k: int) -> Dict[str, int]:
    csv_path = RESULTS_DIR / f"profiling_log_mmse_910b_{m}x{k}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df["Duration"] = df["EndCycle"] - df["StartCycle"]
    df = df[df["Duration"] > 0]
    df["Category"] = df["Unit"].apply(_engine_category)

    total_cycles = int(df["EndCycle"].max())

    breakdown: Dict[str, int] = {}
    for cat in ["MOVIN", "Cube", "Vector", "MOVOUT"]:
        sub = df[df["Category"] == cat]
        intervals = list(zip(sub["StartCycle"].astype(int), sub["EndCycle"].astype(int)))
        breakdown[cat] = _union_length(intervals)

    breakdown["Total"] = total_cycles
    return breakdown


def _build_summary(cases: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for m, k in cases:
        stats = analyze_case(m, k)
        row = {"M": m, "K": k}
        row.update(stats)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["M", "K"]).set_index(["M", "K"])
    return summary


def _emit_markdown(summary_k32: pd.DataFrame, summary_m256: pd.DataFrame) -> None:
    md_path = ROOT / "DOCS" / "MMSE_SCALING_BASELINE.md"
    os.makedirs(md_path.parent, exist_ok=True)

    lines: List[str] = []
    lines.append("# MMSE Baseline Scaling (Ascend 910B)\n")
    lines.append(
        "Cycles vs. antenna configuration for the baseline MMSE operator "
        "(batch=96, Newton–Schulz iterations=10).\n"
    )
    lines.append(
        "Note: per-unit numbers are wall-clock cycles where at least one core "
        "of that unit type is active (union of intervals across cores), not a "
        "sum of per-core active cycles.\n"
    )

    # Fixed K = 32
    lines.append("\n## Fixed K = 32 (varying M)\n")
    lines.append("| M | K | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |")
    lines.append("|---|---|--------------|------|--------|----------------|-------|")
    for (m, k), r in summary_k32.iterrows():
        lines.append(
            f"| {m} | {k} | {int(r['MOVIN'])} | {int(r['Cube'])} | "
            f"{int(r['Vector'])} | {int(r['MOVOUT'])} | {int(r['Total'])} |"
        )

    # Fixed M = 256
    lines.append("\n## Fixed M = 256 (varying K)\n")
    lines.append("| M | K | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |")
    lines.append("|---|---|--------------|------|--------|----------------|-------|")
    for (m, k), r in summary_m256.iterrows():
        lines.append(
            f"| {m} | {k} | {int(r['MOVIN'])} | {int(r['Cube'])} | "
            f"{int(r['Vector'])} | {int(r['MOVOUT'])} | {int(r['Total'])} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_curves(summary_k32: pd.DataFrame, summary_m256: pd.DataFrame) -> None:
    # Fixed K = 32: x-axis is M
    ms = [idx[0] for idx in summary_k32.index]  # (M, K)

    plt.figure(figsize=(6, 4))
    for key, label in [
        ("MOVIN", "Load (MOVIN / MTE2)"),
        ("Cube", "Cube"),
        ("Vector", "Vector"),
        ("MOVOUT", "Store (MOVOUT / MTE3)"),
        ("Total", "Total"),
    ]:
        plt.plot(ms, summary_k32[key].values, marker="o", label=label)

    plt.xlabel("Base-station antennas M (K = 32)")
    plt.ylabel("Cycles")
    plt.title("MMSE Baseline: Cycles vs. M (K = 32)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = RESULTS_DIR / "mmse_910b_scaling_fixed_k32_cycles.png"
    plt.savefig(out_png, dpi=200)

    # Fixed M = 256: x-axis is K
    ks = [idx[1] for idx in summary_m256.index]  # (M, K)

    plt.figure(figsize=(6, 4))
    for key, label in [
        ("MOVIN", "Load (MOVIN / MTE2)"),
        ("Cube", "Cube"),
        ("Vector", "Vector"),
        ("MOVOUT", "Store (MOVOUT / MTE3)"),
        ("Total", "Total"),
    ]:
        plt.plot(ks, summary_m256[key].values, marker="o", label=label)

    plt.xlabel("User antennas K (M = 256)")
    plt.ylabel("Cycles")
    plt.title("MMSE Baseline: Cycles vs. K (M = 256)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = RESULTS_DIR / "mmse_910b_scaling_fixed_m256_cycles.png"
    plt.savefig(out_png, dpi=200)


def main() -> None:
    summary_k32 = _build_summary(CASES_FIXED_K32)
    summary_m256 = _build_summary(CASES_FIXED_M256)

    print("MMSE Scaling (fixed K = 32):")
    print(summary_k32)
    print()
    print("MMSE Scaling (fixed M = 256):")
    print(summary_m256)

    _emit_markdown(summary_k32, summary_m256)
    _plot_curves(summary_k32, summary_m256)


if __name__ == "__main__":
    main()
