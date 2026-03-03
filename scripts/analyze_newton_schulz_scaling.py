#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "newton_schulz" / "910b"

SIZES = [16, 32, 64, 128]


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


def analyze_size(n: int) -> Dict[str, int]:
    csv_path = RESULTS_DIR / f"profiling_log_newton_schulz_910b_{n}x{n}.csv"
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


def main() -> None:
    rows = []
    for n in SIZES:
        stats = analyze_size(n)
        row = {"Size": f"{n}x{n}"}
        row.update(stats)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.set_index("Size")

    print("Summary (cycles):")
    print(summary)

    # Emit Markdown table
    md_lines = ["| Size | MOVIN (Load) | Cube | Vector | MOVOUT (Store) | Total |",
                "|------|--------------|------|--------|----------------|-------|"]
    for size, r in summary.iterrows():
        md_lines.append(
            f"| {size} | {int(r['MOVIN'])} | {int(r['Cube'])} | {int(r['Vector'])} | {int(r['MOVOUT'])} | {int(r['Total'])} |"
        )

    md_path = ROOT / "DOCS" / "NEWTON_SCHULZ_SCALING_BASELINE.md"
    os.makedirs(md_path.parent, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Newton–Schulz Baseline Scaling (Ascend 910B)\n\n")
        f.write("Cycles vs. matrix size for the baseline Newton–Schulz inverse (batch=96, iterations=10).\n\n")
        f.write("""Note: per-unit numbers are wall-clock cycles where at least one core of that unit type is active (union of intervals across cores), not a sum of per-core active cycles.\n\n""")
        f.write("\n".join(md_lines))
        f.write("\n")

    # Plot curves
    sizes_int = [int(s.split("x")[0]) for s in summary.index]

    plt.figure(figsize=(6, 4))
    for key, label in [
        ("MOVIN", "Load (MOVIN / MTE2)"),
        ("Cube", "Cube"),
        ("Vector", "Vector"),
        ("MOVOUT", "Store (MOVOUT / MTE3)"),
        ("Total", "Total"),
    ]:
        plt.plot(sizes_int, summary[key].values, marker="o", label=label)

    plt.xlabel("Matrix dimension N (N x N)")
    plt.ylabel("Cycles")
    plt.title("Newton–Schulz Baseline: Cycles vs. Matrix Size")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = RESULTS_DIR / "newton_schulz_910b_baseline_scaling_cycles.png"
    plt.savefig(out_png, dpi=200)


if __name__ == "__main__":
    main()
