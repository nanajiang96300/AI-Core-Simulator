#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    unit_col = lower.get("unit")
    name_col = lower.get("name")
    start_col = lower.get("startcycle") or lower.get("start_cycle")
    end_col = lower.get("endcycle") or lower.get("end_cycle")
    if unit_col is None or start_col is None or end_col is None:
        raise ValueError(f"Missing required columns in CSV: {df.columns.tolist()}")
    renamed = df.rename(columns={unit_col: "Unit", start_col: "Start", end_col: "End"})
    if name_col is not None:
        renamed = renamed.rename(columns={name_col: "Name"})
    else:
        renamed["Name"] = ""
    renamed["Dur"] = renamed["End"] - renamed["Start"]
    return renamed


def load_case(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    df = df[df["Dur"] > 0].copy()
    return df


def sum_unit_duration(df: pd.DataFrame, key: str) -> float:
    mask = df["Unit"].astype(str).str.contains(key, case=False, regex=False)
    return float(df.loc[mask, "Dur"].sum())


def count_unit_events(df: pd.DataFrame, key: str) -> int:
    mask = df["Unit"].astype(str).str.contains(key, case=False, regex=False)
    return int(mask.sum())


def build_summary(cases: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for case_name, df in cases.items():
        total = float(df["Dur"].sum())
        mte2 = sum_unit_duration(df, "MTE2")
        mte3 = sum_unit_duration(df, "MTE3")
        cube = sum_unit_duration(df, "Cube")
        vector = sum_unit_duration(df, "Vector")
        mte_total = mte2 + mte3
        rows.append(
            {
                "case": case_name,
                "events_total": int(len(df)),
                "dur_total": total,
                "mte2_events": count_unit_events(df, "MTE2"),
                "mte3_events": count_unit_events(df, "MTE3"),
                "cube_events": count_unit_events(df, "Cube"),
                "vector_events": count_unit_events(df, "Vector"),
                "mte2_dur": mte2,
                "mte3_dur": mte3,
                "cube_dur": cube,
                "vector_dur": vector,
                "mte_total_dur": mte_total,
                "mte_ratio": (mte_total / total) if total > 0 else 0.0,
            }
        )

    summary = pd.DataFrame(rows)
    summary["mte_ratio_pct"] = summary["mte_ratio"] * 100.0
    summary["speed_vs_ldl"] = summary["dur_total"] / float(
        summary.loc[summary["case"] == "ldl_256x32", "dur_total"].iloc[0]
    )
    return summary


def plot_transport_bars(summary: pd.DataFrame, out_png: Path) -> None:
    order = ["ldl_256x32", "deepunfold_256x32", "deepunfold_opt_256x32"]
    df = summary.set_index("case").loc[order].reset_index()

    x = range(len(df))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar([i - width / 2 for i in x], df["mte2_dur"], width=width, label="MTE2 duration")
    ax1.bar([i + width / 2 for i in x], df["mte3_dur"], width=width, label="MTE3 duration")
    ax1.set_ylabel("Duration (cycles)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["case"], rotation=0)
    ax1.set_title("Aligned Transport Duration Comparison (M=256, K=32, B=96)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(list(x), df["mte_ratio_pct"], marker="o", color="tab:red", label="MTE ratio %")
    ax2.set_ylabel("MTE ratio (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _pick_core0_rows(df: pd.DataFrame) -> pd.DataFrame:
    unit = df["Unit"].astype(str)
    mask = unit.str.startswith("Core0_")
    picked = df[mask].copy()
    return picked


def plot_core0_timeline(cases: Dict[str, pd.DataFrame], out_png: Path, max_cycle: int) -> None:
    order = ["ldl_256x32", "deepunfold_256x32", "deepunfold_opt_256x32"]
    color_map = {
        "Cube": "#2ca02c",
        "Vector": "#98df8a",
        "MTE2": "#ff7f7f",
        "MTE3": "#1f77b4",
    }
    row_order = ["MTE2", "Cube", "Vector", "MTE3"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for idx, case in enumerate(order):
        ax = axes[idx]
        df = _pick_core0_rows(cases[case])
        if max_cycle > 0:
            df = df[df["Start"] <= max_cycle].copy()

        y_map = {k: i for i, k in enumerate(row_order)}
        for _, row in df.iterrows():
            u = str(row["Unit"])
            lane = None
            for key in row_order:
                if key in u:
                    lane = key
                    break
            if lane is None:
                continue
            dur = row["Dur"]
            if dur <= 0:
                continue
            ax.broken_barh(
                [(row["Start"], dur)],
                (y_map[lane] - 0.35, 0.7),
                facecolors=color_map[lane],
                edgecolors="none",
            )

        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels(row_order)
        ax.set_title(f"{case} (Core0)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)

    axes[-1].set_xlabel("Cycle")
    if max_cycle > 0:
        axes[-1].set_xlim(0, max_cycle)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligned timeline comparison for LDL / DeepUnfold / DeepUnfoldOpt")
    parser.add_argument("--ldl", default="results/compare_aligned/ldl_256x32_trace.csv")
    parser.add_argument("--deep", default="results/compare_aligned/deepunfold_256x32_trace.csv")
    parser.add_argument("--deep-opt", default="results/compare_aligned/deepunfold_opt_256x32_trace.csv")
    parser.add_argument("--out-dir", default="results/compare_aligned")
    parser.add_argument("--max-cycle", type=int, default=7000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "ldl_256x32": load_case(Path(args.ldl)),
        "deepunfold_256x32": load_case(Path(args.deep)),
        "deepunfold_opt_256x32": load_case(Path(args.deep_opt)),
    }

    summary = build_summary(cases)
    summary_csv = out_dir / "aligned_timeline_compare_summary.csv"
    summary.to_csv(summary_csv, index=False)

    transport_png = out_dir / "aligned_transport_compare.png"
    plot_transport_bars(summary, transport_png)

    timeline_png = out_dir / "aligned_core0_timeline_compare.png"
    plot_core0_timeline(cases, timeline_png, args.max_cycle)

    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved plot: {transport_png}")
    print(f"Saved plot: {timeline_png}")


if __name__ == "__main__":
    main()
