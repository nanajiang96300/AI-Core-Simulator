#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CASE_ORDER = [
    "ldl_256x32",
    "deepunfold_256x32",
    "deepunfold_opt_256x32",
    "cholesky_256x32",
]

CASE_LABELS = {
    "ldl_256x32": "LDL",
    "deepunfold_256x32": "DeepUnfold",
    "deepunfold_opt_256x32": "DeepUnfold-Opt",
    "cholesky_256x32": "Cholesky",
}


def fmt_case(series: pd.Series) -> pd.Series:
    return series.map(lambda x: CASE_LABELS.get(x, x))


def ensure_order(df: pd.DataFrame, col: str = "case") -> pd.DataFrame:
    return df.set_index(col).loc[[c for c in CASE_ORDER if c in df[col].values]].reset_index()


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_wall_vs_work(wall_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    labels = fmt_case(wall_df["case"])
    x = np.arange(len(wall_df))

    bars = ax1.bar(x, wall_df["wall_cycles"], color="#4C78A8", alpha=0.9, label="Wall cycles")
    ax1.set_ylabel("Wall cycles")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Wall-cycle latency vs cumulative work")

    ax2 = ax1.twinx()
    ax2.plot(x, wall_df["work_cycles_sum"], marker="o", color="#F58518", linewidth=2.2, label="Work cycles (sum)")
    ax2.set_ylabel("Work cycles (sum of durations)")

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,.0f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8)

    lines, labels2 = ax2.get_legend_handles_labels()
    bars_h, bars_l = ax1.get_legend_handles_labels()
    ax1.legend(bars_h + lines, bars_l + labels2, loc="upper left")
    save_fig(fig, out_dir, "01_wall_vs_work_dual_axis.png")


def plot_wall_speed(wall_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = fmt_case(wall_df["case"])
    x = np.arange(len(wall_df))
    values = wall_df["wall_speed_vs_ldl"]

    colors = ["#72B7B2" if v < 1 else "#E45756" for v in values]
    bars = ax.bar(x, values, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall speed vs LDL (lower is faster)")
    ax.set_title("End-to-end latency ratio (relative to LDL)")

    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    save_fig(fig, out_dir, "02_wall_speed_vs_ldl.png")


def plot_overlap_factor(wall_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = fmt_case(wall_df["case"])
    x = np.arange(len(wall_df))
    values = wall_df["overlap_factor_sum_div_wall"]

    bars = ax.bar(x, values, color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Overlap factor = work / wall")
    ax.set_title("Parallel overlap strength comparison")

    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    save_fig(fig, out_dir, "03_overlap_factor.png")


def plot_engine_duration_stacked(cycle_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = fmt_case(cycle_df["case"])
    x = np.arange(len(cycle_df))

    comp = {
        "MTE2": cycle_df["mte2_dur"],
        "MTE3": cycle_df["mte3_dur"],
        "Cube": cycle_df["cube_dur"],
        "Vector": cycle_df["vector_dur"],
    }
    colors = {"MTE2": "#E45756", "MTE3": "#FF9DA6", "Cube": "#4C78A8", "Vector": "#72B7B2"}

    bottom = np.zeros(len(cycle_df))
    for k in ["MTE2", "MTE3", "Cube", "Vector"]:
        ax.bar(x, comp[k], bottom=bottom, label=k, color=colors[k])
        bottom += comp[k].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cycles (sum duration)")
    ax.set_title("Engine duration composition (absolute)")
    ax.legend(ncol=4, loc="upper center")
    save_fig(fig, out_dir, "04_engine_duration_stacked.png")


def plot_engine_ratio_stacked(cycle_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = fmt_case(cycle_df["case"])
    x = np.arange(len(cycle_df))

    total = cycle_df["dur_total"].replace(0, np.nan)
    comp = {
        "MTE2": cycle_df["mte2_dur"] / total * 100,
        "MTE3": cycle_df["mte3_dur"] / total * 100,
        "Cube": cycle_df["cube_dur"] / total * 100,
        "Vector": cycle_df["vector_dur"] / total * 100,
    }
    colors = {"MTE2": "#E45756", "MTE3": "#FF9DA6", "Cube": "#4C78A8", "Vector": "#72B7B2"}

    bottom = np.zeros(len(cycle_df))
    for k in ["MTE2", "MTE3", "Cube", "Vector"]:
        vals = comp[k].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=k, color=colors[k])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Ratio (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Engine duration composition (percentage)")
    ax.legend(ncol=4, loc="upper center")
    save_fig(fig, out_dir, "05_engine_duration_ratio_stacked.png")


def plot_engine_events(cycle_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels = fmt_case(cycle_df["case"])
    x = np.arange(len(cycle_df))
    width = 0.2

    series = [
        ("mte2_events", "MTE2", "#E45756"),
        ("mte3_events", "MTE3", "#FF9DA6"),
        ("cube_events", "Cube", "#4C78A8"),
        ("vector_events", "Vector", "#72B7B2"),
    ]

    for idx, (col, name, color) in enumerate(series):
        ax.bar(x + (idx - 1.5) * width, cycle_df[col], width=width, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Event count")
    ax.set_title("Engine event-count comparison")
    ax.legend(ncol=4, loc="upper center")
    save_fig(fig, out_dir, "06_engine_event_count_grouped.png")


def plot_mte_vs_compute_scatter(cycle_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    mte = cycle_df["mte_total_dur"]
    compute = cycle_df["cube_dur"] + cycle_df["vector_dur"]
    sizes = np.clip(cycle_df["dur_total"] / cycle_df["dur_total"].max() * 1200, 150, 1200)

    ax.scatter(mte, compute, s=sizes, alpha=0.7, c="#B279A2")
    for _, row in cycle_df.iterrows():
        ax.annotate(CASE_LABELS.get(row["case"], row["case"]), (row["mte_total_dur"], row["cube_dur"] + row["vector_dur"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)

    ax.set_xlabel("MTE duration (cycles)")
    ax.set_ylabel("Compute duration = Cube + Vector (cycles)")
    ax.set_title("MTE vs compute workload distribution")
    save_fig(fig, out_dir, "07_mte_vs_compute_scatter.png")


def build_load_store_share(step_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, group in step_df.groupby("case"):
        load_pct = group.loc[group["step"].str.lower() == "load", "pct"].sum()
        store_pct = group.loc[group["step"].str.lower() == "store", "pct"].sum()
        rows.append({"case": case, "load_pct": load_pct, "store_pct": store_pct})
    agg = pd.DataFrame(rows)
    agg["other_pct"] = (100.0 - agg["load_pct"] - agg["store_pct"]).clip(lower=0)
    return agg


def plot_load_store_share(step_df: pd.DataFrame, out_dir: Path) -> None:
    share = build_load_store_share(step_df)
    share = ensure_order(share)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(share))
    labels = fmt_case(share["case"])

    ax.bar(x, share["load_pct"], label="Load", color="#E45756")
    ax.bar(x, share["store_pct"], bottom=share["load_pct"], label="Store", color="#FF9DA6")
    ax.bar(x, share["other_pct"], bottom=share["load_pct"] + share["store_pct"], label="Others", color="#9D9D9D")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of total work (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Top-step view: Load/Store dominance")
    ax.legend(ncol=3, loc="upper center")
    save_fig(fig, out_dir, "08_load_store_other_share.png")


def plot_top_steps_small_multiples(step_df: pd.DataFrame, out_dir: Path, topn: int = 8) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()

    for idx, case in enumerate(CASE_ORDER):
        ax = axes_flat[idx]
        data = step_df[step_df["case"] == case].sort_values("dur", ascending=True).tail(topn)
        if data.empty:
            ax.set_title(f"{CASE_LABELS.get(case, case)} (no data)")
            continue

        ax.barh(data["step"], data["dur"], color="#4C78A8")
        ax.set_title(CASE_LABELS.get(case, case))
        ax.set_xlabel("Duration (cycles)")

    fig.suptitle(f"Top-{topn} steps by duration (per method)")
    save_fig(fig, out_dir, "09_top_steps_small_multiples.png")


def plot_step_heatmap(step_df: pd.DataFrame, out_dir: Path, topn: int = 12) -> None:
    total_by_step = step_df.groupby("step", as_index=False)["dur"].sum().sort_values("dur", ascending=False)
    top_steps = total_by_step.head(topn)["step"].tolist()

    pivot = (
        step_df[step_df["step"].isin(top_steps)]
        .pivot_table(index="step", columns="case", values="dur", aggfunc="sum", fill_value=0.0)
    )
    ordered_cols = [c for c in CASE_ORDER if c in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot.columns = [CASE_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 7.2))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_title(f"Cross-method heatmap of top-{topn} high-cost steps")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Duration (cycles)")
    save_fig(fig, out_dir, "10_step_heatmap_top12.png")


def plot_efficiency_map(cycle_df: pd.DataFrame, wall_df: pd.DataFrame, out_dir: Path) -> None:
    merged = pd.merge(cycle_df[["case", "dur_total"]], wall_df[["case", "wall_cycles"]], on="case", how="inner")

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    x = merged["dur_total"]
    y = merged["wall_cycles"]
    ax.scatter(x, y, s=320, c="#54A24B", alpha=0.75)

    for _, row in merged.iterrows():
        ax.annotate(CASE_LABELS.get(row["case"], row["case"]), (row["dur_total"], row["wall_cycles"]),
                    xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("Work cycles sum (lower is better)")
    ax.set_ylabel("Wall cycles (lower is better)")
    ax.set_title("Work-latency efficiency map")
    save_fig(fig, out_dir, "11_work_latency_efficiency_map.png")


def write_figure_index(out_dir: Path, names: List[str]) -> None:
    lines = ["# Rich Figure Index", "", "Generated figures:"]
    for name in names:
        lines.append(f"- `{name}`")
    (out_dir / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rich aligned comparison figures.")
    parser.add_argument("--cycle-summary", default="results/compare_aligned/four_method_cycle_summary.csv")
    parser.add_argument("--wall-work", default="results/compare_aligned/four_method_wallclock_vs_work.csv")
    parser.add_argument("--step-top", default="results/compare_aligned/four_method_step_breakdown_top20.csv")
    parser.add_argument("--out-dir", default="results/compare_aligned/rich_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle_df = ensure_order(pd.read_csv(args.cycle_summary))
    wall_df = ensure_order(pd.read_csv(args.wall_work))
    step_df = pd.read_csv(args.step_top)

    plot_wall_vs_work(wall_df, out_dir)
    plot_wall_speed(wall_df, out_dir)
    plot_overlap_factor(wall_df, out_dir)
    plot_engine_duration_stacked(cycle_df, out_dir)
    plot_engine_ratio_stacked(cycle_df, out_dir)
    plot_engine_events(cycle_df, out_dir)
    plot_mte_vs_compute_scatter(cycle_df, out_dir)
    plot_load_store_share(step_df, out_dir)
    plot_top_steps_small_multiples(step_df, out_dir, topn=8)
    plot_step_heatmap(step_df, out_dir, topn=12)
    plot_efficiency_map(cycle_df, wall_df, out_dir)

    figures = [
        "01_wall_vs_work_dual_axis.png",
        "02_wall_speed_vs_ldl.png",
        "03_overlap_factor.png",
        "04_engine_duration_stacked.png",
        "05_engine_duration_ratio_stacked.png",
        "06_engine_event_count_grouped.png",
        "07_mte_vs_compute_scatter.png",
        "08_load_store_other_share.png",
        "09_top_steps_small_multiples.png",
        "10_step_heatmap_top12.png",
        "11_work_latency_efficiency_map.png",
    ]
    write_figure_index(out_dir, figures)

    print(f"Generated {len(figures)} figures in: {out_dir}")


if __name__ == "__main__":
    main()
