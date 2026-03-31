#!/usr/bin/env python3

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_to_orig = {c.lower(): c for c in df.columns}
    unit_col = lower_to_orig.get("unit")
    name_col = lower_to_orig.get("name")
    start_col = lower_to_orig.get("start_cycle") or lower_to_orig.get("startcycle")
    end_col = lower_to_orig.get("end_cycle") or lower_to_orig.get("endcycle")

    required = {"unit": unit_col, "name": name_col, "start": start_col, "end": end_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"CSV is missing required columns (or variants): {', '.join(missing)}")

    return df.rename(
        columns={unit_col: "unit", name_col: "name", start_col: "start", end_col: "end"}
    )


def _parse_unit(raw: str) -> Tuple[int, str]:
    match = re.match(r"^Core(\d+)_([A-Za-z0-9]+)$", str(raw))
    if not match:
        return -1, "Unknown"
    core_id = int(match.group(1))
    engine_raw = match.group(2)
    engine_map = {
        "Cube": "Cube Unit",
        "Vector": "Vector Unit",
        "MTE2": "MTE2 (Load)",
        "MTE3": "MTE3 (Store)",
    }
    return core_id, engine_map.get(engine_raw, engine_raw)


def _label_operator(name: str, engine: str) -> str:
    name_u = str(name).upper()
    if engine == "MTE2 (Load)":
        return "MTE2"
    if engine == "MTE3 (Store)":
        return "MTE3"
    if engine == "Vector Unit" and "NS_R" in name_u:
        return "Demod"
    if engine == "Cube Unit" and "NS_T" in name_u:
        return "CE"
    if engine == "Cube Unit" and "NS_X" in name_u:
        return "Det"
    if engine == "Cube Unit":
        return "TENN"
    return "Barrier/Other"


def _track_order() -> List[Tuple[int, str]]:
    engines = ["Cube Unit", "Vector Unit", "MTE3 (Store)", "MTE2 (Load)"]
    order: List[Tuple[int, str]] = []
    for core_id in [0, 1]:
        for engine in engines:
            order.append((core_id, engine))
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style 2-core timeline plot for batch96 profiling")
    parser.add_argument(
        "-i",
        "--input",
        default="results/newton_schulz/910b/profiling_log_newton_910b_batch96.csv",
        help="Input profiling CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/newton_schulz/910b/profiling_log_newton_910b_batch96_paper.png",
        help="Output PNG",
    )
    parser.add_argument("--xmax", type=float, default=None, help="Optional fixed x-axis max")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    df = _normalize_columns(df)

    parsed = df["unit"].map(_parse_unit)
    df["core_id"] = parsed.map(lambda x: x[0])
    df["engine"] = parsed.map(lambda x: x[1])

    df = df[(df["core_id"].isin([0, 1])) & (df["engine"].isin(["Cube Unit", "Vector Unit", "MTE2 (Load)", "MTE3 (Store)"]))]
    df = df[df["end"] > df["start"]].copy()
    if df.empty:
        raise SystemExit("No valid events for Core0/Core1 in input CSV")

    df["duration"] = df["end"] - df["start"]
    df["operator"] = [
        _label_operator(name=n, engine=e) for n, e in zip(df["name"], df["engine"])
    ]

    color_map = {
        "CE": "#8e44ad",
        "Det": "#f39c12",
        "Demod": "#2ecc71",
        "TENN": "#3498db",
        "MTE2": "#e74c3c",
        "MTE3": "#16a085",
        "Barrier/Other": "#95a5a6",
    }

    row_order = _track_order()
    row_to_y = {row: idx for idx, row in enumerate(row_order)}

    fig, ax = plt.subplots(figsize=(24, 14))
    bar_h = 0.74

    for _, rec in df.iterrows():
        row = (int(rec["core_id"]), rec["engine"])
        y = row_to_y[row]
        op = rec["operator"]
        ax.broken_barh(
            [(float(rec["start"]), float(rec["duration"]))],
            (y - bar_h / 2.0, bar_h),
            facecolors=color_map.get(op, "#7f8c8d"),
            edgecolors="none",
            alpha=0.95,
            zorder=2,
        )

    yticks = list(range(len(row_order)))
    ylabels = [f"Core {core} | {engine}" for core, engine in row_order]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    x_min = float(df["start"].min())
    x_max_data = float(df["end"].max())
    x_max = float(args.xmax) if args.xmax is not None else x_max_data

    ax.set_xlim(x_min, x_max * 1.03)
    plot_x_max = x_max * 1.16
    ax.set_xlim(x_min, plot_x_max)

    ax.set_xlabel("Cycles", fontsize=30)
    ax.set_ylabel("Micro-architecture Tracks", fontsize=30)
    ax.set_title("Two-Core Timeline (Data-driven): Core 0 & Core 1", fontsize=36, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=24)

    legend_items = [
        Patch(facecolor=color_map["CE"], label="CE"),
        Patch(facecolor=color_map["Det"], label="Det"),
        Patch(facecolor=color_map["Demod"], label="Demod"),
        Patch(facecolor=color_map["TENN"], label="TENN"),
        Patch(facecolor=color_map["MTE2"], label="MTE2 (Load)"),
        Patch(facecolor=color_map["MTE3"], label="MTE3 (Store)"),
        Patch(facecolor=color_map["Barrier/Other"], label="Barrier/Other"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=22)

    core0_center = (row_to_y[(0, "Cube Unit")] + row_to_y[(0, "MTE2 (Load)")]) / 2.0
    core1_center = (row_to_y[(1, "Cube Unit")] + row_to_y[(1, "MTE2 (Load)")]) / 2.0

    x_anno = x_min + 1.03 * (x_max - x_min)
    ax.annotate(
        "",
        xy=(x_anno, core0_center),
        xytext=(x_anno, core1_center),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.6),
    )
    # SPMD注释左移并上提
    ax.text(
        x_anno - 0.03 * (plot_x_max - x_min),
        (core0_center + core1_center) / 2.0 - 0.7,
        "SPMD Execution:\nSynchronized across 24 cores",
        ha="right",
        va="center",
        fontsize=24,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95),
    )

    cube_df = df[df["engine"] == "Cube Unit"]
    vec_df = df[(df["engine"] == "Vector Unit") & (df["name"].str.upper().str.contains("NS_R"))]
    if not cube_df.empty and not vec_df.empty:
        overlap_start = max(float(cube_df["start"].min()), float(vec_df["start"].min()))
        overlap_end = min(float(cube_df["end"].max()), float(vec_df["end"].max()))
        if overlap_end > overlap_start:
            x_mid = (overlap_start + overlap_end) / 2.0
            y_mid = (row_to_y[(0, "Cube Unit")] + row_to_y[(0, "Vector Unit")]) / 2.0
            y_cube = row_to_y[(0, "Cube Unit")]
            y_vec = row_to_y[(0, "Vector Unit")]
            y_text = (y_cube + y_vec) / 2.0
            x_text = x_min + 0.73 * (x_max - x_min)
            ax.text(
                x_text,
                y_text,
                "Heterogeneous Overlap\n(Cube & Vector execute concurrently)",
                fontsize=24,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.95),
            )
            arrow_start = (x_text - 0.015 * (x_max - x_min), y_text)
            ax.annotate(
                "",
                xy=(x_mid, y_cube),
                xytext=arrow_start,
                arrowprops=dict(arrowstyle="->", color="black", lw=2.2),
            )
            ax.annotate(
                "",
                xy=(x_mid, y_vec),
                xytext=arrow_start,
                arrowprops=dict(arrowstyle="->", color="black", lw=2.2),
            )

    cube_end = float(df[df["engine"] == "Cube Unit"]["end"].max()) if not df[df["engine"] == "Cube Unit"].empty else x_max
    vec_end = float(df[df["engine"] == "Vector Unit"]["end"].max()) if not df[df["engine"] == "Vector Unit"].empty else x_max
    if vec_end > cube_end:
        idle_start = cube_end
        idle_w = vec_end - cube_end
        for core_id in [0, 1]:
            y = row_to_y[(core_id, "Cube Unit")]
            rect = Rectangle(
                (idle_start, y - bar_h / 2.0),
                idle_w,
                bar_h,
                fill=False,
                edgecolor="#c0392b",
                linestyle="--",
                linewidth=1.6,
                zorder=3,
            )
            ax.add_patch(rect)
        ax.text(
            idle_start + idle_w * 0.03,
            row_to_y[(1, "Cube Unit")] - 0.55,
            "Cube Idling (Execution Imbalance)",
            color="#c0392b",
            fontsize=24,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.9),
        )

    total_latency = int(round(x_max_data))
    ax.axvline(x_max_data, color="black", linestyle="--", linewidth=1.3, zorder=1)
    # Total Slot Latency上移
    # ax.text(
    #     x_max_data - 0.35 * (plot_x_max - x_min),
    #     1.8,
    #     f"Total Slot Latency: {total_latency:,} cy",
    #     fontsize=24,
    #     color="black",
    #     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95),
    # )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
