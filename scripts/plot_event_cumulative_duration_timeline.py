#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Event:
    unit: str
    name: str
    start: int
    end: int


UNIT_ORDER = ["MTE2", "Cube", "Vector", "Wait", "MTE3"]
COLORS = {
    "MTE2": "#4e79a7",
    "Cube": "#f28e2b",
    "Vector": "#59a14f",
    "Wait": "#e15759",
    "MTE3": "#b07aa1",
}


def read_events(trace_path: Path, core_prefix: str | None = None) -> list[Event]:
    events: list[Event] = []
    with trace_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_unit = row["unit"].strip('"')
            if core_prefix is not None and not raw_unit.startswith(core_prefix):
                continue
            unit = raw_unit.split("_", 1)[1] if "_" in raw_unit else raw_unit
            if unit not in UNIT_ORDER:
                continue
            events.append(
                Event(
                    unit=unit,
                    name=row["name"].strip('"'),
                    start=int(row["start_cycle"]),
                    end=int(row["end_cycle"]),
                )
            )
    return events


def build_cumulative_series(events: list[Event], max_cycle: int, n_points: int = 500):
    x = np.linspace(0, max_cycle, n_points, dtype=np.int64)
    cum = {u: np.zeros_like(x, dtype=np.float64) for u in UNIT_ORDER}

    by_unit = {u: [] for u in UNIT_ORDER}
    for event in events:
        by_unit[event.unit].append((event.start, event.end))

    for unit in UNIT_ORDER:
        intervals = by_unit[unit]
        if not intervals:
            continue
        starts = np.array([s for s, _ in intervals], dtype=np.int64)
        ends = np.array([e for _, e in intervals], dtype=np.int64)
        durs = ends - starts
        order = np.argsort(ends)
        ends_sorted = ends[order]
        durs_sorted = durs[order]
        prefix = np.cumsum(durs_sorted)

        idx = np.searchsorted(ends_sorted, x, side="right") - 1
        valid = idx >= 0
        cum_arr = np.zeros_like(x, dtype=np.float64)
        cum_arr[valid] = prefix[idx[valid]]
        cum[unit] = cum_arr

    final_dur = {u: float(cum[u][-1]) for u in UNIT_ORDER}
    total = sum(final_dur.values())
    shares = {u: (final_dur[u] / total if total > 0 else 0.0) for u in UNIT_ORDER}
    return x, cum, final_dur, shares


def merge_intervals_len(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    s, e = intervals[0]
    total = 0
    for a, b in intervals[1:]:
        if a <= e:
            e = max(e, b)
        else:
            total += e - s
            s, e = a, b
    total += e - s
    return total


def occupancy_metrics(events: list[Event], max_cycle: int):
    occ_len = {u: 0 for u in UNIT_ORDER}
    intervals_by_unit = {u: [] for u in UNIT_ORDER}
    for event in events:
        intervals_by_unit[event.unit].append((event.start, event.end))
    for unit in UNIT_ORDER:
        occ_len[unit] = merge_intervals_len(intervals_by_unit[unit])
    occ_share = {u: (occ_len[u] / max_cycle if max_cycle > 0 else 0.0) for u in UNIT_ORDER}
    return occ_len, occ_share


def plot_all(events: list[Event], out_png: Path, title: str):
    if not events:
        raise ValueError("No events to plot.")

    max_cycle = max(event.end for event in events)
    x, cum, final_dur, shares = build_cumulative_series(events, max_cycle)
    occ_len, occ_share = occupancy_metrics(events, max_cycle)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.8, 1.3], hspace=0.25)

    ax0 = fig.add_subplot(gs[0])
    y_map = {u: i for i, u in enumerate(UNIT_ORDER)}
    for event in events:
        ax0.barh(
            y=y_map[event.unit],
            width=event.end - event.start,
            left=event.start,
            height=0.6,
            color=COLORS[event.unit],
            alpha=0.9,
        )
    ax0.set_yticks(range(len(UNIT_ORDER)))
    ax0.set_yticklabels(UNIT_ORDER)
    ax0.set_xlim(0, max_cycle)
    ax0.set_title(f"{title} | Timeline Layout (Cycle Axis)")
    ax0.grid(axis="x", linestyle="--", alpha=0.3)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for unit in UNIT_ORDER:
        ax1.plot(x, cum[unit], label=unit, color=COLORS[unit], linewidth=2)
    ax1.set_ylabel("Cumulative duration (sum of event durs)")
    ax1.set_title("Cumulative Duration vs Cycle (duration-sum denominator)")
    ax1.grid(axis="both", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", ncol=5, fontsize=9)

    ax2 = fig.add_subplot(gs[2])
    units = UNIT_ORDER
    dur_vals = [shares[u] * 100.0 for u in units]
    occ_vals = [occ_share[u] * 100.0 for u in units]
    x_idx = np.arange(len(units))
    w = 0.38
    bars1 = ax2.bar(x_idx - w / 2, dur_vals, width=w, color=[COLORS[u] for u in units], alpha=0.9, label="dur-sum share")
    bars2 = ax2.bar(x_idx + w / 2, occ_vals, width=w, color=[COLORS[u] for u in units], alpha=0.35, edgecolor="black", linewidth=0.5, label="wall-time occupancy share")
    for bar, val in zip(bars1, dur_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, occ_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(units)
    ax2.set_ylabel("Share (%)")
    ymax = max(dur_vals + occ_vals) if (dur_vals or occ_vals) else 1
    ax2.set_ylim(0, ymax * 1.35)
    ax2.set_title("Two aligned denominators: duration-sum vs wall-time occupancy")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("output_png", str(out_png))
    print("max_cycle", max_cycle)
    total = sum(final_dur.values())
    print("total_duration_sum", int(total))
    for unit in UNIT_ORDER:
        inflight = (final_dur[unit] / occ_len[unit]) if occ_len[unit] > 0 else 0.0
        print(
            f"{unit}_dur", int(final_dur[unit]),
            f"{unit}_dur_share", round(shares[unit], 6),
            f"{unit}_occ_len", int(occ_len[unit]),
            f"{unit}_occ_share", round(occ_share[unit], 6),
            f"{unit}_avg_inflight", round(inflight, 3),
        )


def main():
    parser = argparse.ArgumentParser(description="Plot event timeline + cumulative duration (duration-sum denominator)")
    parser.add_argument("--trace", default="results/LDL/falsification/ldl_block_64x16_trace_opt2.csv")
    parser.add_argument("--core-prefix", default="", help="Use e.g. Core0_ to plot single core; empty means all cores.")
    parser.add_argument(
        "--out",
        default="results/LDL/falsification/ldl_block_opt2_cumulative_duration_timeline.png",
    )
    parser.add_argument("--title", default="LDL block opt2")
    args = parser.parse_args()

    core_prefix = args.core_prefix if args.core_prefix else None
    events = read_events(Path(args.trace), core_prefix=core_prefix)
    plot_all(events, Path(args.out), title=args.title + (f" ({args.core_prefix})" if args.core_prefix else " (all cores)"))


if __name__ == "__main__":
    main()
