#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class Event:
    unit: str
    name: str
    start: int
    end: int


def read_events(path: Path, core_prefix: str = "Core0_") -> List[Event]:
    events: List[Event] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = row["unit"].strip('"')
            if not unit.startswith(core_prefix):
                continue
            events.append(
                Event(
                    unit=unit.split("_", 1)[1],
                    name=row["name"].strip('"'),
                    start=int(row["start_cycle"]),
                    end=int(row["end_cycle"]),
                )
            )
    return events


def global_max_end(path: Path) -> int:
    max_end = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            end = int(row["end_cycle"])
            if end > max_end:
                max_end = end
    return max_end


def unit_share(events: List[Event]) -> Dict[str, float]:
    dur = {"MTE2": 0, "Cube": 0, "Vector": 0, "Scalar": 0, "Wait": 0, "MTE3": 0}
    total = 0
    for event in events:
        d = event.end - event.start
        total += d
        if event.unit in dur:
            dur[event.unit] += d
    if total == 0:
        return {key: 0.0 for key in dur}
    return {key: (value * 100.0 / total) for key, value in dur.items()}


def draw(events_by_name: List[Tuple[str, List[Event]]], out_png: Path) -> None:
    unit_order = ["MTE2", "Cube", "Vector", "Scalar", "Wait", "MTE3"]
    unit_y = {unit: idx for idx, unit in enumerate(unit_order)}
    colors = {
        "MTE2": "#4e79a7",
        "Cube": "#f28e2b",
        "Vector": "#59a14f",
        "Scalar": "#76b7b2",
        "Wait": "#e15759",
        "MTE3": "#b07aa1",
    }

    fig, axes = plt.subplots(len(events_by_name), 1, figsize=(19, 11), sharex=True)
    if len(events_by_name) == 1:
        axes = [axes]

    xmax = 0
    for _, events in events_by_name:
        if events:
            xmax = max(xmax, max(event.end for event in events))

    for ax, (title, events) in zip(axes, events_by_name):
        for event in events:
            if event.unit not in unit_y:
                continue
            ax.barh(
                y=unit_y[event.unit],
                width=event.end - event.start,
                left=event.start,
                height=0.56,
                color=colors[event.unit],
                alpha=0.92,
            )
        ax.set_yticks(range(len(unit_order)))
        ax.set_yticklabels(unit_order)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.28)
        ax.set_xlim(0, int(xmax * 1.03) if xmax > 0 else 1)

    axes[-1].set_xlabel("Cycle")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline compare: Cholesky block vs standard LDL vs blocked LDL")
    parser.add_argument("--chol-trace", default="results/CHOL/falsification/cholesky_block_64x16_trace_opt.csv")
    parser.add_argument("--ldl-std-trace", default="results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv")
    parser.add_argument("--ldl-block-trace", default="results/LDL/falsification/ldl_block_64x16_trace_moderate3.csv")
    parser.add_argument("--chol-label", default="Cholesky block(opt) - Core0")
    parser.add_argument("--ldl-std-label", default="LDL standard(no-block) - Core0")
    parser.add_argument("--ldl-block-label", default="LDL block(moderate3) - Core0")
    parser.add_argument("--chol-case", default="CHOL_BLOCK_OPT")
    parser.add_argument("--ldl-std-case", default="LDL_STANDARD_NOBLOCK")
    parser.add_argument("--ldl-block-case", default="LDL_BLOCK_MODERATE3")
    parser.add_argument("--png", default="results/LDL/falsification/chol_stdldl_blockldl_timeline_compare_20260327.png")
    parser.add_argument("--summary", default="results/LDL/falsification/chol_stdldl_blockldl_timeline_compare_20260327.csv")
    args = parser.parse_args()

    chol_path = Path(args.chol_trace)
    ldl_std_path = Path(args.ldl_std_trace)
    ldl_block_path = Path(args.ldl_block_trace)

    chol_events = read_events(chol_path)
    ldl_std_events = read_events(ldl_std_path)
    ldl_block_events = read_events(ldl_block_path)

    draw(
        [
            (args.chol_label, chol_events),
            (args.ldl_std_label, ldl_std_events),
            (args.ldl_block_label, ldl_block_events),
        ],
        Path(args.png),
    )

    rows = []
    for name, path, events in [
        (args.chol_case, chol_path, chol_events),
        (args.ldl_std_case, ldl_std_path, ldl_std_events),
        (args.ldl_block_case, ldl_block_path, ldl_block_events),
    ]:
        shares = unit_share(events)
        rows.append(
            {
                "case": name,
                "global_max_end": global_max_end(path),
                "core0_max_end": max((event.end for event in events), default=0),
                "core0_cube_share_pct": round(shares["Cube"], 2),
                "core0_vector_share_pct": round(shares["Vector"], 2),
                "core0_scalar_share_pct": round(shares["Scalar"], 2),
                "core0_mte2_share_pct": round(shares["MTE2"], 2),
                "core0_wait_share_pct": round(shares["Wait"], 2),
                "core0_mte3_share_pct": round(shares["MTE3"], 2),
            }
        )

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("png", args.png)
    print("summary", args.summary)


if __name__ == "__main__":
    main()
