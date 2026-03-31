#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Event:
    unit: str
    name: str
    start: int
    end: int


def has_time_overlap(events_a: list[Event], events_b: list[Event]) -> bool:
    for event_a in events_a:
        for event_b in events_b:
            if max(event_a.start, event_b.start) < min(event_a.end, event_b.end):
                return True
    return False


def read_core0(path: Path):
    events: list[Event] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = row["unit"].strip('"')
            if not unit.startswith("Core0_"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cholesky Core0 timeline with Scalar lane")
    parser.add_argument("--trace", default="results/CHOL/falsification/cholesky_noblock_64x16_trace_iso_scalar.csv")
    parser.add_argument("--png", default="results/CHOL/falsification/cholesky_noblock_64x16_timeline_iso_scalar.png")
    parser.add_argument("--title", default="Cholesky (no-block, iso) - Core0")
    args = parser.parse_args()

    events = read_core0(Path(args.trace))

    cube_events = [event for event in events if event.unit == "Cube"]
    wait_events = [event for event in events if event.unit == "Wait"]
    can_share_lane = not has_time_overlap(cube_events, wait_events)

    unit_order = ["MTE2", "Cube", "Wait", "Vector", "Scalar", "MTE3"]
    if can_share_lane:
        unit_order = ["MTE2", "Cube/Wait", "Vector", "Scalar", "MTE3"]

    unit_y = {unit: idx for idx, unit in enumerate(unit_order)}
    colors = {
        "MTE2": "#4e79a7",
        "Cube": "#f28e2b",
        "Vector": "#59a14f",
        "Scalar": "#76b7b2",
        "Wait": "#e15759",
        "MTE3": "#b07aa1",
    }

    xmax = max((event.end for event in events), default=1)

    fig, ax = plt.subplots(figsize=(18, 4.8))
    for event in events:
        draw_unit = event.unit
        if can_share_lane and event.unit in {"Cube", "Wait"}:
            draw_unit = "Cube/Wait"

        if draw_unit not in unit_y:
            continue

        ax.barh(
            y=unit_y[draw_unit],
            width=event.end - event.start,
            left=event.start,
            height=0.48,
            color=colors[event.unit],
            alpha=0.92,
            edgecolor="#ffffff",
            linewidth=0.15,
        )

    ax.set_yticks(range(len(unit_order)))
    ax.set_yticklabels(unit_order)
    ax.set_xlim(0, int(xmax * 1.02))
    ax.set_xlabel("Cycle")
    ax.set_title(args.title)
    ax.grid(axis="x", linestyle="--", alpha=0.28)

    out = Path(args.png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=240)
    plt.close(fig)

    print("layout", "shared-cube-wait" if can_share_lane else "separate-cube-wait")
    print("png", out)


if __name__ == "__main__":
    main()
