#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def suffix_unit(unit: str) -> str:
    if "_" in unit:
        return unit.split("_", 1)[1]
    return unit


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-step cycle statistics from simulator trace CSV.")
    parser.add_argument("--trace", required=True, help="Trace CSV path")
    parser.add_argument("--output", required=True, help="Output step statistics CSV path")
    parser.add_argument("--core-prefix", default="", help="Optional core filter, e.g. Core0_")
    parser.add_argument("--drop-events", default="Load,Store,CubeWait", help="Comma-separated event names to exclude")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.output)

    drop_events = {name.strip() for name in args.drop_events.split(",") if name.strip()}

    buckets: dict[str, list[int]] = defaultdict(list)
    unit_set: dict[str, set[str]] = defaultdict(set)
    start_min: dict[str, int] = {}
    end_max: dict[str, int] = {}

    with trace_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if name in drop_events:
                continue
            unit = row["unit"]
            if args.core_prefix and not unit.startswith(args.core_prefix):
                continue

            start = int(row["start_cycle"])
            end = int(row["end_cycle"])
            dur = end - start
            if dur < 0:
                continue

            buckets[name].append(dur)
            unit_set[name].add(suffix_unit(unit))
            start_min[name] = min(start_min.get(name, start), start)
            end_max[name] = max(end_max.get(name, end), end)

    rows: list[dict[str, str]] = []
    for idx, name in enumerate(sorted(buckets.keys())):
        values = buckets[name]
        rows.append(
            {
                "step_idx": str(idx),
                "event_name": name,
                "unit_types": "|".join(sorted(unit_set[name])),
                "count": str(len(values)),
                "cycle_median": f"{statistics.median(values):.2f}",
                "cycle_mean": f"{statistics.mean(values):.2f}",
                "cycle_max": str(max(values)),
                "cycle_min": str(min(values)),
                "cycle_sum": str(sum(values)),
                "start_min": str(start_min[name]),
                "end_max": str(end_max[name]),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "event_name",
                "unit_types",
                "count",
                "cycle_median",
                "cycle_mean",
                "cycle_max",
                "cycle_min",
                "cycle_sum",
                "start_min",
                "end_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
