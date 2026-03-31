#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class Event:
    unit: str
    name: str
    start: int
    end: int


def read_core_events(path: Path, core: str = "Core0_") -> list[Event]:
    events: list[Event] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = row["unit"].strip('"')
            if not unit.startswith(core):
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


def group_ldl(name: str) -> str:
    if name in ("Load", "Store", "CubeWait"):
        return name
    for p in [
        "LDL_GRAM",
        "LDL_REG",
        "LDL_D_UPDATE",
        "LDL_D_DIAG_INV",
        "LDL_D_INV_MUL",
        "LDL_D_INV",
        "LDL_L_UPDATE",
        "LDL_BWD_DIAG_MUL",
        "LDL_BWD_DIAG_ACC",
        "LDL_BWD_OFF_MUL",
        "LDL_BWD_OFF_ACC",
        "LDL_BARRIER_LOAD2GRAM",
        "LDL_BARRIER_GRAM2REG",
        "LDL_BARRIER_REG2BLDL",
        "LDL_BARRIER_BLDL_STEP",
        "LDL_BARRIER_BWD_DIAG2OFF",
        "LDL_BARRIER_BWD_COL",
        "LDL_BARRIER_BWD2STORE",
    ]:
        if name.startswith(p):
            return p
    return "OTHER"


def group_chol(name: str) -> str:
    if name in ("Load", "Store", "CubeWait"):
        return name
    for p in [
        "CHOL_GRAM",
        "CHOL_REG",
        "CHOL_POTRF_DIAG_UPD",
        "CHOL_POTRF_DIAG_SQRT",
        "CHOL_TRSM_DIAG_INV",
        "CHOL_TRSM_NUM_UPD",
        "CHOL_TRSM_DIV",
        "CHOL_TRSM_MUL",
        "CHOL_RK_UPDATE",
        "CHOL_FWD_DIAG_INV",
        "CHOL_FWD_OFF_MAC",
        "CHOL_FWD_OFF_UPD",
        "CHOL_FWD_OFF_MUL",
        "CHOL_BWD_MAC_FULL",
        "CHOL_BARRIER_LOAD2GRAM",
        "CHOL_BARRIER_REG2FACTOR",
        "CHOL_BARRIER_FACTOR_STEP",
        "CHOL_BARRIER_FWD_COL",
        "CHOL_BARRIER_SOLVE2STORE",
    ]:
        if name.startswith(p):
            return p
    return "OTHER"


def aggregate(events: list[Event], grouper):
    stats = defaultdict(lambda: [0, 0])
    for event in events:
        key = grouper(event.name)
        stats[key][0] += 1
        stats[key][1] += event.end - event.start
    return {key: (value[0], value[1]) for key, value in stats.items()}


def draw_timeline(ldl_events: list[Event], chol_events: list[Event], output_png: Path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    unit_order = ["MTE2", "Cube", "Vector", "Wait", "MTE3"]
    unit_y = {unit: idx for idx, unit in enumerate(unit_order)}
    colors = {
        "MTE2": "#4e79a7",
        "Cube": "#f28e2b",
        "Vector": "#59a14f",
        "Wait": "#e15759",
        "MTE3": "#b07aa1",
    }

    for ax, events, title in [
        (axes[0], ldl_events, "LDL block (Core0)"),
        (axes[1], chol_events, "Cholesky block (Core0)"),
    ]:
        for event in events:
            if event.unit not in unit_y:
                continue
            ax.barh(
                y=unit_y[event.unit],
                width=event.end - event.start,
                left=event.start,
                height=0.6,
                color=colors.get(event.unit, "#999999"),
                alpha=0.9,
            )
        ax.set_yticks(range(len(unit_order)))
        ax.set_yticklabels(unit_order)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Cycle")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    ldl_stats: dict[str, tuple[int, int]],
    chol_stats: dict[str, tuple[int, int]],
    ldl_end: int,
    chol_end: int,
):
    keys = sorted(set(ldl_stats) | set(chol_stats))

    lines = []
    lines.append("# Blocked Cholesky vs Blocked LDL (Core0)\n")
    lines.append(f"- LDL Core0 max_end_cycle: `{ldl_end}`")
    lines.append(f"- Cholesky Core0 max_end_cycle: `{chol_end}`")
    lines.append(f"- ratio (CHOL/LDL): `{chol_end / ldl_end:.4f}`\n")

    lines.append("## Per-operation table (Core0)\n")
    lines.append("| Operation | LDL cnt | LDL dur | CHOL cnt | CHOL dur | CHOL-LDL dur |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for key in keys:
        l_cnt, l_dur = ldl_stats.get(key, (0, 0))
        c_cnt, c_dur = chol_stats.get(key, (0, 0))
        lines.append(f"| {key} | {l_cnt} | {l_dur} | {c_cnt} | {c_dur} | {c_dur - l_dur} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare blocked LDL and blocked Cholesky on Core0")
    parser.add_argument("--ldl-trace", default="results/LDL/falsification/ldl_block_64x16_trace.csv")
    parser.add_argument("--chol-trace", default="results/CHOL/falsification/cholesky_block_64x16_trace.csv")
    parser.add_argument("--png", default="results/LDL/falsification/ldl_cholesky_block_core0_timeline.png")
    parser.add_argument("--report", default="results/LDL/falsification/LDL_CHOLESKY_BLOCK_CORE0_COMPARE_20260327.md")
    args = parser.parse_args()

    ldl_events = read_core_events(Path(args.ldl_trace), core="Core0_")
    chol_events = read_core_events(Path(args.chol_trace), core="Core0_")

    draw_timeline(ldl_events, chol_events, Path(args.png))

    ldl_stats = aggregate(ldl_events, group_ldl)
    chol_stats = aggregate(chol_events, group_chol)

    ldl_end = max(event.end for event in ldl_events)
    chol_end = max(event.end for event in chol_events)

    write_report(Path(args.report), ldl_stats, chol_stats, ldl_end, chol_end)

    print("timeline_png", args.png)
    print("report_md", args.report)
    print("core0_cycle_ldl", ldl_end)
    print("core0_cycle_chol", chol_end)
    print("ratio", round(chol_end / ldl_end, 4))


if __name__ == "__main__":
    main()
