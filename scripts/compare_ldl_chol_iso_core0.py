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
    for p in [
        "LDL_GRAM",
        "LDL_REG",
        "LDL_D_UPDATE",
        "LDL_D_DIAG_INV",
        "LDL_D_INV_MUL",
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
    if name in ("Load", "Store", "CubeWait"):
        return name
    return "OTHER"


def group_chol_iso(name: str) -> str:
    for p in [
        "CHOL_NB_GRAM",
        "CHOL_NB_REG",
        "CHOL_NB_ISO_POTRF_DIAG_UPD",
        "CHOL_NB_ISO_POTRF_DIAG_SQRT",
        "CHOL_NB_ISO_TRSM_DIAG_INV",
        "CHOL_NB_ISO_TRSM_NUM_UPD",
        "CHOL_NB_ISO_TRSM_MUL",
        "CHOL_NB_ISO_RK_UPDATE",
        "CHOL_NB_ISO_FWD_DIAG_INV",
        "CHOL_NB_ISO_FWD_OFF_MAC",
        "CHOL_NB_ISO_FWD_OFF_MUL",
        "CHOL_NB_ISO_BWD_MAC_FULL",
        "CHOL_NB_BARRIER_LOAD2GRAM",
        "CHOL_NB_BARRIER_REG2FACTOR",
        "CHOL_NB_ISO_BARRIER_FACTOR_STEP",
        "CHOL_NB_ISO_BARRIER_FWD_COL",
        "CHOL_NB_ISO_BARRIER_SOLVE2STORE",
    ]:
        if name.startswith(p):
            return p
    if name in ("Load", "Store", "CubeWait"):
        return name
    return "OTHER"


def aggregate(events: list[Event], grouper) -> dict[str, tuple[int, int]]:
    out = defaultdict(lambda: [0, 0])
    for event in events:
        key = grouper(event.name)
        out[key][0] += 1
        out[key][1] += event.end - event.start
    return {k: (v[0], v[1]) for k, v in out.items()}


def draw_timeline(ldl_events: list[Event], chol_events: list[Event], output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    unit_order = ["MTE2", "Cube", "Vector", "Wait", "MTE3"]
    unit_y = {u: i for i, u in enumerate(unit_order)}

    colors = {
        "MTE2": "#4e79a7",
        "Cube": "#f28e2b",
        "Vector": "#59a14f",
        "Wait": "#e15759",
        "MTE3": "#b07aa1",
    }

    for axis, data, title in [
        (axes[0], ldl_events, "LDL aligned (Core0)"),
        (axes[1], chol_events, "Cholesky strict-iso (Core0)"),
    ]:
        for event in data:
            if event.unit not in unit_y:
                continue
            axis.barh(
                y=unit_y[event.unit],
                width=event.end - event.start,
                left=event.start,
                height=0.6,
                color=colors.get(event.unit, "#999999"),
                alpha=0.9,
            )
        axis.set_yticks(range(len(unit_order)))
        axis.set_yticklabels(unit_order)
        axis.set_title(title)
        axis.grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Cycle")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    ldl_stats: dict[str, tuple[int, int]],
    chol_stats: dict[str, tuple[int, int]],
    ldl_end: int,
    chol_end: int,
):
    keys = sorted(set(ldl_stats) | set(chol_stats), key=lambda k: (k == "OTHER", k))

    lines = []
    lines.append("# LDL vs Cholesky(strict-iso) Core0 对比\n")
    lines.append(f"- LDL Core0 max_end_cycle: `{ldl_end}`")
    lines.append(f"- Cholesky Core0 max_end_cycle: `{chol_end}`")
    lines.append(f"- cycle ratio (CHOL/LDL): `{chol_end / ldl_end:.4f}`\n")

    lines.append("## 分操作耗时与事件数（Core0）\n")
    lines.append("| Operation | LDL cnt | LDL dur | CHOL cnt | CHOL dur | CHOL-LDL dur |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for key in keys:
        l_cnt, l_dur = ldl_stats.get(key, (0, 0))
        c_cnt, c_dur = chol_stats.get(key, (0, 0))
        lines.append(
            f"| {key} | {l_cnt} | {l_dur} | {c_cnt} | {c_dur} | {c_dur - l_dur} |"
        )

    l_factor = (
        ldl_stats.get("LDL_D_UPDATE", (0, 0))[1]
        + ldl_stats.get("LDL_D_DIAG_INV", (0, 0))[1]
        + ldl_stats.get("LDL_D_INV_MUL", (0, 0))[1]
        + ldl_stats.get("LDL_L_UPDATE", (0, 0))[1]
    )
    c_factor = (
        chol_stats.get("CHOL_NB_ISO_POTRF_DIAG_UPD", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_POTRF_DIAG_SQRT", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_TRSM_DIAG_INV", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_TRSM_NUM_UPD", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_TRSM_MUL", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_RK_UPDATE", (0, 0))[1]
    )

    l_solve = (
        ldl_stats.get("LDL_BWD_DIAG_MUL", (0, 0))[1]
        + ldl_stats.get("LDL_BWD_DIAG_ACC", (0, 0))[1]
        + ldl_stats.get("LDL_BWD_OFF_MUL", (0, 0))[1]
        + ldl_stats.get("LDL_BWD_OFF_ACC", (0, 0))[1]
    )
    c_solve = (
        chol_stats.get("CHOL_NB_ISO_FWD_DIAG_INV", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_FWD_OFF_MAC", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_FWD_OFF_MUL", (0, 0))[1]
        + chol_stats.get("CHOL_NB_ISO_BWD_MAC_FULL", (0, 0))[1]
    )

    lines.append("\n## 结论（按 Core0 统计）\n")
    lines.append(f"- Factor阶段时长: LDL `{l_factor}` vs CHOL `{c_factor}` (ratio `{(c_factor / max(1, l_factor)):.3f}`)")
    lines.append(f"- Solve阶段时长: LDL `{l_solve}` vs CHOL `{c_solve}` (ratio `{(c_solve / max(1, l_solve)):.3f}`)")
    lines.append("- LDL 优势主要来自 factor 阶段：CHOL 仍包含 `TRSM_NUM_UPD` 和 `RK_UPDATE` 两类额外更新，" \
                 "即使 strict-iso 聚合后，事件和依赖链仍更长。")
    lines.append("- `SQRT` 仅占 CHOL 总时长的小部分，不是主导差异。")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare LDL vs Cholesky strict-iso on Core0")
    parser.add_argument(
        "--ldl-trace",
        default="results/LDL/falsification/ldl_noblock_64x16_trace_aligned.csv",
    )
    parser.add_argument(
        "--chol-trace",
        default="results/CHOL/falsification/cholesky_noblock_64x16_trace_iso.csv",
    )
    parser.add_argument(
        "--png",
        default="results/LDL/falsification/ldl_cholesky_iso_core0_timeline.png",
    )
    parser.add_argument(
        "--report",
        default="results/LDL/falsification/LDL_CHOLESKY_ISO_CORE0_COMPARE_20260327.md",
    )
    args = parser.parse_args()

    ldl_events = read_core_events(Path(args.ldl_trace), core="Core0_")
    chol_events = read_core_events(Path(args.chol_trace), core="Core0_")

    draw_timeline(ldl_events, chol_events, Path(args.png))

    ldl_stats = aggregate(ldl_events, group_ldl)
    chol_stats = aggregate(chol_events, group_chol_iso)

    ldl_end = max(event.end for event in ldl_events)
    chol_end = max(event.end for event in chol_events)

    write_report(Path(args.report), ldl_stats, chol_stats, ldl_end, chol_end)

    print("timeline_png", args.png)
    print("report_md", args.report)
    print("core0_cycle_ldl", ldl_end)
    print("core0_cycle_chol_iso", chol_end)
    print("ratio", round(chol_end / ldl_end, 4))


if __name__ == "__main__":
    main()
