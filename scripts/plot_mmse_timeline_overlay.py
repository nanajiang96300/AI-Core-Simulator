#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "mmse"

CSV_256 = RESULTS_DIR / "profiling_log_mmse_910b_256x32.csv"
CSV_512 = RESULTS_DIR / "profiling_log_mmse_910b_512x32.csv"
OUT_PNG = RESULTS_DIR / "pipeline_mmse_910b_256x32_vs_512x32_overlay.png"


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


def _union_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            if e > cur_e:
                cur_e = e
    merged.append((cur_s, cur_e))
    return merged


def _load_phase_intervals(csv_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    # 保留有意义的持续时间
    df["Duration"] = df["EndCycle"] - df["StartCycle"]
    df = df[df["Duration"] > 0]

    # 只看 MMSE 主要阶段和 load/store，去掉大量零星的 barrier / 向量辅助阶段
    # 256x32 和 512x32 的命名集合不同，这里取两者的“主干阶段”交并集，便于对比。
    name = df["Name"].astype(str)

    # 核心算子阶段（GEMM / INV / APPLY / HtH / WH / WY 等）
    keep_keywords = [
        "MMSE_HtH",
        "MMSE_G_PLUS_SIGMA",
        "MMSE_INV_T",
        "MMSE_INV_R",
        "MMSE_INV_X",
        "MMSE_NS_T",
        "MMSE_NS_R",
        "MMSE_NS_X",
        "MMSE_APPLY_GEMM",
        "MMSE_APPLY_ADD",
        "MMSE_WH",
        "MMSE_WY",
    ]

    # 一律保留 Load / Store 方便看首尾
    mask_core = False
    for kw in keep_keywords:
        mask_core |= name.str.contains(kw)

    mask = mask_core | name.isin(["Load", "Store"])
    df = df[mask]

    phases: Dict[str, List[Tuple[int, int]]] = {}
    for name, sub in df.groupby("Name"):
        intervals = list(zip(sub["StartCycle"].astype(int), sub["EndCycle"].astype(int)))
        phases[name] = _union_intervals(intervals)
    return phases


def main() -> None:
    phases_256 = _load_phase_intervals(CSV_256)
    phases_512 = _load_phase_intervals(CSV_512)

    # 统一相位顺序：按第一次出现时间排序，便于对比
    first_start: Dict[str, int] = {}
    for name, intervals in {**phases_256, **phases_512}.items():
        if not intervals:
            continue
        first_start[name] = min(s for s, _ in intervals)
    phase_order = sorted(first_start.keys(), key=lambda n: first_start[n])

    if not phase_order:
        raise SystemExit("No MMSE phases found in CSVs.")

    # 构建 y 轴
    y_pos = {name: idx for idx, name in enumerate(phase_order)}

    fig_height = max(4, len(phase_order) * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 256x32: 蓝色，半透明
    for name, intervals in phases_256.items():
        y = y_pos.get(name)
        if y is None:
            continue
        xranges = [(s, e - s) for s, e in intervals]
        if not xranges:
            continue
        ax.broken_barh(xranges, (y - 0.2, 0.4), facecolors=(0.2, 0.4, 0.8, 0.4), edgecolors='none', label="256x32" if name == phase_order[0] else "")

    # 512x32: 橙色，半透明
    for name, intervals in phases_512.items():
        y = y_pos.get(name)
        if y is None:
            continue
        xranges = [(s, e - s) for s, e in intervals]
        if not xranges:
            continue
        ax.broken_barh(xranges, (y - 0.2, 0.4), facecolors=(1.0, 0.5, 0.0, 0.4), edgecolors='none', label="512x32" if name == phase_order[0] else "")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(phase_order)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("MMSE Phase")
    ax.set_title("MMSE 256x32 vs 512x32: Phase Timeline Overlay")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # 去重 legend 项
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen.add(l)
                uniq_handles.append(h)
                uniq_labels.append(l)
        ax.legend(uniq_handles, uniq_labels, loc="upper right")

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)


if __name__ == "__main__":
    main()
