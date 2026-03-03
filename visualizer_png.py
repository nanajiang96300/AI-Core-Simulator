#!/usr/bin/env python3

import argparse
import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# --- 保持之前的辅助函数不变 ---

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


def _parse_unit(df: pd.DataFrame) -> pd.DataFrame:
    # 简单的预处理优化：直接操作 Series 字符串比循环快，但为了兼容性保留逻辑，
    # 这里的瓶颈不在解析，而是在绘图，所以维持原样即可。
    cores: List[str] = []
    engines: List[str] = []
    rows: List[str] = []

    for raw in df["Unit"].astype(str):
        parts = raw.split("_", 1)
        if len(parts) == 2:
            core, engine = parts
        else:
            core, engine = "Core0", parts[0]

        if engine == "Cube":
            engine_label = "CubeCore"
        elif engine == "Vector":
            engine_label = "VectorCore"
        elif engine == "MTE2":
            engine_label = "MTE2 (Load)"
        elif engine == "MTE3":
            engine_label = "MTE3 (Store)"
        else:
            engine_label = engine

        cores.append(core)
        engines.append(engine_label)
        rows.append(f"{core}_{engine_label}")

    df["Core"] = cores
    df["Engine"] = engines
    df["Row"] = rows
    return df


def _build_row_order(df: pd.DataFrame) -> List[str]:
    def core_index(core: str) -> int:
        digits = "".join(ch for ch in str(core) if ch.isdigit())
        return int(digits) if digits else 0

    cores = sorted(df["Core"].unique(), key=core_index)
    engine_order = ["CubeCore", "VectorCore", "MTE2 (Load)", "MTE3 (Store)"]

    existing = set(df["Row"].unique())
    order: List[str] = []
    for core in cores:
        for eng in engine_order:
            label = f"{core}_{eng}"
            if label in existing:
                order.append(label)
    return order

# --- 主函数优化 ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast render ONNXim profiling CSV using vectorized plotting."
    )
    parser.add_argument("-i", "--input", default="profiling_log.csv", help="Input CSV file")
    parser.add_argument("-o", "--output", default="pipeline_timeline.png", help="Output PNG file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    print("Loading data...")
    df = pd.read_csv(args.input)
    df = _normalize_columns(df)
    df = _parse_unit(df)
    
    # 预计算 Duration
    df["Duration"] = df["EndCycle"] - df["StartCycle"]
    # 过滤掉 duration <= 0 的无效数据，避免绘图报错或浪费资源
    df = df[df["Duration"] > 0]

    row_order = _build_row_order(df)
    if not row_order:
        row_order = sorted(df["Row"].unique())

    row_to_y = {row: idx for idx, row in enumerate(row_order)}

    # Colors
    color_map = {
        "CubeCore": "#2ca02c",      # green
        "VectorCore": "#98df8a",    # light green
        "MTE2 (Load)": "#ff7f7f",   # red
        "MTE3 (Store)": "#1f77b4",  # blue
    }

    # 动态调整高度，防止行太多挤在一起
    fig_height = max(4, len(row_order) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    print("Rendering timeline...")

    # ==========================================
    # 核心优化：使用 GroupBy 进行批量绘图
    # ==========================================
    # 按 "Row" (Y轴位置) 和 "Engine" (颜色) 分组
    # 这样每一组只需要调用一次 broken_barh
    grouped = df.groupby(["Row", "Engine"])

    for (row_label, engine), group in grouped:
        y = row_to_y.get(row_label)
        if y is None:
            continue
        
        # 获取颜色，如果未知类型则用灰色
        color = color_map.get(engine, "#7f7f7f")
        
        # 提取 (start, width) 列表
        # list(zip(...)) 是极快的数据转换方式
        xranges = list(zip(group["StartCycle"], group["Duration"]))
        
        # 一次性绘制该行、该颜色的所有矩形
        # edgecolors='none' 非常重要！
        # 当有几万个小矩形时，描边(edge)会导致严重的锯齿和性能下降，
        # 去掉描边能让图更清晰且渲染更快。
        ax.broken_barh(xranges, (y - 0.4, 0.8), facecolors=color, edgecolors='none')

    # 设置图表格式
    ax.set_yticks(list(row_to_y.values()))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Core / Unit")
    ax.set_title(f"ONNXim Pipeline Timeline ({len(df)} events)")
    
    # 设置 X 轴范围，避免太多留白
    if not df.empty:
        ax.set_xlim(df["StartCycle"].min(), df["EndCycle"].max())

    print(f"Saving to {args.output}...")
    plt.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("Done.")

if __name__ == "__main__":
    main()