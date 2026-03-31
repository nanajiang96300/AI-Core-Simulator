#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from export_operator_cycle_table import export_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LDL detailed cycle table.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="auto", choices=["auto", "ldl_block", "ldl_noblock"])
    parser.add_argument("--matrix-m", type=int, default=64)
    parser.add_argument("--matrix-u", type=int, default=16)
    parser.add_argument("--reducer", choices=["median", "mean", "max", "sum"], default="median")
    parser.add_argument("--core-prefix", default="")
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    summary_output = args.summary_output
    if not summary_output:
        summary_output = str(Path(args.output).with_name(Path(args.output).stem + "_major_summary.csv"))

    mode, rows, major_rows = export_table(
        trace_path=Path(args.trace),
        output_path=Path(args.output),
        summary_output_path=Path(summary_output),
        mode=args.mode,
        matrix_m=args.matrix_m,
        matrix_u=args.matrix_u,
        reducer=args.reducer,
        core_prefix=args.core_prefix,
    )
    print(f"mode={mode}")
    print(f"detail_rows={rows}")
    print(f"major_rows={major_rows}")
    print(f"output={args.output}")
    print(f"summary_output={summary_output}")


if __name__ == "__main__":
    main()
