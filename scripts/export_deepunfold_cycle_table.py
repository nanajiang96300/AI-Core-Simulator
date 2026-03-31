#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceEvent:
    name: str
    unit: str
    start_cycle: int
    end_cycle: int

    @property
    def duration(self) -> int:
        return self.end_cycle - self.start_cycle


@dataclass
class StepRule:
    key: str
    pattern: re.Pattern[str]
    onnx_op: str
    compute_op: str
    formula_template: str
    dimension_template: str


def _reduce(values: list[int], reducer: str) -> float:
    if not values:
        return 0.0
    if reducer == "median":
        return float(statistics.median(values))
    if reducer == "max":
        return float(max(values))
    if reducer == "mean":
        return float(sum(values) / len(values))
    if reducer == "sum":
        return float(sum(values))
    raise ValueError(f"Unsupported reducer: {reducer}")


def build_rules(mode: str) -> list[StepRule]:
    if mode == "du":
        return [
            StepRule("GRAM", re.compile(r"^DU_GRAM$"), "MatMul", "MATMUL", "G = H^H @ H", "{u}*{m}*{m}*{u}"),
            StepRule("REG", re.compile(r"^DU_REG$"), "Add", "VECTOR_ADD", "A = G + RegI", "{u}*{u}"),
            StepRule("AX", re.compile(r"^DU_AX_(\d+)$"), "MatMul", "MATMUL", "AX_{layer} = A @ X_{layer}", "{u}*{u}*{u}*{u}"),
            StepRule("RES", re.compile(r"^DU_RES_(\d+)_\d+$"), "Add", "VECTOR_ADD", "R_{layer} = RegI + AX_{layer}", "{u}*{u}"),
            StepRule("XNEXT", re.compile(r"^DU_XNEXT_(\d+)$"), "MatMul", "MATMUL", "Xtmp_{next_layer} = X_{layer} @ R_{layer}", "{u}*{u}*{u}*{u}"),
            StepRule("STORE_XK", re.compile(r"^DU_STORE_XK_(\d+)$"), "Add", "VECTOR_ADD", "X_{next_layer} = Xtmp_{next_layer} + RegI", "{u}*{u}"),
            StepRule("W", re.compile(r"^DU_W$"), "MatMul", "MATMUL", "W = X_last @ H^H", "{u}*{u}*{u}*{m}"),
            StepRule("XHAT", re.compile(r"^DU_XHAT$"), "MatMul", "MATMUL", "X_hat = W @ Y", "{u}*{m}*{m}*{u}"),
        ]

    if mode == "duo":
        return [
            StepRule("GRAM", re.compile(r"^DUO_GRAM$"), "MatMul", "MATMUL", "G = H^H @ H", "{u}*{m}*{m}*{u}"),
            StepRule("REG", re.compile(r"^DUO_REG$"), "Add", "VECTOR_ADD", "A = G + RegI", "{u}*{u}"),
            StepRule("INIT_XK", re.compile(r"^DUO_INIT_XK$"), "Add", "VECTOR_ADD", "X_0 = RegI + RegI", "{u}*{u}"),
            StepRule("AX", re.compile(r"^DUO_AX_(\d+)$"), "MatMul", "MATMUL", "AX_{layer} = A @ X_{layer}", "{u}*{u}*{u}*{u}"),
            StepRule("XNEXT", re.compile(r"^DUO_XNEXT_(\d+)$"), "MatMul", "MATMUL", "Xtmp_{next_layer} = X_{layer} @ A", "{u}*{u}*{u}*{u}"),
            StepRule("STORE_XK", re.compile(r"^DUO_STORE_XK_(\d+)$"), "Add", "VECTOR_ADD", "X_{next_layer} = Xtmp_{next_layer} + RegI", "{u}*{u}"),
            StepRule("VEC_CORR", re.compile(r"^DUO_VEC_CORR_(\d+)$"), "Add", "VECTOR_ADD", "Zcorr_{layer} = Z_{layer} + RegI", "{u}*{u}"),
            StepRule("VEC_MERGE", re.compile(r"^DUO_VEC_MERGE_(\d+)$"), "Add", "VECTOR_ADD", "X_{next_layer} = Zcorr_{layer} + RegI", "{u}*{u}"),
            StepRule("W", re.compile(r"^DUO_W$"), "MatMul", "MATMUL", "W = X_last @ H^H", "{u}*{u}*{u}*{m}"),
            StepRule("XHAT", re.compile(r"^DUO_XHAT$"), "MatMul", "MATMUL", "X_hat = W @ Y", "{u}*{m}*{m}*{u}"),
        ]

    if mode == "chol_nb":
        return [
            StepRule("GRAM", re.compile(r"^CHOL_NB_GRAM$"), "MatMul", "MATMUL", "G = H^H @ H", "{u}*{m}*{m}*{u}"),
            StepRule("REG", re.compile(r"^CHOL_NB_REG$"), "Add", "VECTOR_ADD", "A = G + RegI", "{u}*{u}"),
            StepRule("POTRF_DIAG_SQRT", re.compile(r"^CHOL_NB_POTRF_DIAG_SQRT_(\d+)$"), "Sqrt", "SCALAR_SQRT", "L_diag_{layer} = sqrt(A_diag_{layer})", "1"),
            StepRule("TRSM_DIV", re.compile(r"^CHOL_NB_TRSM_DIV_(\d+)_(\d+)$"), "Div", "SCALAR_DIV", "L_{i,k} = A_{i,k} / L_{k,k}", "1"),
            StepRule("RK_UPDATE", re.compile(r"^CHOL_NB_RK_UPDATE_(\d+)_(\d+)_(\d+)$"), "SubMul", "VECTOR_MAC", "A_{i,j} = A_{i,j} - L_{i,k} * L_{j,k}", "1"),
        ]

    raise ValueError(f"Unsupported mode: {mode}")


def detect_mode(events: list[TraceEvent]) -> str:
    chol = 0
    duo = 0
    du = 0
    for event in events:
        if event.name.startswith("CHOL_NB_"):
            chol += 1
        elif event.name.startswith("DUO_"):
            duo += 1
        elif event.name.startswith("DU_"):
            du += 1
    if chol >= duo and chol >= du:
        return "chol_nb"
    if duo > du:
        return "duo"
    return "du"


def read_events(path: Path, core_prefix: str) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = row["unit"]
            if core_prefix and not unit.startswith(core_prefix):
                continue
            events.append(
                TraceEvent(
                    name=row["name"],
                    unit=unit,
                    start_cycle=int(row["start_cycle"]),
                    end_cycle=int(row["end_cycle"]),
                )
            )
    return events


def sort_key(item: tuple[str, int, str]) -> tuple[int, str, tuple[int, ...], str]:
    step_key, layer_idx, token = item
    token_ints: tuple[int, ...] = tuple()
    if token:
        parts = token.split("_")
        if all(part.isdigit() for part in parts):
            token_ints = tuple(int(part) for part in parts)
    if layer_idx < 0:
        return (-1, step_key, token_ints, token)
    return (layer_idx, step_key, token_ints, token)


def parse_group_token(key: str, match: re.Match[str]) -> tuple[int, str]:
    groups = list(match.groups())
    if not groups:
        return -1, ""

    if key == "TRSM_DIV" and len(groups) >= 2:
        i, k = groups[0], groups[1]
        return int(i), f"{i}_{k}"

    if key == "RK_UPDATE" and len(groups) >= 3:
        i, j, k = groups[0], groups[1], groups[2]
        return int(i), f"{i}_{j}_{k}"

    layer = int(groups[0])
    return layer, groups[0]


def build_formula(rule: StepRule, layer_idx: int, token: str, m: int, u: int) -> str:
    if rule.key == "TRSM_DIV":
        i, k = token.split("_")
        return f"L_{{{i},{k}}} = A_{{{i},{k}}} / L_{{{k},{k}}}"
    if rule.key == "RK_UPDATE":
        i, j, k = token.split("_")
        return f"A_{{{i},{j}}} = A_{{{i},{j}}} - L_{{{i},{k}}} * L_{{{j},{k}}}"
    next_layer = layer_idx + 1 if layer_idx >= 0 else layer_idx
    return rule.formula_template.format(layer=layer_idx, next_layer=next_layer, m=m, u=u)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detailed cycle table from simulator trace.")
    parser.add_argument("--trace", required=True, help="Trace CSV path (name,unit,start_cycle,end_cycle)")
    parser.add_argument("--output", required=True, help="Output detailed cycle CSV path")
    parser.add_argument("--mode", choices=["auto", "du", "duo", "chol_nb"], default="auto", help="Trace profile type")
    parser.add_argument("--core-prefix", default="", help="Optional core filter, e.g. Core0_")
    parser.add_argument("--matrix-m", type=int, default=64, help="M dimension for formula/dimension text")
    parser.add_argument("--matrix-u", type=int, default=8, help="U dimension for formula/dimension text")
    parser.add_argument("--reducer", choices=["median", "max", "mean", "sum"], default="median")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.output)

    events = read_events(trace_path, core_prefix=args.core_prefix)
    if not events:
        raise SystemExit(f"No events found in trace: {trace_path}")

    mode = detect_mode(events) if args.mode == "auto" else args.mode
    rules = build_rules(mode)

    grouped: dict[tuple[str, int, str], list[TraceEvent]] = {}
    for event in events:
        for rule in rules:
            match = rule.pattern.match(event.name)
            if not match:
                continue
            layer_idx, token = parse_group_token(rule.key, match)
            grouped.setdefault((rule.key, layer_idx, token), []).append(event)
            break

    if not grouped:
        raise SystemExit(
            f"No matched compute events for mode={mode}. "
            f"Try --mode chol_nb/du/duo explicitly, or verify trace event names."
        )

    rows: list[dict[str, str]] = []
    sorted_keys = sorted(grouped.keys(), key=sort_key)
    step_idx = 0

    for key, layer_idx, token in sorted_keys:
        rule = next(rule for rule in rules if rule.key == key)
        matched_events = grouped[(key, layer_idx, token)]
        durations = [event.duration for event in matched_events]
        cycles = _reduce(durations, args.reducer)
        formula = build_formula(rule, layer_idx, token, args.matrix_m, args.matrix_u)
        dims = rule.dimension_template.format(m=args.matrix_m, u=args.matrix_u)
        name_set = sorted({event.name for event in matched_events})
        event_key = key if token in {"", str(layer_idx)} else f"{key}_{token}"

        rows.append(
            {
                "step_idx": str(step_idx),
                "layer_idx": str(layer_idx),
                "trace_mode": mode,
                "event_key": event_key,
                "onnx_op": rule.onnx_op,
                "compute_op": rule.compute_op,
                "formula": formula,
                "formula_dims": dims,
                "compute_cycles": f"{cycles:.2f}",
                "sample_count": str(len(matched_events)),
                "reducer": args.reducer,
                "matched_events": "|".join(name_set),
            }
        )
        step_idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "layer_idx",
                "trace_mode",
                "event_key",
                "onnx_op",
                "compute_op",
                "formula",
                "formula_dims",
                "compute_cycles",
                "sample_count",
                "reducer",
                "matched_events",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"mode={mode}")
    print(f"rows={len(rows)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
