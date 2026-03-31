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


def reduce_cycles(values: list[int], reducer: str) -> float:
    if not values:
        return 0.0
    if reducer == "median":
        return float(statistics.median(values))
    if reducer == "mean":
        return float(statistics.mean(values))
    if reducer == "max":
        return float(max(values))
    if reducer == "sum":
        return float(sum(values))
    raise ValueError(f"Unsupported reducer: {reducer}")


def normalize_compute_op_by_units(compute_op: str, units: set[str]) -> str:
    if not units:
        return compute_op
    all_scalar = all("Scalar" in unit for unit in units)
    all_vector = all("Vector" in unit for unit in units)
    if compute_op.startswith("VECTOR_") and all_scalar:
        return "SCALAR_" + compute_op[len("VECTOR_"):]
    if compute_op.startswith("SCALAR_") and all_vector:
        return "VECTOR_" + compute_op[len("SCALAR_"):]
    return compute_op


def infer_mode(events: list[TraceEvent], matrix_u: int) -> str:
    names = [event.name for event in events]
    if any(name.startswith("CHOL_NB_") for name in names):
        return "chol_nb"
    if any(name.startswith("CHOL_") for name in names):
        return "chol_block"
    if any(name.startswith("DUO_") for name in names):
        return "deepunfold_duo"
    if any(name.startswith("DU_") for name in names):
        return "deepunfold_du"
    if any(name.startswith("LDL_") for name in names):
        max_j = -1
        pat = re.compile(r"^LDL_D_UPDATE_(\d+)$")
        for name in names:
            match = pat.match(name)
            if match:
                max_j = max(max_j, int(match.group(1)))
        if max_j >= 0 and (max_j + 1) >= matrix_u:
            return "ldl_noblock"
        return "ldl_block"
    return "unknown"


def detect_block_size(events: list[TraceEvent], mode: str, matrix_u: int) -> int:
    if mode == "chol_nb" or mode == "ldl_noblock":
        return 1

    if mode == "chol_block":
        max_j = -1
        pat = re.compile(r"^CHOL_POTRF_DIAG_SQRT_(\d+)$")
        for event in events:
            match = pat.match(event.name)
            if match:
                max_j = max(max_j, int(match.group(1)))
        if max_j >= 0:
            n_blocks = max_j + 1
            if n_blocks > 0 and matrix_u % n_blocks == 0:
                return max(1, matrix_u // n_blocks)

    if mode.startswith("ldl"):
        max_j = -1
        pat = re.compile(r"^LDL_D_UPDATE_(\d+)$")
        for event in events:
            match = pat.match(event.name)
            if match:
                max_j = max(max_j, int(match.group(1)))
        if max_j >= 0:
            n_blocks = max_j + 1
            if n_blocks > 0 and matrix_u % n_blocks == 0:
                return max(1, matrix_u // n_blocks)

    return 1


def map_deepunfold(name: str, m: int, u: int) -> tuple[str, int, str, str, str, str] | None:
    rules = [
        (r"^DU_GRAM$|^DUO_GRAM$", "GRAM", -1, "MatMul", "MATMUL", "G = H^H @ H", f"{u}*{m}*{m}*{u}"),
        (r"^DU_REG$|^DUO_REG$", "REG", -1, "Add", "VECTOR_ADD", "A = G + RegI", f"{u}*{u}"),
        (r"^DU_AX_(\d+)$|^DUO_AX_(\d+)$", "AX", 0, "MatMul", "MATMUL", "AX_{l} = A @ X_{l}", f"{u}*{u}*{u}*{u}"),
        (r"^DU_RES_(\d+)_\d+$", "RES", 0, "Add", "VECTOR_ADD", "R_{l} = RegI + AX_{l}", f"{u}*{u}"),
        (r"^DU_XNEXT_(\d+)$", "XNEXT", 0, "MatMul", "MATMUL", "Xtmp_{lp1} = X_{l} @ R_{l}", f"{u}*{u}*{u}*{u}"),
        (r"^DUO_XNEXT_(\d+)$", "XNEXT", 0, "MatMul", "MATMUL", "Xtmp_{lp1} = X_{l} @ A", f"{u}*{u}*{u}*{u}"),
        (r"^DU_STORE_XK_(\d+)$|^DUO_STORE_XK_(\d+)$", "STORE_XK", 0, "Add", "VECTOR_ADD", "X_{lp1} = Xtmp_{lp1} + RegI", f"{u}*{u}"),
        (r"^DUO_VEC_CORR_(\d+)$", "VEC_CORR", 0, "Add", "VECTOR_ADD", "Zcorr_{l} = Z_{l} + RegI", f"{u}*{u}"),
        (r"^DUO_VEC_MERGE_(\d+)$", "VEC_MERGE", 0, "Add", "VECTOR_ADD", "X_{lp1} = Zcorr_{l} + RegI", f"{u}*{u}"),
        (r"^DU_W$|^DUO_W$", "W", -1, "MatMul", "MATMUL", "W = X_last @ H^H", f"{u}*{u}*{u}*{m}"),
        (r"^DU_XHAT$|^DUO_XHAT$", "XHAT", -1, "MatMul", "MATMUL", "X_hat = W @ Y", f"{u}*{m}*{m}*{u}"),
    ]
    for pattern, key, layer_default, onnx, comp, formula_t, dims in rules:
        match = re.match(pattern, name)
        if not match:
            continue
        group_vals = [value for value in match.groups() if value is not None]
        layer = layer_default
        formula = formula_t
        if group_vals:
            layer = int(group_vals[0])
            formula = formula.replace("{l}", str(layer)).replace("{lp1}", str(layer + 1))
            key = f"{key}_{layer}"
        return key, layer, onnx, comp, formula, dims
    return None


def map_chol(name: str, m: int, u: int, blk: int) -> tuple[str, int, str, str, str, str] | None:
    if name in {"CHOL_GRAM", "CHOL_NB_GRAM"}:
        return "GRAM", -1, "MatMul", "MATMUL", "G = H^H @ H", f"{u}*{m}*{m}*{u}"
    if name in {"CHOL_REG", "CHOL_NB_REG"}:
        return "REG", -1, "Add", "VECTOR_ADD", "A = G + RegI", f"{u}*{u}"

    match = re.match(r"^CHOL(_NB)?_POTRF_DIAG_UPD_(\d+)_(\d+)$", name)
    if match:
        j, k = int(match.group(2)), int(match.group(3))
        return f"POTRF_DIAG_UPD_{j}_{k}", j, "SubMul", "VECTOR_MAC", f"A_{{{j},{j}}} = A_{{{j},{j}}} - L_{{{j},{k}}}^2", f"{blk}"

    match = re.match(r"^CHOL(_NB)?_POTRF_DIAG_SQRT_(\d+)$", name)
    if match:
        j = int(match.group(2))
        return f"POTRF_DIAG_SQRT_{j}", j, "Sqrt", "SCALAR_SQRT", f"L_diag_{j} = sqrt(A_diag_{j})", f"{blk}*{blk}"

    match = re.match(r"^CHOL(_NB)?_TRSM_NUM_UPD_(\d+)_(\d+)_(\d+)$", name)
    if match:
        i, j, k = int(match.group(2)), int(match.group(3)), int(match.group(4))
        return f"TRSM_NUM_UPD_{i}_{j}_{k}", i, "SubMul", "VECTOR_MAC", f"A_{{{i},{j}}} = A_{{{i},{j}}} - L_{{{i},{k}}}L_{{{j},{k}}}", f"{blk}"

    match = re.match(r"^CHOL(_NB)?_TRSM_DIV_(\d+)_(\d+)$", name)
    if match:
        i, j = int(match.group(2)), int(match.group(3))
        return f"TRSM_DIV_{i}_{j}", i, "Div", "SCALAR_DIV", f"L_{{{i},{j}}} = A_{{{i},{j}}}/L_{{{j},{j}}}", f"{blk}*{blk}"

    match = re.match(r"^CHOL_NB_TRSM_DIAG_INV_(\d+)$", name)
    if match:
        j = int(match.group(1))
        return f"TRSM_DIAG_INV_{j}", j, "Div", "SCALAR_DIV", f"invL_{{{j}}} = 1/L_{{{j},{j}}}", "1"

    match = re.match(r"^CHOL_NB_TRSM_MUL_(\d+)_(\d+)$", name)
    if match:
        i, j = int(match.group(1)), int(match.group(2))
        return f"TRSM_MUL_{i}_{j}", i, "Mul", "VECTOR_MUL", f"L_{{{i},{j}}} = A_{{{i},{j}}} * invL_{{{j}}}", "1"

    match = re.match(r"^CHOL(_NB)?_RK_UPDATE_(\d+)_(\d+)_(\d+)$", name)
    if match:
        i, k, j = int(match.group(2)), int(match.group(3)), int(match.group(4))
        return f"RK_UPDATE_{i}_{k}_{j}", i, "SubMul", "VECTOR_MAC", f"A_{{{i},{k}}} = A_{{{i},{k}}} - L_{{{i},{j}}}L_{{{k},{j}}}", f"{blk}"

    match = re.match(r"^CHOL_FWD_DIAG_INV_(\d+)$", name)
    if match:
        c = int(match.group(1))
        return f"FWD_DIAG_INV_{c}", c, "Div", "SCALAR_DIV", f"Y_{{{c}}} = 1/L_{{{c},{c}}}", f"{blk}*{blk}"

    match = re.match(r"^CHOL_FWD_OFF_MAC_(\d+)_(\d+)$", name)
    if match:
        i, c = int(match.group(1)), int(match.group(2))
        return f"FWD_OFF_MAC_{i}_{c}", i, "SubMul", "VECTOR_MAC", f"T_{{{i},{c}}} = Y_{{{i},:}} - L_{{{i},:}}Y_:", f"{blk}"

    match = re.match(r"^CHOL_FWD_OFF_UPD_(\d+)_(\d+)$", name)
    if match:
        i, c = int(match.group(1)), int(match.group(2))
        return f"FWD_OFF_UPD_{i}_{c}", i, "Div", "SCALAR_DIV", f"Y_{{{i},{c}}} = T_{{{i},{c}}}/L_{{{i},{i}}}", f"{blk}*{blk}"

    if name == "CHOL_BWD_MAC_FULL":
        return "BWD_MAC_FULL", -1, "MatMul", "MATMUL", "A_inv = Y^T @ Y", f"{u}*{u}*{u}*{u}"

    return None


def map_ldl(name: str, m: int, u: int, blk: int) -> tuple[str, int, str, str, str, str] | None:
    if name == "LDL_GRAM":
        return "GRAM", -1, "MatMul", "MATMUL", "G = H^H @ H", f"{u}*{m}*{m}*{u}"
    if name == "LDL_REG":
        return "REG", -1, "Add", "VECTOR_ADD", "A = G + RegI", f"{u}*{u}"

    match = re.match(r"^LDL_D_UPDATE_(\d+)$", name)
    if match:
        j = int(match.group(1))
        return f"D_UPDATE_{j}", j, "SubMul", "VECTOR_MAC", f"D_{{{j}}} = A_{{{j},{j}}} - L D L^T", f"{blk}*{u}*{blk}"

    match = re.match(r"^LDL_D_DIAG_INV_(\d+)$", name)
    if match:
        j = int(match.group(1))
        return f"D_DIAG_INV_{j}", j, "Div", "SCALAR_DIV", f"Dinv_{{{j}}} = 1 / D_{{{j}}}", f"{blk}*{blk}"

    match = re.match(r"^LDL_D_INV_MUL_(\d+)$", name)
    if match:
        j = int(match.group(1))
        return f"D_INV_MUL_{j}", j, "Mul", "VECTOR_MUL", f"DinvBlk_{{{j}}} = D_{{{j}}} * Dinv_{{{j}}}", f"{blk}*{blk}"

    match = re.match(r"^LDL_L_UPDATE_(\d+)_(\d+)_PACK(\d+)$", name)
    if match:
        i, j, p = int(match.group(1)), int(match.group(2)), int(match.group(3))
        pd = blk * p
        return f"L_UPDATE_{i}_{j}_PACK{p}", i, "MatMul", "MATMUL", f"L_{{{i},{j}}} = A_{{{i},{j}}} @ Dinv_{{{j}}}", f"{pd}*{pd}*{pd}*{pd}"

    match = re.match(r"^LDL_BWD_DIAG_MUL_(\d+)_(\d+)$", name)
    if match:
        j, rep = int(match.group(1)), int(match.group(2))
        return f"BWD_DIAG_MUL_{j}_{rep}", j, "MatMul", "MATMUL", f"T_{{{j},{j}}}^{rep} = L @ X", f"{blk}*(U-{j+1}*{blk})*{blk}"

    match = re.match(r"^LDL_BWD_DIAG_ACC_(\d+)_(\d+)$", name)
    if match:
        j, rep = int(match.group(1)), int(match.group(2))
        return f"BWD_DIAG_ACC_{j}_{rep}", j, "Add", "VECTOR_ADD", f"X_{{{j},{j}}}^{rep} = Dinv + T_{{{j},{j}}}^{rep}", f"{blk}*{blk}"

    match = re.match(r"^LDL_BWD_OFF_MUL_(\d+)_(\d+)_(\d+)$", name)
    if match:
        i, j, rep = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"BWD_OFF_MUL_{i}_{j}_{rep}", i, "MatMul", "MATMUL", f"T_{{{i},{j}}}^{rep} = L @ X", f"{blk}*(U-{i+1}*{blk})*{blk}"

    match = re.match(r"^LDL_BWD_OFF_ACC_(\d+)_(\d+)_(\d+)$", name)
    if match:
        i, j, rep = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"BWD_OFF_ACC_{i}_{j}_{rep}", i, "Add", "VECTOR_ADD", f"X_{{{i},{j}}}^{rep} = X_{{{i},{j}}} + T_{{{i},{j}}}^{rep}", f"{blk}*{blk}"

    return None


def numeric_token(event_key: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", event_key)
    if not parts:
        return tuple()
    return tuple(int(part) for part in parts)


def major_step_key(event_key: str, mode: str) -> str:
    if mode.startswith("chol"):
        match = re.match(r"^POTRF_DIAG_UPD_(\d+)_\d+$", event_key)
        if match:
            j = int(match.group(1))
            return f"POTRF_DIAG_A_{j}_{j}"

        match = re.match(r"^RK_UPDATE_(\d+)_(\d+)_\d+$", event_key)
        if match:
            i, k = int(match.group(1)), int(match.group(2))
            return f"RK_UPDATE_A_{i}_{k}"

        match = re.match(r"^TRSM_NUM_UPD_(\d+)_(\d+)_\d+$", event_key)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            return f"TRSM_{i}_{j}"

        match = re.match(r"^(TRSM_DIV|TRSM_MUL)_(\d+)_(\d+)$", event_key)
        if match:
            i, j = int(match.group(2)), int(match.group(3))
            return f"TRSM_{i}_{j}"

        match = re.match(r"^TRSM_DIAG_INV_(\d+)$", event_key)
        if match:
            j = int(match.group(1))
            return f"TRSM_DIAG_{j}"

        match = re.match(r"^POTRF_DIAG_SQRT_(\d+)$", event_key)
        if match:
            j = int(match.group(1))
            return f"POTRF_DIAG_{j}"

        match = re.match(r"^FWD_(DIAG_INV|OFF_MAC|OFF_UPD)_(\d+)(?:_(\d+))?$", event_key)
        if match:
            if match.group(1) == "DIAG_INV":
                c = int(match.group(2))
                return f"FWD_COL_{c}"
            i = int(match.group(2))
            c = int(match.group(3))
            return f"FWD_COL_{c}_ROW_{i}"

    if mode.startswith("ldl"):
        match = re.match(r"^(D_UPDATE|D_DIAG_INV|D_INV_MUL)_(\d+)$", event_key)
        if match:
            j = int(match.group(2))
            return f"D_BLOCK_{j}"

        match = re.match(r"^L_UPDATE_(\d+)_(\d+)_PACK\d+$", event_key)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            return f"L_UPDATE_{i}_{j}"

        match = re.match(r"^BWD_DIAG_(MUL|ACC)_(\d+)_\d+$", event_key)
        if match:
            j = int(match.group(2))
            return f"BWD_DIAG_{j}"

        match = re.match(r"^BWD_OFF_(MUL|ACC)_(\d+)_(\d+)_\d+$", event_key)
        if match:
            i, j = int(match.group(2)), int(match.group(3))
            return f"BWD_OFF_{i}_{j}"

    if mode.startswith("deepunfold"):
        match = re.match(r"^(AX|RES|XNEXT|STORE_XK|VEC_CORR|VEC_MERGE)_(\d+)$", event_key)
        if match:
            return f"{match.group(1)}_LAYER_{match.group(2)}"

    return event_key


def export_table(
    trace_path: Path,
    output_path: Path,
    summary_output_path: Path,
    mode: str,
    matrix_m: int,
    matrix_u: int,
    reducer: str,
    core_prefix: str,
) -> tuple[str, int, int]:
    events: list[TraceEvent] = []
    with trace_path.open(newline="") as f:
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

    if not events:
        raise SystemExit(f"No events found in {trace_path}")

    resolved_mode = infer_mode(events, matrix_u) if mode == "auto" else mode
    blk = detect_block_size(events, resolved_mode, matrix_u)

    grouped: dict[tuple[str, int, str, str, str], list[TraceEvent]] = {}

    for event in events:
        if event.name in {"Load", "Store", "CubeWait"}:
            continue

        mapped = None
        if resolved_mode.startswith("deepunfold"):
            mapped = map_deepunfold(event.name, matrix_m, matrix_u)
        elif resolved_mode.startswith("chol"):
            mapped = map_chol(event.name, matrix_m, matrix_u, blk)
        elif resolved_mode.startswith("ldl"):
            mapped = map_ldl(event.name, matrix_m, matrix_u, blk)

        if not mapped:
            continue

        event_key, layer_idx, onnx_op, compute_op, formula, dims = mapped
        grouped.setdefault((event_key, layer_idx, onnx_op, compute_op, formula + "|" + dims), []).append(event)

    if not grouped:
        raise SystemExit(f"No mapped compute events for mode={resolved_mode} from trace {trace_path}")

    rows: list[dict[str, str]] = []
    sorted_items = sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][0], numeric_token(item[0][0])),
    )

    major_buckets: dict[tuple[str, int], list[float]] = {}
    major_substeps: dict[tuple[str, int], set[str]] = {}
    major_ops: dict[tuple[str, int], set[str]] = {}

    for idx, (key_tuple, matched_events) in enumerate(sorted_items):
        event_key, layer_idx, onnx_op, compute_op, formula_dims = key_tuple
        formula, dims = formula_dims.split("|", 1)
        matched_units = sorted({event.unit for event in matched_events})
        normalized_compute_op = normalize_compute_op_by_units(compute_op, set(matched_units))
        durations = [event.duration for event in matched_events]
        step_cycle = reduce_cycles(durations, reducer)
        major_key = major_step_key(event_key, resolved_mode)

        major_buckets.setdefault((major_key, layer_idx), []).append(step_cycle)
        major_substeps.setdefault((major_key, layer_idx), set()).add(event_key)
        major_ops.setdefault((major_key, layer_idx), set()).add(normalized_compute_op)

        rows.append(
            {
                "step_idx": str(idx),
                "layer_idx": str(layer_idx),
                "operator_mode": resolved_mode,
                "major_step": major_key,
                "event_key": event_key,
                "onnx_op": onnx_op,
                "compute_op": normalized_compute_op,
                "formula": formula,
                "formula_dims": dims,
                "compute_cycles": f"{step_cycle:.2f}",
                "matched_events": "|".join(sorted({event.name for event in matched_events})),
                "matched_units": "|".join(matched_units),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_idx",
                "layer_idx",
                "operator_mode",
                "major_step",
                "event_key",
                "onnx_op",
                "compute_op",
                "formula",
                "formula_dims",
                "compute_cycles",
                "matched_events",
                "matched_units",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, str]] = []
    for idx, ((major_key, layer_idx), values) in enumerate(
        sorted(major_buckets.items(), key=lambda item: (item[0][1], item[0][0], numeric_token(item[0][0])))
    ):
        summary_rows.append(
            {
                "major_idx": str(idx),
                "layer_idx": str(layer_idx),
                "operator_mode": resolved_mode,
                "major_step": major_key,
                "compute_ops": "|".join(sorted(major_ops[(major_key, layer_idx)])),
                "substep_count": str(len(major_substeps[(major_key, layer_idx)])),
                "major_cycle_sum": f"{sum(values):.2f}",
                "major_cycle_mean": f"{statistics.mean(values):.2f}",
                "major_cycle_max": f"{max(values):.2f}",
                "major_cycle_min": f"{min(values):.2f}",
                "substeps": "|".join(sorted(major_substeps[(major_key, layer_idx)])),
            }
        )

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "major_idx",
                "layer_idx",
                "operator_mode",
                "major_step",
                "compute_ops",
                "substep_count",
                "major_cycle_sum",
                "major_cycle_mean",
                "major_cycle_max",
                "major_cycle_min",
                "substeps",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return resolved_mode, len(rows), len(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detailed operator cycle table from trace CSV.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        default="auto",
        choices=[
            "auto",
            "deepunfold_du",
            "deepunfold_duo",
            "chol_block",
            "chol_nb",
            "ldl_block",
            "ldl_noblock",
        ],
    )
    parser.add_argument("--matrix-m", type=int, default=64)
    parser.add_argument("--matrix-u", type=int, default=8)
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
