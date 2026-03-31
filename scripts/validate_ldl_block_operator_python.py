#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def cube_cycles(tile_m: int, tile_n: int, tile_k: int,
                cube_m: int = 16, cube_n: int = 16, cube_k: int = 16,
                base_latency: int = 1) -> int:
    blocks_m = ceil_div(max(1, tile_m), max(1, cube_m))
    blocks_n = ceil_div(max(1, tile_n), max(1, cube_n))
    blocks_k = ceil_div(max(1, tile_k), max(1, cube_k))
    cube_steps = blocks_m * blocks_n * blocks_k
    pipeline_fill_drain = cube_m + cube_n - 2
    return base_latency + pipeline_fill_drain + max(1, cube_steps)


def pick_mul_opcode(tile_m: int, tile_k: int, tile_n: int) -> str:
    if tile_m <= 2 and tile_k <= 2 and tile_n <= 2:
        return "MAC"
    return "GEMM_PRELOAD"


def pick_ldl_step_mul_opcode(blk: int, tile_m: int, tile_k: int, tile_n: int) -> str:
    if blk == 1:
        return "MAC"
    return pick_mul_opcode(tile_m, tile_k, tile_n)


def pick_ldl_micro_mul_opcode(mode: str, blk: int, tile_m: int, tile_k: int, tile_n: int) -> str:
    if mode == "opt2" and blk <= 2 and tile_m <= 2 and tile_n <= 2:
        return "MAC"
    return pick_ldl_step_mul_opcode(blk, tile_m, tile_k, tile_n)


def to_unit(opcode: str) -> str:
    if opcode == "GEMM_PRELOAD":
        return "Cube"
    if opcode in {"MOVIN"}:
        return "MTE2"
    if opcode in {"MOVOUT"}:
        return "MTE3"
    return "Vector"


def generate_ldl_ops_per_batch(mode: str, m: int, u: int, blk: int, bwd_steps: int,
                               cube_dim_target: int) -> list[tuple[str, str, int, int, int]]:
    n_blocks = max(1, u // blk)
    auto_pack_blocks = max(1, cube_dim_target // blk) if blk == 2 else 1
    cube_pack_blocks = auto_pack_blocks

    ops = []

    ops.append(("LDL_BARRIER_LOAD2GRAM", "PIPE_BARRIER", 0, 0, 0))
    ops.append(("LDL_GRAM", "GEMM_PRELOAD", u, m, u))
    ops.append(("LDL_BARRIER_GRAM2REG", "PIPE_BARRIER", 0, 0, 0))
    ops.append(("LDL_REG", "ADD", u, u, u))
    ops.append(("LDL_BARRIER_REG2BLDL", "PIPE_BARRIER", 0, 0, 0))

    for j in range(n_blocks):
        if mode == "old":
            d_update_k_len = max(1, j) if blk == 1 else u
        else:
            d_update_k_len = max(blk, j * blk)
        d_update_opcode = pick_ldl_micro_mul_opcode(mode, blk, blk, d_update_k_len, blk)
        ops.append(("LDL_D_UPDATE", d_update_opcode, blk, d_update_k_len, blk))

        ops.append(("LDL_D_DIAG_INV", "DIV", blk, blk, blk))
        ops.append(("LDL_D_INV_MUL", "MUL", blk, blk, blk))

        i = j + 1
        while i < n_blocks:
            packed_blocks = min(cube_pack_blocks, n_blocks - i)
            packed_dim = blk * packed_blocks
            l_upd_opcode = pick_ldl_step_mul_opcode(blk, packed_dim, packed_dim, packed_dim)
            ops.append(("LDL_L_UPDATE", l_upd_opcode, packed_dim, packed_dim, packed_dim))
            i += cube_pack_blocks

        ops.append(("LDL_BARRIER_BLDL_STEP", "PIPE_BARRIER", 0, 0, 0))

    for col in range(n_blocks - 1, -1, -1):
        j = col
        diag_k_blocks = (n_blocks - (j + 1)) if n_blocks > (j + 1) else 0
        if diag_k_blocks > 0:
            diag_k_len = diag_k_blocks * blk
            diag_mul_opcode = pick_ldl_micro_mul_opcode(mode, blk, blk, diag_k_len, blk)
            for _ in range(bwd_steps):
                ops.append(("LDL_BWD_DIAG_MUL", diag_mul_opcode, blk, diag_k_len, blk))
                ops.append(("LDL_BWD_DIAG_ACC", "ADD", blk, blk, blk))

        ops.append(("LDL_BARRIER_BWD_DIAG2OFF", "PIPE_BARRIER", 0, 0, 0))

        for i in range(col - 1, -1, -1):
            off_k_blocks = (n_blocks - (i + 1)) if n_blocks > (i + 1) else 0
            if off_k_blocks == 0:
                continue
            off_k_len = off_k_blocks * blk
            off_mul_opcode = pick_ldl_micro_mul_opcode(mode, blk, blk, off_k_len, blk)
            for _ in range(bwd_steps):
                ops.append(("LDL_BWD_OFF_MUL", off_mul_opcode, blk, off_k_len, blk))
                ops.append(("LDL_BWD_OFF_ACC", "ADD", blk, blk, blk))

        ops.append(("LDL_BARRIER_BWD_COL", "PIPE_BARRIER", 0, 0, 0))

    ops.append(("LDL_BARRIER_BWD2STORE", "PIPE_BARRIER", 0, 0, 0))
    return ops


def expected_stats(mode: str, batch_size: int, m: int, u: int, blk: int, bwd_steps: int,
                   cube_dim_target: int) -> dict:
    ops = generate_ldl_ops_per_batch(mode, m, u, blk, bwd_steps, cube_dim_target)
    op_cnt = Counter()
    unit_cnt = Counter()
    for op_name, opcode, _, _, _ in ops:
        op_cnt[op_name] += 1
        unit_cnt[to_unit(opcode)] += 1

    key_op_unit = defaultdict(Counter)
    for op_name, opcode, _, _, _ in ops:
        if op_name in {"LDL_D_UPDATE", "LDL_L_UPDATE", "LDL_BWD_DIAG_MUL", "LDL_BWD_OFF_MUL"}:
            key_op_unit[op_name][to_unit(opcode)] += 1

    return {
        "per_batch_op_cnt": op_cnt,
        "per_batch_unit_cnt": unit_cnt,
        "per_batch_key_op_unit": key_op_unit,
        "total_op_cnt": Counter({k: v * batch_size for k, v in op_cnt.items()}),
        "total_unit_cnt": Counter({k: v * batch_size for k, v in unit_cnt.items()}),
        "total_key_op_unit": {k: Counter({u0: c * batch_size for u0, c in v.items()})
                              for k, v in key_op_unit.items()},
    }


def trace_stats(path: str) -> dict:
    unit_cnt = Counter()
    key_op_unit = defaultdict(Counter)
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["name"].strip('"')
            unit = row["unit"].strip('"')
            unit_short = unit.split("_", 1)[1] if "_" in unit else unit
            unit_cnt[unit_short] += 1

            if name.startswith("LDL_D_UPDATE"):
                key_op_unit["LDL_D_UPDATE"][unit_short] += 1
            elif name.startswith("LDL_L_UPDATE"):
                key_op_unit["LDL_L_UPDATE"][unit_short] += 1
            elif name.startswith("LDL_BWD_DIAG_MUL"):
                key_op_unit["LDL_BWD_DIAG_MUL"][unit_short] += 1
            elif name.startswith("LDL_BWD_OFF_MUL"):
                key_op_unit["LDL_BWD_OFF_MUL"][unit_short] += 1

    return {
        "unit_cnt": unit_cnt,
        "key_op_unit": key_op_unit,
    }


def print_counter(title: str, counter_obj: Counter):
    print(title)
    for k in sorted(counter_obj.keys()):
        print(f"  {k}: {counter_obj[k]}")


def main():
    parser = argparse.ArgumentParser(description="Validate LDL block operator logic in Python (old vs opt2).")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--u", type=int, default=16)
    parser.add_argument("--blk", type=int, default=2)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--bwd-steps", type=int, default=1)
    parser.add_argument("--cube-m", type=int, default=16)
    parser.add_argument("--cube-n", type=int, default=16)
    parser.add_argument("--cube-k", type=int, default=16)
    parser.add_argument("--cube-base", type=int, default=1)
    parser.add_argument("--old-trace", default="results/LDL/falsification/ldl_block_64x16_trace.csv")
    parser.add_argument("--opt2-trace", default="results/LDL/falsification/ldl_block_64x16_trace_opt2.csv")
    args = parser.parse_args()

    cube_dim_target = min(args.cube_m, args.cube_n, args.cube_k)

    print("=== Cube 公式（来自 SystolicWS::get_inst_compute_cycles）===")
    print("C_cube = base_latency + (cube_m + cube_n - 2) + max(1, Bm*Bn*Bk)")
    print("Bm=ceil(tile_m/cube_m), Bn=ceil(tile_n/cube_n), Bk=ceil(tile_k/cube_k)")
    example = cube_cycles(16, 16, 64, args.cube_m, args.cube_n, args.cube_k, args.cube_base)
    print(f"示例: tile=(16,16,64), cube=(16,16,16), base=1 => C_cube={example}")
    print()

    print("=== 本次 LDL 优化对应公式 ===")
    print("old: d_update_k_len = U (blk>1)")
    print("opt2: d_update_k_len = max(blk, j*blk)")
    print("old opcode(route): pick_ldl_step_mul_opcode")
    print("opt2 opcode(route): if blk<=2 and tile_m<=2 and tile_n<=2 => MAC else old route")
    print()

    exp_old = expected_stats("old", args.batch, args.m, args.u, args.blk, args.bwd_steps, cube_dim_target)
    exp_opt2 = expected_stats("opt2", args.batch, args.m, args.u, args.blk, args.bwd_steps, cube_dim_target)
    obs_old = trace_stats(args.old_trace)
    obs_opt2 = trace_stats(args.opt2_trace)

    print("=== 预测 vs Trace：unit 事件数（含 Wait/MTE）===")
    print("[old] expected compute/barrier unit events (batch total):")
    print_counter("", exp_old["total_unit_cnt"])
    print("[old] observed unit events from trace:")
    print_counter("", obs_old["unit_cnt"])
    print()
    print("[opt2] expected compute/barrier unit events (batch total):")
    print_counter("", exp_opt2["total_unit_cnt"])
    print("[opt2] observed unit events from trace:")
    print_counter("", obs_opt2["unit_cnt"])
    print()

    print("=== 关键操作 unit 路由（预测 vs Trace）===")
    key_ops = ["LDL_D_UPDATE", "LDL_L_UPDATE", "LDL_BWD_DIAG_MUL", "LDL_BWD_OFF_MUL"]
    for mode_name, exp, obs in [("old", exp_old, obs_old), ("opt2", exp_opt2, obs_opt2)]:
        print(f"[{mode_name}]")
        for op in key_ops:
            exp_dict = dict(exp["total_key_op_unit"].get(op, {}))
            obs_dict = dict(obs["key_op_unit"].get(op, {}))
            print(f"  {op}: expected={exp_dict}, observed={obs_dict}")
        print()

    print("=== 结论提示 ===")
    print("1) 若 key op 的 expected/observed 一致，说明 Python 版 lowering 与当前 trace 对齐。")
    print("2) old->opt2 重点应表现为 D_UPDATE/BWD_*_MUL 从 Cube 迁移到 Vector。")
    print("3) Wait 事件来自调度/队列时序，Python 结构模型不直接预测 Wait。")


if __name__ == "__main__":
    main()
