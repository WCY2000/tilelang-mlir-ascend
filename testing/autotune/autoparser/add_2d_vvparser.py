import os

import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

from tilelang.autotuner.dsl_analysis.vv_param_parser import (
    parse_tl_axis_info_from_fn,
    print_vv_axis_parse_result,
)


# ── 1. Get the raw function AST ───────────────────────────────────────────────
def elementwise_add_kernel(M, N, block_M, block_N):
    @T.prim_func
    def elemAdd(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N) * T.ceildiv(M, block_M),
            is_npu=True,
        ) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            B_shared = T.alloc_shared((block_M, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float16")
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.vadd(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return elemAdd


# ── 2. Parse — no keys or candidates needed ───────────────────────────────────
result = parse_tl_axis_info_from_fn(elementwise_add_kernel)

# ── 3. Results ────────────────────────────────────────────────────────────────
print_vv_axis_parse_result("VV parser output", result)
