import ast
import inspect
import textwrap
import os

import tilelang.language as T
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

from tilelang.autotuner.vv_parser import parse_tilelang_axes

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
            C_local  = T.alloc_fragment((block_M, block_N), "float16")
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.vadd(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return elemAdd

# ── 2. Parse — no keys or candidates needed ───────────────────────────────────
source   = textwrap.dedent(inspect.getsource(elementwise_add_kernel))
func_ast = ast.parse(source)

result = parse_tilelang_axes(func_ast)

# ── 3. Results ────────────────────────────────────────────────────────────────
print("Status:         ", result.status)
print("Inferred keys:  ", result.inferred_keys)   # auto-discovered axis→size map
print("Split params:   ", result.split_params)    # replaces SplitAxesParser
print("Tiling params:  ", result.tiling_params)   # replaces TilingAxesParser
print("Reduction axes: ", result.reduction_axes)  # replaces ReductionAxesParser
print("Low dim axes:   ", result.low_dim_axes)    # replaces LowDimsAxesParser
print("Buffer count:   ", result.buf_count)       # replaces BufferNumsParser
print("Buffer params:  ", result.buffer_params)
if result.diagnostics:
    print("Diagnostics:    ", result.diagnostics)
