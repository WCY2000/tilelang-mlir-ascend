
import ast
import inspect
import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

from tilelang.autotuner.kernel_classifier import analyze_kernel_type
os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import textwrap
from tilelang.autotuner.autoparser import (
    SplitAxesParser,
    TilingAxesParser,
    ReductionAxesParser,
    BufferNumsParser,
)

# ── 1. Get the raw (undecorated) function AST ─────────────────────────────────
# elementwise_add is wrapped by @autotune and @jit, so we define the raw fn
# separately, or unwrap it. Simplest is to define the generator standalone:

def elementwise_add_kernel(M, N, block_M, block_N):
    @T.prim_func
    def elemAdd(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):  
        num_physical_kernels = 48
        num_logical_kernels = (N // block_N) * (M // block_M)
        with T.Kernel(num_physical_kernels, is_npu=True) as (kernel_id, _):
            num_local_tasks = T.ceildiv(
                num_logical_kernels - kernel_id, num_physical_kernels
            )

            for task_id in T.serial(num_local_tasks):
                cid = task_id * num_physical_kernels + kernel_id
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

source   = textwrap.dedent(inspect.getsource(elementwise_add_kernel))
func_ast = ast.parse(source)

# ── 2. Define keys and candidates ─────────────────────────────────────────────
# keys:             axis_name → problem-size variable name in the generator sig
# candidates_params: tunable params (not supplied by the user at call-site)
# miss_params:       same list, used by BufferNumsParser for warnings

keys             = {"M": "M", "N": "N"}
candidates_params = ["block_M", "block_N"]
miss_params       = ["block_M", "block_N"]

# ── 3. Run each parser ────────────────────────────────────────────────────────

split_parser = SplitAxesParser(func_ast, keys, candidates_params)
split_axes   = split_parser.parse()
print("Split axes:    ", split_axes)
# Expected: {} for this kernel because T.Kernel uses a *product* form,
# not separate T.ceildiv args. See note below.

tiling_parser = TilingAxesParser(func_ast, keys, candidates_params)
tiling_axes   = tiling_parser.parse()
print("Tiling axes:   ", tiling_axes)
# Expected: {} — no T.Pipelined / range loop in this elementwise kernel

reduction_parser = ReductionAxesParser(func_ast, keys)
reduction_axes   = reduction_parser.parse()
print("Reduction axes:", reduction_axes)
# Expected: [] — no reduction loop

buf_parser       = BufferNumsParser(func_ast, keys, miss_params)
buf_nums, buf_params = buf_parser.parse()
print("Buffer count:  ", buf_nums)   # Expected: 3  (A, B, C)
print("Buffer params: ", buf_params) # Expected: ['A', 'B', 'C']