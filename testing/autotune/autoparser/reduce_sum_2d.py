
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
    LowDimsAxesParser,
)

# ── 1. Get the raw (undecorated) function AST ─────────────────────────────────
# elementwise_add is wrapped by @autotune and @jit, so we define the raw fn
# separately, or unwrap it. Simplest is to define the generator standalone:

def compute_reduce_sum(M, N, block_M):
    @T.prim_func
    def reduce_sum_2D(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, 1), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M),
            is_npu=True,
        ) as (cid, _):
            A_shared = T.alloc_shared((block_M, N), "float16")
            B_local = T.alloc_fragment((block_M, 1), "float16")
            offset = cid * block_M

            T.copy(A[offset, 0], A_shared, size=[block_M, N])
            T.reduce(
                A_shared, B_local, dims=1, reduce_mode="sum", clear=True
            )
            T.copy(B_local, B[offset, 0], size=[block_M, 1])

    return reduce_sum_2D

source   = textwrap.dedent(inspect.getsource(compute_reduce_sum))
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


reduction_parser = ReductionAxesParser(func_ast, keys)
reduction_axes   = reduction_parser.parse()
print("Reduction axes:", reduction_axes)


low_dims_parser = LowDimsAxesParser(func_ast, keys)  
low_dims_axes   = low_dims_parser.parse()            
print("Low dims axes: ", low_dims_axes) 

buf_parser       = BufferNumsParser(func_ast, keys, miss_params)
buf_nums, buf_params = buf_parser.parse()
print("Buffer count:  ", buf_nums)   # Expected: 3  (A, B, C)
print("Buffer params: ", buf_params) # Expected: ['A', 'B', 'C']