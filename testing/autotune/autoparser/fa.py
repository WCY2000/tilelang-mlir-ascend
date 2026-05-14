
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

seq_len, dim = 512, 128
# ── 1. Get the raw (undecorated) function AST ─────────────────────────────────
# elementwise_add is wrapped by @autotune and @jit, so we define the raw fn
# separately, or unwrap it. Simplest is to define the generator standalone:

def online_flash_attention(block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]
    shape_work = [seq_len, seq_len]
    block_m = block_M
    block_n = block_N
    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
            offset = cid * block_m
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])

            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            scores_cast = T.alloc_fragment([block_m, block_n], dtype)
            correction = T.alloc_fragment([block_m,1], accum_dtype)
            local_max = T.alloc_fragment([block_m,1], accum_dtype)
            local_sum = T.alloc_fragment([block_m,1], accum_dtype)
            acc_m = T.alloc_fragment([block_m, 1], accum_dtype)
            acc_l = T.alloc_fragment([block_m, 1], accum_dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            tmp = T.alloc_fragment([block_m, block_n], accum_dtype)
            tmp1 = T.alloc_fragment([block_m,1], accum_dtype)
            new_max = T.alloc_fragment([block_m,1], accum_dtype)
            scales = T.alloc_fragment([block_m, block_n], accum_dtype)

            value_zero = 0
            scale = (1.0 / dim)**0.5
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(scale, scales)

            for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):

                # cube
                T.copy(K[k * block_n, 0], K_shared, size=[block_n, dim])
                T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                # vec
                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max ,tmp1)
                T.vexp(tmp1, correction)
                #scores for current loop
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                T.vmul(acc_o, correction, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                #copy new_max to acc_m
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)

                # cube
                T.copy(V[k * block_n, 0], V_shared, size=[block_n, dim])
                T.gemm(scores_cast, V_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_m, dim], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            real_m = T.min(block_m, seq_len - cid * block_m)
            T.copy(O_cast, Output[cid * block_m, 0], size=[real_m, dim])

    return flash_attention

source   = textwrap.dedent(inspect.getsource(online_flash_attention))
func_ast = ast.parse(source)

# ── 2. Define keys and candidates ─────────────────────────────────────────────
# keys:             axis_name → problem-size variable name in the generator sig
# candidates_params: tunable params (not supplied by the user at call-site)
# miss_params:       same list, used by BufferNumsParser for warnings

keys = {"M": "seq_len", "N": "seq_len", "D": "dim"}
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