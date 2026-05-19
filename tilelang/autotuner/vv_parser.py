# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Unified TileLang axis parser entry point.

This module exposes ``parse_tilelang_axes``, a single call that runs all five
analysis passes over a TileLang kernel generator AST and returns a unified
:class:`ParseTilelangAxesResult`.  It replaces the five separate parser classes
from ``autoparser.py`` (``SplitAxesParser``, ``TilingAxesParser``,
``ReductionAxesParser``, ``LowDimsAxesParser``, ``BufferNumsParser``) with a
single, dependency-free call.

Typical usage
-------------
::

    import ast
    import inspect
    import textwrap

    from tilelang.autotuner.vv_parser import parse_tilelang_axes

    source   = textwrap.dedent(inspect.getsource(my_kernel_fn))
    func_ast = ast.parse(source)

    result = parse_tilelang_axes(func_ast, provided_args={"M": 1024, "N": 1024})

    print(result.split_params)    # {"M": "block_M", "N": "block_N"}
    print(result.tiling_params)   # {"K": "block_K"}
    print(result.reduction_axes)  # ["K"]
    print(result.buf_count)       # 3

Design
------
``parse_tilelang_axes`` delegates axis-semantic analysis to
``parse_tl_axis_semantic`` from ``vv_param_parser`` and buffer counting to
``BufferNumsParser`` from ``autoparser``.

``provided_args`` maps problem-dimension variable names to their concrete
values (e.g. ``{"M": 1024, "N": 1024}``).  When omitted, axis extents are
still resolved structurally (``inferred_keys`` is populated) but their
``state`` will be ``RUNTIME_NON_TUNABLE`` rather than ``FIXED_COMPILE_TIME``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .autoparser import BufferNumsParser
from .dsl_analysis.vv_param_parser import (
    _resolve_function_node,
    _resolve_tunable_params,
    parse_tl_axis_semantic,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseTilelangAxesResult:
    """
    Unified result of all TileLang axis analysis passes.

    Fields
    ------
    status : str
        ``"ok"`` — all axes resolved without warnings.
        ``"partial"`` — resolved but with diagnostic messages.
        ``"failed"`` — no axis information could be recovered.

    inferred_keys : Dict[str, str]
        Axis-name → size-expression mapping auto-discovered from
        ``T.ceildiv`` numerators.  Example: ``{"M": "M", "N": "N"}``.
        This is the ``keys`` dict that used to be hand-crafted for the
        legacy ``SplitAxesParser`` / ``TilingAxesParser`` constructors.

    split_params : Dict[str, str]
        Axis-name → block-parameter name for grid-split axes.
        Example: ``{"M": "block_M", "N": "block_N"}``.
        Replaces ``SplitAxesParser``.

    tiling_params : Dict[str, str]
        Axis-name → block-parameter name for inner-loop tiling axes.
        Example: ``{"K": "block_K"}``.
        Replaces ``TilingAxesParser``.

    reduction_axes : List[str]
        Names of reduction axes (from ``T.Pipelined`` loops or ``range``
        loops containing ``T.gemm / T.reduce``).
        Replaces ``ReductionAxesParser``.

    low_dim_axes : List[str]
        Names of innermost (low) dimension axes, identified from the last
        element of ``T.alloc_shared / T.alloc_fragment`` shape tuples.
        Replaces ``LowDimsAxesParser``.

    buf_count : int
        Number of ``T.Buffer / T.Tensor`` parameters in the ``@T.prim_func``
        inner function.  Replaces ``BufferNumsParser``.

    buffer_params : List[str]
        Names of the buffer parameters in the ``@T.prim_func`` inner function.

    axis_pid_dims : Dict[str, int]
        Axis-name → 0-based ``T.Kernel`` grid dimension index.
        Example: ``{"M": 0, "N": 1}``.

    axis_length_exprs : Dict[str, str]
        Axis-name → total-length expression string.
        Same as ``inferred_keys`` for simple single-variable axes.

    diagnostics : List[str]
        Human-readable warnings or analysis notes.
    """

    status: str
    inferred_keys: Dict[str, str]
    split_params: Dict[str, str]
    tiling_params: Dict[str, str]
    reduction_axes: List[str]
    low_dim_axes: List[str]
    buf_count: int
    buffer_params: List[str]
    axis_pid_dims: Dict[str, int]
    axis_length_exprs: Dict[str, str]
    diagnostics: List[str]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_tilelang_axes(
    func_ast: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
    hints: Optional[Mapping[str, object]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> ParseTilelangAxesResult:
    """
    Analyse a TileLang kernel generator function and return axis semantics
    together with buffer-parameter counts.

    This is the single entry-point that replaces the five ``autoparser``
    classes (``SplitAxesParser``, ``TilingAxesParser``, ``ReductionAxesParser``,
    ``LowDimsAxesParser``, ``BufferNumsParser``).

    Parameters
    ----------
    func_ast : ast.AST
        AST of the kernel generator function (or a ``Module`` containing it).
        Obtain with ``ast.parse(textwrap.dedent(inspect.getsource(fn)))``.
    provided_args : Mapping[str, object], optional
        Problem-dimension arguments already supplied by the caller, e.g.
        ``{"M": 1024, "N": 1024}``.  When omitted, axis extents are resolved
        structurally but their state will be ``RUNTIME_NON_TUNABLE``.
    hints : Mapping[str, object], optional
        Optional hint dict.  Supported key:
        ``"tunable_parameter"`` : ``list[str]`` — explicit tunable overrides.
    module_ast : ast.AST, optional
        Full module AST for inter-procedural name resolution.
    entry_function_name : str, optional
        Target function name inside *module_ast*.

    Returns
    -------
    ParseTilelangAxesResult

    Examples
    --------
    **2D elementwise (product-flattened Ascend grid)**::

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
                    ...

        result = parse_tilelang_axes(ast.parse(src), provided_args={"M": 1024, "N": 1024})
        # result.split_params    → {"N": "block_N", "M": "block_M"}
        # result.tiling_params   → {}
        # result.reduction_axes  → []
        # result.buf_count       → 3
        # result.buffer_params   → ["A", "B", "C"]

    **Standard GEMM**::

        result = parse_tilelang_axes(ast.parse(src), provided_args={"M": 1024, "N": 1024, "K": 1024})
        # result.split_params    → {"M": "block_M", "N": "block_N"}
        # result.tiling_params   → {"K": "block_K"}
        # result.reduction_axes  → ["K"]
        # result.low_dim_axes    → ["K", "N"]
        # result.buf_count       → 3
    """
    provided_args = dict(provided_args or {})
    diagnostics: List[str] = []

    # ── 1. Axis-semantic analysis ─────────────────────────────────────────
    semantic = parse_tl_axis_semantic(
        func_ast,
        provided_args=provided_args,
        hints=hints,
        module_ast=module_ast,
        entry_function_name=entry_function_name,
    )
    diagnostics.extend(semantic.diagnostics)

    # ── 2. Buffer-parameter counting (BufferNumsParser) ───────────────────
    #
    # BufferNumsParser needs:
    #   keys       = {axis_name: size_var_name}  — to exclude size variables
    #                from the buffer count (avoids counting M, N as buffers
    #                if they appear as Buffer shape dimensions)
    #   miss_params = tunable params not supplied by the caller
    #
    # Both are derived from the semantic result and provided_args.
    func_node = _resolve_function_node(func_ast, module_ast, entry_function_name)
    tunable_params = _resolve_tunable_params(func_node, provided_args, hints)
    miss_params = sorted(tunable_params)           # params the autotuner will search

    # keys: axis_name → size_variable (for BufferNumsParser exclusion list)
    # Use inferred_keys from the semantic result; these are axis→length_expr
    # pairs, e.g. {"M": "M", "N": "N"}.
    keys: Dict[str, str] = dict(semantic.inferred_keys)

    buf_parser = BufferNumsParser(func_ast, keys=keys, miss_params=miss_params)
    buf_count, buffer_params = buf_parser.parse()

    # ── 3. Assemble result ────────────────────────────────────────────────
    status = semantic.status
    if diagnostics and status == "ok":
        status = "partial"

    return ParseTilelangAxesResult(
        status=status,
        inferred_keys=dict(semantic.inferred_keys),
        split_params=dict(semantic.split_params),
        tiling_params=dict(semantic.tiling_params),
        reduction_axes=list(semantic.reduction_axes),
        low_dim_axes=list(semantic.low_dim_axes),
        buf_count=buf_count,
        buffer_params=list(buffer_params),
        axis_pid_dims=dict(semantic.axis_pid_dims),
        axis_length_exprs=dict(semantic.axis_length_exprs),
        diagnostics=diagnostics,
    )
