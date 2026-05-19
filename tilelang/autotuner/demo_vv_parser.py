"""
Demo: parse_tilelang_axes — TileLang-Ascend axis parser entry point.

Covers three kernel patterns:
  1. 2D elementwise  — product-flattened Ascend grid, no reduction
  2. Standard GEMM   — split + pipelined reduction
  3. 1D elementwise  — no default values, no annotations (user's real kernel)

Run:
    python demo_vv_parser.py
"""

import ast
import inspect
import os
import textwrap

import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

from tilelang.autotuner.vv_parser import parse_tilelang_axes


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def elementwise_add_2d(M, N, block_M, block_N):
    """
    2D elementwise — Ascend NPU product-flattened grid.

    T.Kernel receives a single linearised grid dimension
    T.ceildiv(N, block_N) * T.ceildiv(M, block_M), then the kernel body
    decomposes the flat index with div/mod to recover (by, bx).
    """
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
            bx = cid %  T.ceildiv(N, block_N)
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            B_shared = T.alloc_shared((block_M, block_N), "float16")
            C_local  = T.alloc_fragment((block_M, block_N), "float16")
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.vadd(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return elemAdd

def elementwise_add_1d(M, block_M):
    """
    1D elementwise — no default values, no type annotations on block_M.
    This is the real-world pattern from the autotuner test suite.
    """
    @T.prim_func
    def elemAdd(
        A: T.Tensor((M,), "float16"),
        B: T.Tensor((M,), "float16"),
        C: T.Tensor((M,), "float16"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bid, _):
            offset   = bid * block_M
            A_shared = T.alloc_shared((block_M,), "float16")
            B_shared = T.alloc_shared((block_M,), "float16")
            C_local  = T.alloc_fragment((block_M,), "float16")
            T.copy(A[offset], A_shared)
            T.copy(B[offset], B_shared)
            T.vadd(A_shared, B_shared, C_local)
            T.copy(C_local, C[offset])
    return elemAdd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_kernel(fn: object) -> ast.AST:
    source = textwrap.dedent(inspect.getsource(fn))
    return ast.parse(source)


def _print_result(label: str, result) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Status:          {result.status}")
    print(f"  Inferred keys:   {result.inferred_keys}")
    print(f"  Split params:    {result.split_params}")
    print(f"  Tiling params:   {result.tiling_params}")
    print(f"  Reduction axes:  {result.reduction_axes}")
    print(f"  Low dim axes:    {result.low_dim_axes}")
    print(f"  Axis pid dims:   {result.axis_pid_dims}")
    print(f"  Buf count:       {result.buf_count}")
    print(f"  Buffer params:   {result.buffer_params}")
    if result.diagnostics:
        print(f"  Diagnostics:     {result.diagnostics}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Case 1: 2D elementwise (product-flattened Ascend grid) ────────────
    result1 = parse_tilelang_axes(
        _parse_kernel(elementwise_add_2d),
        provided_args={"M": 1024, "N": 1024},
    )
    _print_result("2D Elementwise  (product-flattened Ascend grid)", result1)

    assert result1.status == "ok"
    assert "block_M" in result1.split_params.values()
    assert "block_N" in result1.split_params.values()
    assert result1.tiling_params == {}
    assert result1.reduction_axes == []
    assert result1.buf_count == 3
    assert set(result1.buffer_params) == {"A", "B", "C"}

    # ── Case 2: 1D elementwise (no default, no annotation) ────────────────
    result2 = parse_tilelang_axes(
        _parse_kernel(elementwise_add_1d),
        provided_args={"M": 2048},
    )
    _print_result("1D Elementwise  (no default, no annotation on block_M)", result2)

    assert result2.status == "ok"
    assert result2.split_params == {"M": "block_M"}
    assert result2.tiling_params == {}
    assert result2.reduction_axes == []
    assert result2.axis_pid_dims == {"M": 0}
    assert result2.buf_count == 3
    assert set(result2.buffer_params) == {"A", "B", "C"}

    print("\n" + "=" * 60)
    print("  All assertions passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
