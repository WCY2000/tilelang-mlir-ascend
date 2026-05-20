import ast
import inspect
import os
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend
from tilelang.autotuner.dsl_analysis.vv_param_parser import parse_tl_axis_info
from dataclasses import asdict
from pprint import pprint

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
# torch.npu.set_device(15)

SHAPES = [
    (8, 64),
    (8, 128),
    # (8, 2048),
    # (8, 127),
    # (16, 255),
    # (32, 1025),
    # (1024, 10240),
    # (1024, 14336),
    # (1024, 18432),
    # (1024, 22528),
    # (1024, 1048576),
]

def _parse_kernel(fn: object) -> ast.AST:
    source = textwrap.dedent(inspect.getsource(fn))
    return ast.parse(source)


def _print_result(label: str, result) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    pprint(asdict(result), sort_dicts=False)


# ---------------------------------------------------------------------------
# Helper: get a clean AST from a (possibly tilelang-decorated) function
# ---------------------------------------------------------------------------

def _vv_func_ast(fn) -> ast.AST:
    """
    Return a parseable AST of *fn*, stripping ``@decorator`` lines.

    Handles three cases:
      - Plain Python function                        → use directly
      - After ``@tilelang.jit``   (has __jit_impl__) → unwrap to .func
      - After ``@tilelang.autotune`` (AutoTuneImpl)  → unwrap to .jit_impl.func
    """
    raw_fn = fn

    # Unwrap tilelang decorators to reach the original Python function
    if hasattr(fn, "jit_impl") and hasattr(fn.jit_impl, "func"):
        # AutoTuneImpl (after @autotune wrapping @jit)
        raw_fn = fn.jit_impl.func
    elif hasattr(fn, "__jit_impl__") and hasattr(fn.__jit_impl__, "func"):
        # Wrapper function returned by @jit (before @autotune)
        raw_fn = fn.__jit_impl__.func
    elif hasattr(fn, "__wrapped__"):
        raw_fn = fn.__wrapped__

    src = textwrap.dedent(inspect.getsource(raw_fn))

    # Strip @decorator lines so ast.parse only sees the def block
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            src = textwrap.dedent("\n".join(lines[i:]))
            break

    return ast.parse(src)


def _print_vv(vv) -> None:
    print("\n── VV Parser ─────────────────────────────────────────────────")
    print(f"  Status:         {vv.status}")
    print(f"  Inferred keys:  {vv.inferred_keys}")
    print(f"  Split params:   {vv.split_params}")
    print(f"  Tiling params:  {vv.tiling_params}")
    print(f"  Reduction axes: {vv.reduction_axes}")
    print(f"  Low dim axes:   {vv.low_dim_axes}")
    print(f"  Axis pid dims:  {vv.axis_pid_dims}")
    print(f"  Buf count:      {vv.buf_count}")
    print(f"  Buffer params:  {vv.buffer_params}")
    if vv.diagnostics:
        print(f"  Diagnostics:    {vv.diagnostics}")
    print("──────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_single_shape(shape, log_dir: Path):
    tilelang.cache.clear_cache()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            M, N = shape if len(shape) == 2 else (shape[0], 1)

            def ref_prog(x, y):
                return x + y

            def get_config():
                arch = Ascend()
                carver_template = carver.ElementwiseTemplate(
                    shape=[M, N],
                    dtype="float16",
                ).with_arch(arch)
                hints = carver_template.recommend_hints(topk=20)
                configs = []
                for hint in hints:
                    print("Hint:", hint)
                    configs.append({
                        "block_M": hint.block[0],
                        "block_N": hint.block[1],
                    })
                return configs

            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, N, dtype=torch.float16).npu(),
                    torch.randn(M, N, dtype=torch.float16).npu(),
                ]

            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def elementwise_add(M, N, block_M, block_N):
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

            # ── VV Parser ──────────────────────────────────────────────
            # try:
            #     vv = parse_tilelang_axes(
            #         _vv_func_ast(elementwise_add),
            #         provided_args={"M": M, "N": N},
            #     )
            #     _print_vv(vv)
            # except Exception:
            #     print("\n[VV Parser] failed to analyse kernel:")
            #     traceback.print_exc()

            # ── Autotune ───────────────────────────────────────────────
            func = elementwise_add(M, N)
            key_args, key_kwargs = list(elementwise_add._tuner_cache.keys())[-1]
            provided_args = {
                name: val
                for (name, val) in zip(
                    inspect.signature(elementwise_add.jit_impl.func).parameters,
                    key_args,
                )
            }
            print("<<<<< provided_args", provided_args)
            vv = parse_tl_axis_info(
                    _vv_func_ast(elementwise_add),
                    provided_args=provided_args,
                )


            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

            _print_result("vv parser output", vv)

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./shape_logs_2d")
    root_log_dir.mkdir(exist_ok=True)
    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
