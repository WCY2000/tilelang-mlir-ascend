import inspect
import os
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.autotuner.dsl_analysis.vv_param_parser import (
    parse_tl_axis_info_from_fn,
    print_vv_axis_parse_result,
)
from tilelang.carver.arch.ascend import Ascend

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

SHAPES = [
    (1, 32, 64),
    (1, 32, 128),
]


def run_single_shape(shape, log_dir: Path):
    # tilelang.cache.clear_cache()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            B, M, N = shape

            def ref_prog(x):
                return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

            def get_config():
                arch = Ascend()

                carver_template = carver.ElementwiseTemplate(
                    shape=[B, M, N],
                    dtype="float32",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []

                for hint in hints:
                    print("Hint:", hint)

                    blocks = hint.block
                    ndim = len(blocks)

                    shape_dims = [B, M, N]
                    result_blocks = []

                    j = 0

                    for dim in shape_dims:
                        if dim == 1:
                            result_blocks.append(1)
                        else:
                            if j < ndim:
                                result_blocks.append(blocks[j])
                                j += 1
                            else:
                                result_blocks.append(1)

                    configs.append(
                        {
                            "block_B": result_blocks[0],
                            "block_M": result_blocks[1],
                            "block_N": result_blocks[2],
                        }
                    )

                return configs

            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.empty(B, M, N).uniform_(-1.0, 1.0).npu(),
                    # torch.randn(B, M, N).half().npu(),
                ]

            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def compute_gelu(B, M, N, block_B, block_M, block_N):

                @T.prim_func
                def gelu_3D(
                    A: T.Tensor((B, M, N), "float32"),
                    B_out: T.Tensor((B, M, N), "float32"),
                ):

                    with T.Kernel(
                        T.ceildiv(B, block_B)
                        * T.ceildiv(M, block_M)
                        * T.ceildiv(N, block_N),
                        is_npu=True,
                    ) as (cid, _):
                        tmp = cid

                        bz = tmp // (T.ceildiv(M, block_M) * T.ceildiv(N, block_N))
                        tmp = tmp % (T.ceildiv(M, block_M) * T.ceildiv(N, block_N))

                        by = tmp // T.ceildiv(N, block_N)
                        bx = tmp % T.ceildiv(N, block_N)
                        scale1 = 1 / (2.0**0.5)
                        scale2 = 1.0
                        scale3 = 0.5
                        A_shared = T.alloc_shared(
                            (block_B, block_M, block_N), "float32"
                        )
                        B_local = T.alloc_fragment(
                            (block_B, block_M, block_N), "float32"
                        )
                        C_local = T.alloc_fragment(
                            (block_B, block_M, block_N), "float32"
                        )
                        D_local = T.alloc_fragment(
                            (block_B, block_M, block_N), "float32"
                        )
                        T.copy(
                            A[bz * block_B, by * block_M, bx * block_N],
                            A_shared,
                        )
                        T.vmul(A_shared, scale1, B_local)
                        T.npuir_verf(B_local, C_local)
                        T.vadd(C_local, scale2, C_local)
                        T.vmul(C_local, scale3, C_local)
                        T.vmul(A_shared, C_local, D_local)

                        T.copy(
                            D_local,
                            B_out[bz * block_B, by * block_M, bx * block_N],
                        )

                return gelu_3D

            func = compute_gelu(B, M, N)
            key_args, _ = list(compute_gelu._tuner_cache.keys())[-1]
            provided_args = {
                name: val
                for (name, val) in zip(
                    inspect.signature(compute_gelu.jit_impl.func).parameters,
                    key_args,
                    strict=False,
                )
            }
            print("<<<<< provided_args", provided_args)
            vv = parse_tl_axis_info_from_fn(
                compute_gelu,
                provided_args=provided_args,
            )

            print("\nBest Config:")
            print(func.get_tuner_result())
            print("\nTest passed!")

            print_vv_axis_parse_result("VV parser output", vv)

        except Exception:
            print("\nERROR OCCURRED\n")
            traceback.print_exc()

    print(f"Finished shape {shape}, log saved to {log_file}")


def main():
    root_log_dir = Path("./gelu_shape_logs_3d")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
