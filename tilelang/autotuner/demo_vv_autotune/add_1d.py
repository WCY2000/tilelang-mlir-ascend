import inspect
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang import carver
from tilelang.autotuner.dsl_analysis.vv_param_parser import (
    parse_tl_axis_info_from_fn,
    print_vv_axis_parse_result,
)
from tilelang.utils.npu_arch import AscendArch

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

# torch.npu.set_device(15)

SHAPES = [
    (64,),
    # (128,),
    # (2048,),
    # (127,),
    # (255,),
    # (1025,),
]


def run_single_shape(shape, log_dir: Path):
    tilelang.cache.clear_cache()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print("=" * 80)
        print("Running shape:", shape)
        print("=" * 80)

        try:
            M = shape[0]

            def ref_prog(x, y):
                return x + y

            def get_config():
                arch = AscendArch()

                # 1D template
                carver_template = carver.ElementwiseTemplate(
                    shape=[M],
                    dtype="float16",
                ).with_arch(arch)

                hints = carver_template.recommend_hints(topk=20)

                configs = []
                for hint in hints:
                    print("Hint:", hint)
                    configs.append(
                        {
                            "block_M": hint.block[0],
                        }
                    )

                return configs

            def supply_prog(params):
                torch.manual_seed(0)
                return [
                    torch.randn(M, dtype=torch.float16).npu(),
                    torch.randn(M, dtype=torch.float16).npu(),
                ]

            @tilelang.autotune(
                configs=get_config(),
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                atol=1e-2,
                rtol=1e-2,
            )
            @tilelang.jit(out_idx=[-1], target="npuir")
            def elementwise_add(M, block_M):

                @T.prim_func
                def elemAdd(
                    A: T.Tensor((M,), "float16"),
                    B: T.Tensor((M,), "float16"),
                    C: T.Tensor((M,), "float16"),
                ):
                    with T.Kernel(
                        T.ceildiv(M, block_M),
                        is_npu=True,
                    ) as (bid, _):
                        offset = bid * block_M

                        A_shared = T.alloc_shared((block_M,), "float16")
                        B_shared = T.alloc_shared((block_M,), "float16")
                        C_local = T.alloc_fragment((block_M,), "float16")

                        T.copy(A[offset], A_shared)
                        T.copy(B[offset], B_shared)

                        T.vadd(A_shared, B_shared, C_local)

                        T.copy(C_local, C[offset])

                return elemAdd

            func = elementwise_add(M)
            key_args, key_kwargs = list(elementwise_add._tuner_cache.keys())[-1]
            provided_args = {
                name: val
                for (name, val) in zip(
                    inspect.signature(elementwise_add.jit_impl.func).parameters,
                    key_args,
                    strict=False,
                )
            }
            print("<<<<< provided_args", provided_args)
            vv = parse_tl_axis_info_from_fn(
                elementwise_add,
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
    root_log_dir = Path("./add_shape_logs_1d")
    root_log_dir.mkdir(exist_ok=True)

    for shape in SHAPES:
        shape_str = "x".join(map(str, shape))
        log_dir = root_log_dir / shape_str
        run_single_shape(shape, log_dir)


if __name__ == "__main__":
    main()
