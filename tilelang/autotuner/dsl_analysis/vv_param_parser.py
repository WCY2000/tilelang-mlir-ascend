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
VV parameter parser for TileLang-Ascend kernel code.

Analyses the Python source of a TileLang kernel *generator* function via AST
inspection and recovers axis-level semantics (split / tiling / reduction /
low-dim / extent) in the same output format produced by ``vv_param_parser_v2``
for Triton-Ascend kernels.

Signature classification design
--------------------------------
TileLang outer generator functions carry **no** ``tl.constexpr`` annotations.
Neither problem-dimension params (``M``, ``N``, ``K``) nor block-size params
(``block_M``, ``block_K``) carry any annotation at all::

    def elementwise_add(M, block_M):      # no annotation whatsoever
        ...
        T.ceildiv(M, block_M)             # this is the structural signal

``classify_length_symbol`` from ``axis_length_resolver`` gates its
``provided_args`` lookup behind an ``is_constexpr`` check::

    if signature.is_constexpr(length_symbol):
        if length_symbol in provided_args:
            return FIXED_COMPILE_TIME, value   # only reachable if constexpr
        return TUNABLE, None
    # falls through to RUNTIME_NON_TUNABLE for everything else

Therefore ``_build_tl_signature`` must mark **any parameter that appears
inside a ``T.ceildiv`` expression** (numerator OR denominator) as
``is_constexpr=True``.  Only parameters with no relationship to any
``T.ceildiv`` (e.g. ``num_stages``, ``threads``) remain ``False``.

Classification outcome
~~~~~~~~~~~~~~~~~~~~~~
+------------------+-------------------------------------+------------------+
| param            | condition                           | state            |
+==================+=====================================+==================+
| ``M`` / ``N``    | is_constexpr=True, in provided_args | FIXED_COMPILE_TIME|
+------------------+-------------------------------------+------------------+
| ``M`` / ``N``    | is_constexpr=True, NOT in provided  | TUNABLE (dyn)    |
+------------------+-------------------------------------+------------------+
| ``block_M``      | is_constexpr=True, NOT in provided  | TUNABLE          |
+------------------+-------------------------------------+------------------+
| ``num_stages``   | is_constexpr=False                  | RUNTIME_NON_TUNABLE|
+------------------+-------------------------------------+------------------+

TileLang pattern reference
--------------------------
**Grid / split axes**::

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                  threads=128) as (bx, by):

**Tiling / reduction axes**::

    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    for k in range(T.ceildiv(K, block_K)):
        T.gemm(A_shared, B_shared, C_local)

**Low-dim axes**::

    A_shared = T.alloc_shared((block_M, block_K), dtype)
    #                                    ^^^^^^^  K-axis is the low dim

**Grid-variable indirection**::

    grid_m = T.ceildiv(M, block_M)
    with T.Kernel(grid_m, ...) as (bx, ...):

**Product-flattened grid** (Ascend NPU)::

    with T.Kernel(T.ceildiv(M, bM) * T.ceildiv(N, bN), ...) as cid:
        by = cid // T.ceildiv(N, bN)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .axis_length_resolver import classify_length_symbol
from .axis_semantic_schema import (
    AxisExtent,
    AxisSemanticInfo,
    AxisSemanticResult,
    AxisSplit,
    AxisTiling,
)
from .dynamic_source_utils import resolve_dynamic_source
from .schema import AXIS_LENGTH_STATE_TUNABLE, ParameterSpec, SignatureInfo
from .vv_param_parser_v2 import VvAxisInfoV2, VvAxisParseResultV2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CEILDIV_ATTRS: frozenset = frozenset(("ceildiv", "cdiv"))
_REDUCTION_CALLS: frozenset = frozenset(("gemm", "reduce", "dot", "matmul"))
_ALLOC_CALLS: frozenset = frozenset(("alloc_shared", "alloc_fragment"))
_LOOP_RANGE_IDS: frozenset = frozenset(("range", "tl_range"))


# ---------------------------------------------------------------------------
# Low-level AST helpers
# ---------------------------------------------------------------------------


def _is_t_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "T"
    )


def _is_t_call(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Call) and _is_t_attr(node.func, attr)


def _is_ceildiv(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _CEILDIV_ATTRS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
        and len(node.args) == 2
    )


def _is_floordiv_shorthand(node: ast.AST) -> bool:
    """``(axis + param - 1) // param`` — manual equivalent of T.ceildiv."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and isinstance(node.right, ast.Name)
    )


def _is_prim_func_decorated(func_node: ast.FunctionDef) -> bool:
    for dec in func_node.decorator_list:
        if _is_t_attr(dec, "prim_func"):
            return True
        if isinstance(dec, ast.Call) and _is_t_attr(dec.func, "prim_func"):
            return True
    return False


def _ast_to_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Function-node resolution
# ---------------------------------------------------------------------------


def _resolve_function_node(
    func_ast: ast.AST,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> ast.AST:
    """
    Locate the outer TileLang kernel generator function (skip @T.prim_func).

    Resolution order:
    1. Named lookup in module_ast / func_ast (if entry_function_name given).
    2. Direct use if func_ast is already a non-prim_func FunctionDef.
    3. First non-prim_func function containing a T.Kernel call.
    4. First non-prim_func function (last-resort fallback).
    """

    def _find_by_name(tree: ast.AST, name: str) -> Optional[ast.AST]:
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
                and not _is_prim_func_decorated(node)
            ):
                return node
        return None

    if entry_function_name:
        if module_ast is not None:
            found = _find_by_name(module_ast, entry_function_name)
            if found is not None:
                return found
        if isinstance(func_ast, ast.Module):
            found = _find_by_name(func_ast, entry_function_name)
            if found is not None:
                return found

    if isinstance(
        func_ast, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and not _is_prim_func_decorated(func_ast):
        return func_ast

    search_root = module_ast if module_ast is not None else func_ast
    for node in ast.walk(search_root):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_prim_func_decorated(node):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and _is_t_attr(child.func, "Kernel"):
                return node

    for node in ast.walk(search_root):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not _is_prim_func_decorated(node):
            return node

    return func_ast


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def _get_function_param_names(func_node: ast.AST) -> List[str]:
    """Return all positional + keyword-only parameter names."""
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = func_node.args
    ordered = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    return [a.arg for a in ordered if isinstance(a, ast.arg)]


def _get_default_param_names(func_node: ast.AST) -> Set[str]:
    """Return parameter names that carry a default value."""
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    args = func_node.args
    positional = list(args.posonlyargs) + list(args.args)
    n_defaults = len(args.defaults)
    defaults: Set[str] = {
        positional[i].arg for i in range(len(positional) - n_defaults, len(positional))
    }
    for i, arg in enumerate(args.kwonlyargs):
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            defaults.add(arg.arg)
    return defaults


# ---------------------------------------------------------------------------
# Ceildiv parameter classification
# ---------------------------------------------------------------------------


def _extract_ceildiv_divisor_names(func_ast: ast.AST) -> Set[str]:
    """
    Return every ``Name`` used as a divisor in ``T.ceildiv`` or floor-div
    shorthand expressions.  These are the block-size (tile-size) params
    (``block_M``, ``block_K``, …) that the autotuner will search.

    ``num_stages`` / ``threads`` are never divisors in ``T.ceildiv``,
    so they are naturally excluded.
    """
    names: Set[str] = set()
    for node in ast.walk(func_ast):
        if _is_ceildiv(node) and isinstance(node.args[1], ast.Name):
            names.add(node.args[1].id)
        if _is_floordiv_shorthand(node) and isinstance(node.right, ast.Name):
            names.add(node.right.id)
    return names


def _extract_ceildiv_numerator_param_names(
    func_ast: ast.AST,
    all_param_names: Set[str],
) -> Set[str]:
    """
    Return function parameter names that appear anywhere inside a T.ceildiv
    *numerator* expression.

    These are the "problem dimension" variables (``M``, ``N``, ``K``, …)
    that define the total size of each axis.  They must be marked
    ``is_constexpr=True`` so that ``classify_length_symbol`` can find their
    concrete values in ``provided_args`` and return ``FIXED_COMPILE_TIME``.

    Without this, ``classify_length_symbol`` falls through to
    ``RUNTIME_NON_TUNABLE`` because the ``provided_args`` lookup is gated
    behind the ``is_constexpr`` check.

    Examples::

        T.ceildiv(M, block_M)                        → {"M"}
        T.ceildiv(batch * heads * seq_len, block_M)  → {"batch","heads","seq_len"}
        T.ceildiv(M + 1, block_M)                    → {"M"}   (literals ignored)
    """
    names: Set[str] = set()
    for node in ast.walk(func_ast):
        if not _is_ceildiv(node):
            continue
        # Walk every Name inside the numerator expression
        for child in ast.walk(node.args[0]):
            if isinstance(child, ast.Name) and child.id in all_param_names:
                names.add(child.id)
    return names


# ---------------------------------------------------------------------------
# TileLang-specific signature builder
# ---------------------------------------------------------------------------


def _build_tl_signature(
    func_node: ast.AST,
    provided_args: Mapping[str, object],
    tunable_params: Set[str],
) -> SignatureInfo:
    """
    Build a :class:`SignatureInfo` for a TileLang kernel generator function.

    **Why this is needed**

    TileLang outer generator functions carry **no** ``tl.constexpr``
    annotations — neither problem dimensions nor block sizes have any
    annotation at all.  ``extract_signature_info`` (which scans for
    ``tl.constexpr``) would return an empty constexpr list, causing
    ``classify_length_symbol`` to classify everything as
    ``RUNTIME_NON_TUNABLE``.

    **The gating problem**

    ``classify_length_symbol`` only checks ``provided_args`` when
    ``is_constexpr`` is ``True``::

        if signature.is_constexpr(length_symbol):
            if length_symbol in provided_args:
                return FIXED_COMPILE_TIME, value   # only reachable if constexpr!
            return TUNABLE, None
        ...
        return RUNTIME_NON_TUNABLE, None           # M would land here without fix

    Therefore **any parameter that appears in a T.ceildiv expression**
    (numerator or denominator) must be marked ``is_constexpr=True``:

    * **Numerator params** (``M``, ``N``, ``K``) — problem dimensions.
      Marked constexpr so ``classify_length_symbol`` can look them up in
      ``provided_args`` and return ``FIXED_COMPILE_TIME``.

    * **Divisor params** (``block_M``, ``block_K``) — tile-size params
      to be autotuned.  Marked constexpr so ``classify_length_symbol``
      returns ``TUNABLE`` (they are NOT in ``provided_args``).

    * **Unrelated params** (``num_stages``, ``threads``) — never appear
      in any ``T.ceildiv``, so ``is_constexpr=False`` →
      ``RUNTIME_NON_TUNABLE``.  This is correct; the parser does not
      need to classify them as axis sizes.

    Classification table for ``def matmul(M,N,K,block_M=64,block_N=64,block_K=32,num_stages=2)``
    with ``provided_args={"M":1024,"N":1024,"K":1024}``::

        M, N, K      is_constexpr=True  in provided_args  → FIXED_COMPILE_TIME
        block_M/N/K  is_constexpr=True  NOT in provided   → TUNABLE
        num_stages   is_constexpr=False                   → RUNTIME_NON_TUNABLE
    """
    all_param_names = set(_get_function_param_names(func_node))
    param_names = _get_function_param_names(func_node)
    defaults = _get_default_param_names(func_node)

    # Problem-dimension params (ceildiv numerators: M, N, K, …) are
    # constexpr-equivalent ONLY when they are in provided_args.
    #   - In provided_args → is_constexpr=True  → FIXED_COMPILE_TIME (value known)
    #   - NOT in provided_args → is_constexpr=False → RUNTIME_NON_TUNABLE (dynamic shape)
    #
    # This prevents M from being classified as TUNABLE (autotuner target) when
    # no concrete value is supplied — M is a runtime shape, not a tile-size knob.
    problem_dim_params = _extract_ceildiv_numerator_param_names(
        func_node, all_param_names
    )
    provided_keys = set(provided_args.keys())
    provided_problem_dims = problem_dim_params & provided_keys

    # axis_relevant: params that classify_length_symbol should treat as constexpr
    #   - block params (tunable divisors): always constexpr → TUNABLE
    #   - provided problem dims:           constexpr       → FIXED_COMPILE_TIME
    #   - num_stages, threads, etc.:       NOT constexpr   → RUNTIME_NON_TUNABLE
    axis_relevant = tunable_params | provided_problem_dims

    parameters = [
        ParameterSpec(
            name=name,
            is_constexpr=(name in axis_relevant),
            has_default=(name in defaults),
        )
        for name in param_names
    ]
    return SignatureInfo(parameters=parameters)


# ---------------------------------------------------------------------------
# Tunable-parameter resolution
# ---------------------------------------------------------------------------


def _resolve_tunable_params(
    func_node: ast.AST,
    provided_args: Mapping[str, object],
    hints: Optional[Mapping[str, object]] = None,
) -> Set[str]:
    """
    Identify the autotuner-candidate block-size parameters.

    A parameter is *tunable* when all of:
    1. Declared in the outer generator's signature.
    2. NOT present in *provided_args*.
    3. Appears as a **divisor** in at least one ``T.ceildiv`` expression.
       This filters out ``num_stages``, ``threads``, and any other knob
       that has no tiling relationship.

    Fallback: when no ceildiv divisors are found, every missing param is
    returned (handles degenerate kernels without standard grid expressions).
    """
    all_params = set(_get_function_param_names(func_node))
    provided = set((provided_args or {}).keys())
    missing = all_params - provided

    ceildiv_divisors = _extract_ceildiv_divisor_names(func_node)
    tunable = (missing & ceildiv_divisors) if ceildiv_divisors else missing

    if hints:
        for item in hints.get("tunable_parameter", []):
            if isinstance(item, str):
                tunable.add(item)

    return tunable


# ---------------------------------------------------------------------------
# Ceildiv alias map  (var = T.ceildiv(axis, param))
# ---------------------------------------------------------------------------


def _build_ceildiv_alias_map(func_ast: ast.AST) -> Dict[str, Tuple[str, str]]:
    """
    Build ``{alias_var: (axis_expr_text, param_name)}`` for every
    ``alias = T.ceildiv(axis_expr, param)`` or floor-div equivalent.

    Example::

        grid_m = T.ceildiv(M, block_M)
        → {"grid_m": ("M", "block_M")}
    """
    result: Dict[str, Tuple[str, str]] = {}
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        val = node.value
        alias = node.targets[0].id
        if _is_ceildiv(val):
            result[alias] = (_ast_to_text(val.args[0]), _ast_to_text(val.args[1]))
        elif _is_floordiv_shorthand(val) and isinstance(val.right, ast.Name):
            result[alias] = (_ast_to_text(val.left), val.right.id)
    return result


# ---------------------------------------------------------------------------
# Internal evidence dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _SplitEvidence:
    axis_name: str
    param_name: str
    pid_dim: int
    axis_total_expr: str
    source: str = "T.Kernel"
    confidence: float = 0.95


@dataclass
class _TilingEvidence:
    axis_name: str
    param_name: str
    is_reduction: bool
    axis_total_expr: str
    loop_var: Optional[str] = None
    source: str = "T.Pipelined"
    confidence: float = 0.90


# ---------------------------------------------------------------------------
# Split evidence extraction
# ---------------------------------------------------------------------------


def _process_kernel_arg(
    arg: ast.AST,
    dim_idx: int,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
    seen_params: Set[str],
) -> List[_SplitEvidence]:
    """
    Extract _SplitEvidence from one T.Kernel positional argument.

    Handles: direct ceildiv · product (M*N flattened grid) · alias · floordiv.
    """
    results: List[_SplitEvidence] = []

    # Product: T.ceildiv(M,bM) * T.ceildiv(N,bN) — recurse both sides
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
        results.extend(
            _process_kernel_arg(
                arg.left, dim_idx, tunable_params, ceildiv_alias_map, seen_params
            )
        )
        results.extend(
            _process_kernel_arg(
                arg.right, dim_idx, tunable_params, ceildiv_alias_map, seen_params
            )
        )
        return results

    # Direct T.ceildiv(axis, param)
    if _is_ceildiv(arg):
        param_text = _ast_to_text(arg.args[1])
        if param_text in tunable_params and param_text not in seen_params:
            axis_text = _ast_to_text(arg.args[0])
            seen_params.add(param_text)
            results.append(
                _SplitEvidence(
                    axis_name=axis_text,
                    param_name=param_text,
                    pid_dim=dim_idx,
                    axis_total_expr=axis_text,
                )
            )
        return results

    # Floor-div shorthand: (axis + param - 1) // param
    if _is_floordiv_shorthand(arg) and isinstance(arg.right, ast.Name):
        param_text = arg.right.id
        if param_text in tunable_params and param_text not in seen_params:
            axis_text = _ast_to_text(arg.left)
            seen_params.add(param_text)
            results.append(
                _SplitEvidence(
                    axis_name=axis_text,
                    param_name=param_text,
                    pid_dim=dim_idx,
                    axis_total_expr=axis_text,
                )
            )
        return results

    # Alias reference: grid_m → ceildiv_alias_map
    if isinstance(arg, ast.Name) and arg.id in ceildiv_alias_map:
        axis_text, param_text = ceildiv_alias_map[arg.id]
        if param_text in tunable_params and param_text not in seen_params:
            seen_params.add(param_text)
            results.append(
                _SplitEvidence(
                    axis_name=axis_text,
                    param_name=param_text,
                    pid_dim=dim_idx,
                    axis_total_expr=axis_text,
                )
            )

    return results


def _extract_split_evidence_from_kernel(
    func_ast: ast.AST,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> List[_SplitEvidence]:
    results: List[_SplitEvidence] = []
    seen_params: Set[str] = set()
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not _is_t_attr(node.func, "Kernel"):
            continue
        for dim_idx, arg in enumerate(node.args):
            results.extend(
                _process_kernel_arg(
                    arg, dim_idx, tunable_params, ceildiv_alias_map, seen_params
                )
            )
    return results


def _fallback_split_scan(
    func_ast: ast.AST,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
    already_found: Set[str],
) -> List[_SplitEvidence]:
    """
    Fallback for kernels where T.Kernel receives a constant core count.

    Pass A: cid // T.ceildiv(N, bN)   confidence 0.75
    Pass B: axis_expr // block_param   confidence 0.60
    """
    results: List[_SplitEvidence] = []
    seen_params: Set[str] = already_found.copy()
    dim_counter = 0

    for node in ast.walk(func_ast):
        if not isinstance(node, ast.BinOp):
            continue
        if not isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            continue
        rhs = node.right
        if _is_ceildiv(rhs):
            axis_text = _ast_to_text(rhs.args[0])
            param_text = _ast_to_text(rhs.args[1])
        elif _is_floordiv_shorthand(rhs) and isinstance(rhs.right, ast.Name):
            axis_text = _ast_to_text(rhs.left)
            param_text = rhs.right.id
        else:
            continue
        if param_text not in tunable_params or param_text in seen_params:
            continue
        seen_params.add(param_text)
        results.append(
            _SplitEvidence(
                axis_name=axis_text,
                param_name=param_text,
                pid_dim=dim_counter,
                axis_total_expr=axis_text,
                source="fallback_divmod",
                confidence=0.75,
            )
        )
        dim_counter += 1

    for node in ast.walk(func_ast):
        if not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Name)
            and node.right.id in tunable_params
            and node.right.id not in seen_params
        ):
            continue
        param_text = node.right.id
        seen_params.add(param_text)
        results.append(
            _SplitEvidence(
                axis_name=_ast_to_text(node.left),
                param_name=param_text,
                pid_dim=dim_counter,
                axis_total_expr=_ast_to_text(node.left),
                source="fallback_floordiv",
                confidence=0.60,
            )
        )
        dim_counter += 1

    return results


def _extract_split_evidence(
    func_ast: ast.AST,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> List[_SplitEvidence]:
    primary = _extract_split_evidence_from_kernel(
        func_ast, tunable_params, ceildiv_alias_map
    )
    found_params = {ev.param_name for ev in primary}
    if len(found_params) < len(tunable_params):
        primary.extend(
            _fallback_split_scan(
                func_ast, tunable_params, ceildiv_alias_map, found_params
            )
        )
    return primary


# ---------------------------------------------------------------------------
# Tiling / reduction loop extraction
# ---------------------------------------------------------------------------


def _body_has_reduction(stmts: List[ast.stmt]) -> bool:
    for stmt in stmts:
        for child in ast.walk(stmt):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "T"
                and child.func.attr in _REDUCTION_CALLS
            ):
                return True
    return False


def _extract_loop_bound_info(
    iter_call: ast.Call,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """Return (axis_total_expr, param_name) from a loop bound, or None."""
    if not iter_call.args:
        return None
    bound = iter_call.args[0]

    if _is_ceildiv(bound):
        param_text = _ast_to_text(bound.args[1])
        if param_text in tunable_params:
            return _ast_to_text(bound.args[0]), param_text
        return None

    if _is_floordiv_shorthand(bound) and isinstance(bound.right, ast.Name):
        if bound.right.id in tunable_params:
            return _ast_to_text(bound.left), bound.right.id
        return None

    if isinstance(bound, ast.Name) and bound.id in ceildiv_alias_map:
        axis_text, param_text = ceildiv_alias_map[bound.id]
        if param_text in tunable_params:
            return axis_text, param_text
        return None

    return None


def _extract_tiling_evidence(
    func_ast: ast.AST,
    tunable_params: Set[str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> List[_TilingEvidence]:
    """
    Walk ast.For nodes and extract tiling / reduction evidence.

    * T.Pipelined → always reduction (software-pipelined)  confidence 0.90
    * range / tl_range + T.gemm/reduce/dot in body → reduction  confidence 0.80
    * range / tl_range without reduction primitive → tiling only  confidence 0.80

    Note: ``num_stages`` in ``T.Pipelined(..., num_stages=n)`` is a keyword
    argument and is never confused with a tunable axis parameter because it
    does not appear as a ceildiv divisor.
    """
    results: List[_TilingEvidence] = []
    seen_params: Set[str] = set()

    for node in ast.walk(func_ast):
        if not isinstance(node, ast.For):
            continue
        iter_node = node.iter
        if not isinstance(iter_node, ast.Call):
            continue

        is_pipelined = _is_t_call(iter_node, "Pipelined")
        is_range = (
            isinstance(iter_node.func, ast.Name)
            and iter_node.func.id in _LOOP_RANGE_IDS
        )
        if not (is_pipelined or is_range):
            continue

        info = _extract_loop_bound_info(iter_node, tunable_params, ceildiv_alias_map)
        if info is None:
            continue
        axis_total_expr, param_name = info
        if param_name in seen_params:
            continue
        seen_params.add(param_name)

        is_reduction = is_pipelined or _body_has_reduction(node.body)
        loop_var = node.target.id if isinstance(node.target, ast.Name) else None

        results.append(
            _TilingEvidence(
                axis_name=axis_total_expr,
                param_name=param_name,
                is_reduction=is_reduction,
                axis_total_expr=axis_total_expr,
                loop_var=loop_var,
                source="T.Pipelined" if is_pipelined else "range",
                confidence=0.90 if is_pipelined else 0.80,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Low-dim extraction
# ---------------------------------------------------------------------------


def _shape_elt_to_axis(
    elt: ast.AST,
    param_to_axis: Dict[str, str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> Optional[str]:
    if isinstance(elt, ast.Name):
        name = elt.id
        if name in param_to_axis:
            return param_to_axis[name]
        if name in ceildiv_alias_map:
            return ceildiv_alias_map[name][0]
        return name  # raw problem-size var used directly in shape

    if _is_ceildiv(elt):
        return _ast_to_text(elt.args[0])

    for child in ast.walk(elt):
        if isinstance(child, ast.Name):
            if child.id in param_to_axis:
                return param_to_axis[child.id]
            if child.id in ceildiv_alias_map:
                return ceildiv_alias_map[child.id][0]

    return None


def _extract_low_dim_axes(
    func_ast: ast.AST,
    param_to_axis: Dict[str, str],
    ceildiv_alias_map: Dict[str, Tuple[str, str]],
) -> List[str]:
    """Last element of each T.alloc_shared / T.alloc_fragment shape → low dim."""
    low_dims: List[str] = []
    seen: Set[str] = set()

    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "T"
            and call.func.attr in _ALLOC_CALLS
        ):
            continue
        if not call.args:
            continue
        shape_arg = call.args[0]
        if not isinstance(shape_arg, (ast.Tuple, ast.List)) or not shape_arg.elts:
            continue

        axis = _shape_elt_to_axis(shape_arg.elts[-1], param_to_axis, ceildiv_alias_map)
        if axis and axis not in seen:
            seen.add(axis)
            low_dims.append(axis)

    return low_dims


# ---------------------------------------------------------------------------
# Axis extent construction
# ---------------------------------------------------------------------------


def _build_axis_extent(
    axis_name: str,
    signature: SignatureInfo,
    provided_args: Mapping[str, object],
    load_derived_symbols: Optional[Set[str]] = None,
) -> AxisExtent:
    """
    Build an AxisExtent for *axis_name* using the real classify_length_symbol.

    With _build_tl_signature providing a correct SignatureInfo
    (both problem dims and block params marked is_constexpr=True):

    * axis_name="M", provided={"M":64}      → FIXED_COMPILE_TIME, value=64
    * axis_name="M", provided={}            → RUNTIME_NON_TUNABLE (dynamic shape)
    * axis_name="block_M", provided={"M":64}→ TUNABLE (block_M not in provided)

    Without the fix (M not marked constexpr), classify_length_symbol would
    return RUNTIME_NON_TUNABLE for M even when M=64 is in provided_args,
    because the provided_args lookup is gated behind is_constexpr.
    """
    load_derived_symbols = load_derived_symbols or set()
    state, const_value = classify_length_symbol(axis_name, signature, provided_args)
    dynamic_source = resolve_dynamic_source(
        axis_symbol=axis_name,
        length_expr=axis_name,
        load_derived_symbols=load_derived_symbols,
    )
    return AxisExtent(
        expr=axis_name,
        state=state,
        const_value=const_value,
        source="ceildiv",
        confidence=0.95,
        dynamic_source=dynamic_source,
    )


# ---------------------------------------------------------------------------
# Main semantic analysis
# ---------------------------------------------------------------------------


def parse_tl_axis_semantic(
    func_ast: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
    hints: Optional[Mapping[str, object]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> AxisSemanticResult:
    """
    Analyse a TileLang kernel generator function and return an
    AxisSemanticResult with split / tiling / reduction / low-dim / extent.
    """
    provided_args = dict(provided_args or {})
    diagnostics: List[str] = []

    # 1. Locate outer generator function (skip @T.prim_func)
    func_node = _resolve_function_node(func_ast, module_ast, entry_function_name)

    # 2. Tunable params: ceildiv DIVISORS not in provided_args
    #    (num_stages / threads excluded because they are never ceildiv divisors)
    tunable_params = _resolve_tunable_params(func_node, provided_args, hints)
    if not tunable_params:
        diagnostics.append(
            "no tunable parameters identified; check provided_args coverage"
        )

    # 3. Build TileLang-specific signature
    #
    #    Rules:
    #      ceildiv divisors  (block_M, block_K)  → is_constexpr=True → TUNABLE
    #      ceildiv numerators (M, N, K)           → is_constexpr=True → FIXED_COMPILE_TIME
    #      unrelated params  (num_stages,threads) → is_constexpr=False → RUNTIME_NON_TUNABLE
    #
    #    classify_length_symbol gates the provided_args lookup behind is_constexpr,
    #    so M must be marked constexpr to get FIXED_COMPILE_TIME when M is in provided_args.
    signature = _build_tl_signature(func_node, provided_args, tunable_params)

    # 4. Ceildiv alias map
    ceildiv_alias_map = _build_ceildiv_alias_map(func_node)

    # 5. Split evidence (T.Kernel grid args)
    split_evidences = _extract_split_evidence(
        func_node, tunable_params, ceildiv_alias_map
    )

    # 6. Tiling / reduction evidence (loops)
    tiling_evidences = _extract_tiling_evidence(
        func_node, tunable_params, ceildiv_alias_map
    )

    # 7. param → axis mapping (for low-dim extraction)
    param_to_axis: Dict[str, str] = {}
    for ev in split_evidences:
        param_to_axis.setdefault(ev.param_name, ev.axis_name)
    for ev in tiling_evidences:
        param_to_axis.setdefault(ev.param_name, ev.axis_name)

    # 8. Low-dim axes
    low_dim_axes_list = _extract_low_dim_axes(
        func_node, param_to_axis, ceildiv_alias_map
    )
    low_dim_set = set(low_dim_axes_list)

    # 9. Ordered unique axis list: split axes (by pid_dim), then tiling-only
    ordered_axes: List[str] = []
    seen_axes: Set[str] = set()

    pid_to_axes: Dict[int, List[str]] = {}
    for ev in split_evidences:
        pid_to_axes.setdefault(ev.pid_dim, []).append(ev.axis_name)
    for pid_dim in sorted(pid_to_axes):
        for ax in pid_to_axes[pid_dim]:
            if ax not in seen_axes:
                ordered_axes.append(ax)
                seen_axes.add(ax)

    for ev in tiling_evidences:
        if ev.axis_name not in seen_axes:
            ordered_axes.append(ev.axis_name)
            seen_axes.add(ev.axis_name)

    if not ordered_axes:
        return AxisSemanticResult(
            axes={},
            axis_length_exprs={},
            fixed_tiling_exprs={},
            axis_pid_dims={},
            inferred_keys={},
            split_params={},
            tiling_params={},
            low_dim_axes=[],
            reduction_axes=[],
            status="failed",
            diagnostics=diagnostics
            + ["no axis information resolved from T.Kernel / T.Pipelined / range"],
        )

    # 10. Aggregate dicts
    split_params: Dict[str, str] = {}
    for ev in split_evidences:
        split_params.setdefault(ev.axis_name, ev.param_name)

    tiling_params: Dict[str, str] = {}
    for ev in tiling_evidences:
        tiling_params.setdefault(ev.axis_name, ev.param_name)

    axis_pid_dims: Dict[str, int] = {}
    for ev in split_evidences:
        axis_pid_dims.setdefault(ev.axis_name, ev.pid_dim)

    tiling_ev_map: Dict[str, _TilingEvidence] = {
        ev.axis_name: ev for ev in tiling_evidences
    }
    reduction_axes_list: List[str] = []
    seen_red: Set[str] = set()
    for ev in tiling_evidences:
        if ev.is_reduction and ev.axis_name not in seen_red:
            reduction_axes_list.append(ev.axis_name)
            seen_red.add(ev.axis_name)

    # 11. Warn about tunable params not mapped to any axis
    all_mapped = set(split_params.values()) | set(tiling_params.values())
    unmapped = tunable_params - all_mapped
    if unmapped:
        diagnostics.append(
            "tunable params not mapped to any axis: {}".format(sorted(unmapped))
        )

    # 12. Build per-axis AxisSemanticInfo
    axes: Dict[str, AxisSemanticInfo] = {}
    axis_length_exprs: Dict[str, str] = {}
    inferred_keys: Dict[str, str] = {}

    for axis_name in ordered_axes:
        extent = _build_axis_extent(axis_name, signature, provided_args)
        if extent.expr:
            axis_length_exprs[axis_name] = extent.expr
            inferred_keys[axis_name] = extent.expr

        tev = tiling_ev_map.get(axis_name)
        is_reduction = axis_name in seen_red

        axes[axis_name] = AxisSemanticInfo(
            axis_name=axis_name,
            extent=extent,
            split=AxisSplit(
                param=split_params.get(axis_name),
                pid_dim=axis_pid_dims.get(axis_name),
                source="T.Kernel",
                confidence=0.95 if axis_name in split_params else 0.0,
            ),
            tiling=AxisTiling(
                param=tiling_params.get(axis_name),
                loop_var=tev.loop_var if tev else None,
                source=tev.source
                if tev
                else ("T.Pipelined" if is_reduction else "range"),
                confidence=tev.confidence
                if (tev and axis_name in tiling_params)
                else 0.0,
                fixed_expr=None,
            ),
            is_low_dim=(axis_name in low_dim_set),
            is_reduction=is_reduction,
            diagnostics=[],
        )

    status = "partial" if diagnostics else "ok"
    if not axes:
        status = "failed"

    return AxisSemanticResult(
        axes=axes,
        axis_length_exprs=axis_length_exprs,
        fixed_tiling_exprs={},
        axis_pid_dims=axis_pid_dims,
        inferred_keys=inferred_keys,
        split_params=split_params,
        tiling_params=tiling_params,
        low_dim_axes=low_dim_axes_list,
        reduction_axes=reduction_axes_list,
        status=status,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _axis_sort_key(axis_name: str) -> Tuple[int, str]:
    return (1 if axis_name.startswith("r") else 0, axis_name)


def parse_tl_axis_info(
    func_ast: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
    hints: Optional[Mapping[str, object]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> VvAxisParseResultV2:
    """
    Parse a TileLang kernel generator and return VvAxisParseResultV2.

    Counterpart of parse_vv_axis_info_v2 for Triton-Ascend.

    Example::

        def elementwise_add(M, block_M):   # no constexpr annotation
            @T.prim_func
            def elemAdd(A: T.Tensor((M,),"float16"), ...):
                with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (bid, _):
                    ...

        result = parse_tl_axis_info(ast.parse(src), provided_args={"M": 64})
        # result.split_params  = {"M": "block_M"}
        # result.axis_pid_dims = {"M": 0}
        # sem.axes["M"].extent.state = "fixed_compile_time"  (value=64)
    """
    semantic_result = parse_tl_axis_semantic(
        func_ast,
        provided_args=provided_args,
        hints=hints,
        module_ast=module_ast,
        entry_function_name=entry_function_name,
    )

    ordered_axis_names = sorted(semantic_result.axes.keys(), key=_axis_sort_key)
    legacy_axes: List[VvAxisInfoV2] = []
    axis_dynamic_sources: Dict[str, str] = {}

    for axis_index, axis_name in enumerate(ordered_axis_names):
        info = semantic_result.axes[axis_name]
        extent = info.extent

        tunable_param: Optional[str] = None
        if extent.state == AXIS_LENGTH_STATE_TUNABLE:
            sym = extent.expr
            if sym and sym.isidentifier():
                tunable_param = sym

        axis_dynamic_sources[axis_name] = extent.dynamic_source

        legacy_axes.append(
            VvAxisInfoV2(
                axis_index=axis_index,
                length_expr=extent.expr,
                axis_symbol=axis_name,
                state=extent.state,
                tunable_param=tunable_param,
                const_value=extent.const_value,
                split_param=info.split.param,
                tiling_param=info.tiling.param,
                fixed_tiling_expr=info.tiling.fixed_expr,
                is_low_dim=info.is_low_dim,
                is_reduction=info.is_reduction,
                dynamic_source=extent.dynamic_source,
            )
        )

    return VvAxisParseResultV2(
        axis_count=len(legacy_axes),
        axes=legacy_axes,
        source="none" if not semantic_result.axes else "tl.ceildiv",
        diagnostics=list(semantic_result.diagnostics),
        axis_length_exprs=dict(semantic_result.axis_length_exprs),
        fixed_tiling_exprs=dict(semantic_result.fixed_tiling_exprs),
        axis_pid_dims=dict(semantic_result.axis_pid_dims),
        inferred_keys=dict(semantic_result.inferred_keys),
        split_params=dict(semantic_result.split_params),
        tiling_params=dict(semantic_result.tiling_params),
        axis_dynamic_sources=axis_dynamic_sources,
        low_dim_axes=list(semantic_result.low_dim_axes),
        reduction_axes=list(semantic_result.reduction_axes),
        status=semantic_result.status,
    )
