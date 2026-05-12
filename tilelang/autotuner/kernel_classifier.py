"""Kernel-type classification for tilelang programs.

A tilelang kernel is classified as one of three types:

* ``"vector"``  – only element-wise / memory operations; runs on Vector units.
* ``"cube"``    – only matrix-multiply operations; runs on Cube (MTE2) units.
* ``"mix"``     – contains *both* kinds of operation (e.g. ``tl.dot`` alongside
                  element-wise work); the compiler must schedule across both unit
                  types.

The public entry-point is :func:`resolve_kernel_type`, which first checks an
optional *hints* dictionary (letting the caller override the analysis) and then
falls back to static AST inspection of the decorated function when the hint is
``"auto"`` (the default).

"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All recognised kernel-type labels (including the sentinel ``"auto"``).
SUPPORTED_KERNEL_TYPES: frozenset[str] = frozenset({"vector", "cube", "mix", "auto"})

#: Attribute names on the ``tl`` namespace that indicate Cube-unit (matrix)
#: operations.  Extend this set as new intrinsics are added to tilelang.
_CUBE_OPERATOR_ATTRS: frozenset[str] = frozenset({"gemm"})

#: The namespace prefix used for tilelang intrinsics in user-written kernels.
_TL_NAMESPACE: str = "T"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_hint_kernel_type(hints: Optional[Dict[str, Any]]) -> str:
    if not hints:
        return "auto"

    kernel_type = hints.get("kernel_type", "auto")
    if isinstance(kernel_type, str):
        kernel_type = kernel_type.strip().lower()

    if kernel_type not in SUPPORTED_KERNEL_TYPES:
        raise ValueError(
            f"Unsupported kernel_type {kernel_type!r}; "
            f"expected one of: {sorted(SUPPORTED_KERNEL_TYPES)}"
        )
    return kernel_type


def _base_name(expr: ast.AST) -> Optional[str]:
    """Walk a (possibly chained) attribute access to its root :class:`ast.Name`.

    For example, given the AST node for ``tl.something.dot``, this returns
    ``"tl"``.

    Parameters
    ----------
    expr:
        Any AST node.

    Returns
    -------
    str or None
        The identifier of the root name, or ``None`` if the expression does
        not terminate in a plain name.
    """
    node = expr
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _has_cube_ops(func_ast: ast.AST) -> bool:
    """Return ``True`` if *func_ast* contains at least one Cube-unit call.

    A *Cube-unit call* is any call of the form ``tl.<attr>(...)`` where
    ``<attr>`` is listed in :data:`_CUBE_OPERATOR_ATTRS`.

    Parameters
    ----------
    func_ast:
        The root AST node to inspect (typically an :class:`ast.Module` or
        :class:`ast.FunctionDef`).
    """
    print("line 95 _has_cube_ops")
    print("func_ast", func_ast)
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        if not isinstance(func_node, ast.Attribute):
            continue
        if func_node.attr not in _CUBE_OPERATOR_ATTRS:
            continue
        if _base_name(func_node) == _TL_NAMESPACE:
            return True
    print("line 106 _has_cube_ops")
    return False


def _has_vector_ops(func_ast: ast.AST) -> bool:
    """Return ``True`` if *func_ast* contains at least one Vector-unit operation.

    A *Vector-unit operation* is any ``tl.<attr>(...)`` call whose attribute
    name is **not** in :data:`_CUBE_OPERATOR_ATTRS`.  Plain arithmetic /
    indexing nodes are also treated as vector work.

    Parameters
    ----------
    func_ast:
        The root AST node to inspect.
    """
    for node in ast.walk(func_ast):
        # Plain arithmetic / augmented-assign nodes are always vector work.
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.AugAssign)):
            return True
        # tl.<non-cube-attr>(...) calls
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        if not isinstance(func_node, ast.Attribute):
            continue
        if func_node.attr in _CUBE_OPERATOR_ATTRS:
            continue
        if _base_name(func_node) == _TL_NAMESPACE:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_kernel_type_from_dsl(func_ast: Optional[ast.AST]) -> str:
    if func_ast is None:
        return "vector"

    try:
        has_cube = _has_cube_ops(func_ast)
        has_vector = _has_vector_ops(func_ast)
        print("<<<<< has_cube:", has_cube)
        print("<<<<< has_vector:", has_vector)
    except Exception:
        return "vector"

    if has_cube:
        return "mix"
    if has_vector:
        return "vector"

def resolve_kernel_type(
    hints: Optional[Dict[str, Any]],
    func_ast: Optional[ast.AST] = None,
) -> str:
    kernel_type = _get_hint_kernel_type(hints)
    if kernel_type != "auto":
        return kernel_type
    return classify_kernel_type_from_dsl(func_ast)

def _get_inner_prim_func_ast(outer_fn) -> ast.AST | None:
    """
    Parse the source of *outer_fn* and return the AST node of the first
    nested function definition (the prim_func, e.g. elemAdd).

    Returns the full module AST as fallback if no nested def is found.
    """
    try:
        src = inspect.getsource(outer_fn)
        src = inspect.cleandoc(src)
        module_ast = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None

    # The outer function is the first FunctionDef in the module
    outer_def = next(
        (n for n in ast.walk(module_ast) if isinstance(n, ast.FunctionDef)),
        None,
    )
    if outer_def is None:
        return module_ast  # fallback

    # Find the first nested FunctionDef inside the outer one
    inner_def = next(
        (
            n for n in ast.walk(outer_def)
            if isinstance(n, ast.FunctionDef) and n is not outer_def
        ),
        None,
    )
    return inner_def if inner_def is not None else module_ast


def _unwrap_to_original_func(kernel_fn):
    """
    Navigate AutoTuneImpl → _JitImplementation → original Python function.
    Returns the original func, or kernel_fn itself if the chain is absent.
    """
    # AutoTuneImpl stores the jit wrapper in .jit_impl
    jit_impl = getattr(kernel_fn, "jit_impl", None)
    if jit_impl is None:
        return kernel_fn

    # _JitImplementation stores the original function in .func
    original_func = getattr(jit_impl, "func", None)
    return original_func if original_func is not None else kernel_fn


def analyze_kernel_type(kernel_fn, hints: dict | None = None) -> str:
    """
    Resolve the kernel type of *kernel_fn*.

    Traversal order:
      1. AutoTuneImpl.jit_impl.func  (the original Python generator)
      2. First nested FunctionDef inside that function  (the prim_func body)
      3. resolve_kernel_type() on that AST node
    """
    original_func = _unwrap_to_original_func(kernel_fn)
    func_ast = _get_inner_prim_func_ast(original_func)
    return resolve_kernel_type(hints, func_ast)