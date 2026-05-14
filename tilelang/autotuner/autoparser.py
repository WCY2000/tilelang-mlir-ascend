"""
Auto-parser for TileLang-Ascend kernel code using AST analysis.

This module provides parsers that statically analyse the Python source of a
TileLang kernel generator function and recover the mapping between tunable
block-size parameters and the problem-dimension axes they tile.

TileLang pattern reference
--------------------------
Split axes (grid dimensions):
    T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by)

Tiling / reduction axes (loop ranges):
    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    for k in range(T.ceildiv(K, block_K)):

Shared / fragment allocations:
    A_shared = T.alloc_shared((block_M, block_K), dtype)

Buffer (pointer) parameters in the inner @T.prim_func:
    def main(A: T.Buffer((M, K), "float16"), B: T.Buffer((K, N), "float16"), ...):
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_t_attr(node: ast.AST, attr: str) -> bool:
    """Return True when *node* looks like ``T.<attr>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "T"
    )


def _is_t_call(node: ast.AST, attr: str) -> bool:
    """Return True when *node* is a Call to ``T.<attr>(...)``."""
    return isinstance(node, ast.Call) and _is_t_attr(node.func, attr)


# ---------------------------------------------------------------------------
# Base parser
# ---------------------------------------------------------------------------

class AutoParser(ast.NodeVisitor):
    """
    Base class for parsing TileLang kernel generator code via AST analysis.

    Provides the visitor infrastructure and a recursive variable-search helper.
    Subclasses implement specific parsing logic by overriding ``visit_*`` methods.
    """

    def __init__(self, func_ast: ast.AST):
        self.func_ast = func_ast

    def parse(self):
        self.visit(self.func_ast)

    def contains_target_var(self, node: ast.AST, var: str) -> bool:
        """
        Recursively check whether *node* or any of its descendants reference *var*.

        :param node: AST subtree to search.
        :param var:  Variable name to look for.
        :return:     ``True`` if *var* is referenced, ``False`` otherwise.
        """
        if isinstance(node, ast.Name) and node.id == var:
            return True
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST) and self.contains_target_var(item, var):
                        return True
            elif isinstance(value, ast.AST):
                if self.contains_target_var(value, var):
                    return True
        return False


# ---------------------------------------------------------------------------
# Axes-key parser (shared axis-discovery logic)
# ---------------------------------------------------------------------------

class AxesKeyParser(AutoParser):
    """
    Shared base for parsers that map tunable block-size parameters to named axes.

    Two complementary strategies are used:

    1. **Ceiling-division pattern** – ``T.ceildiv(axis_var, block_param)``
       directly encodes the relationship and is the canonical TileLang pattern.

    2. **Floor-division shorthand** – ``(axis_var + block_param - 1) // block_param``
       is also recognised as an alternative to ``T.ceildiv``.

    3. **Assignment following** – an intermediate variable that holds a ceildiv
       result is followed back to its origin, so ``grid_m = T.ceildiv(M, block_M);
       T.Kernel(grid_m, ...)`` works correctly.
    """

    #: Names accepted as ceiling-division primitives.
    CEILDIV_ATTRS: Tuple[str, ...] = ("ceildiv", "cdiv")

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: AST of the kernel generator function.
        :param keys:     Mapping ``{axis_name: size_variable_name}``, e.g.
                         ``{"M": "seq_len", "N": "seq_len", "D": "dim"}``.
        """
        super().__init__(func_ast)
        self.keys = keys
        self.checked_vars: List[str] = []
        # Alias map: local_var → canonical name it was directly/transitively
        # assigned from.  Populated once by _collect_raw_aliases().
        # Example: given  block_m = block_M
        #                 bm      = block_m
        # _raw_alias_map == {"block_m": "block_M", "bm": "block_M"}
        self._raw_alias_map: Dict[str, str] = {}
        self._collect_raw_aliases()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_axis(self, var: str, node: Optional[ast.AST] = None) -> Optional[str]:
        """
        Return the axis name whose size variable appears as the *numerator* in a
        ``T.ceildiv(axis_var, var)`` (or equivalent) expression anywhere in *node*.

        :param var:  Tunable parameter name to look up (e.g. ``"block_M"``).
        :param node: Subtree to search; defaults to the entire function AST.
        :return:     Axis name (e.g. ``"M"``) or ``None``.
        """
        if var in self.checked_vars:
            return None
        if node is None:
            node = self.func_ast

        for child in ast.walk(node):
            axis = self._match_ceildiv(var, child)
            if axis is not None:
                return axis
            if isinstance(child, ast.Assign):
                axis = self.handle_assign_node(var, child)
                if axis is not None:
                    return axis

        self.checked_vars.append(var)
        return None

    def handle_assign_node(self, var: str, node: ast.Assign) -> Optional[str]:
        """
        Follow a single-target assignment to find an indirect axis association.

        Example::

            grid_m = T.ceildiv(M, block_M)   # var == "block_M"
            T.Kernel(grid_m, ...)             # detected in visit_Call

        :param var:  The parameter variable being searched.
        :param node: An assignment node to inspect.
        :return:     Axis name or ``None``.
        """
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.targets, list)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return None

        target = node.targets[0].id
        if target in self.checked_vars or var == target:
            return None
        if not self.contains_target_var(node.value, var):
            return None

        axis = self.get_axis(var, node.value)
        if axis:
            return axis
        return self.get_axis(target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_raw_aliases(self):
        """
        Pre-pass: record every simple ``alias = name`` assignment in the AST,
        resolving alias chains transitively.

        Example::

            block_m = block_M   →  _raw_alias_map["block_m"] = "block_M"
            bm      = block_m   →  _raw_alias_map["bm"]      = "block_M"

        This lets all downstream helpers find ``block_M`` even when the kernel
        body uses the lowercase alias ``block_m``.
        """
        direct: Dict[str, str] = {}
        for node in ast.walk(self.func_ast):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
                and node.targets[0].id != node.value.id   # skip self-assignment
            ):
                direct[node.targets[0].id] = node.value.id

        def _resolve(name: str, seen: set) -> str:
            if name in seen:
                return name
            seen.add(name)
            target = direct.get(name)
            return _resolve(target, seen) if target else name

        for alias in direct:
            self._raw_alias_map[alias] = _resolve(alias, set())

    def _canonical(self, name: str) -> str:
        """Return the canonical name for *name*, resolving any alias."""
        return self._raw_alias_map.get(name, name)

    def _match_ceildiv(self, var: str, node: ast.AST) -> Optional[str]:
        """
        Attempt to match ``T.ceildiv(axis_expr, var)`` or the equivalent
        floor-division expression and return the corresponding axis name.

        Both direct matches (divisor == var) and alias matches
        (divisor is an alias that resolves to var) are accepted.
        """
        # ---- canonical: T.ceildiv(axis_expr, var) ----
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in self.CEILDIV_ATTRS
            and len(node.args) == 2
        ):
            axis_expr, divisor_expr = node.args
            if (
                isinstance(divisor_expr, ast.Name)
                and self._canonical(divisor_expr.id) == var
            ):
                for k, v in self.keys.items():
                    if self.contains_target_var(axis_expr, v):
                        return k

        # ---- alternative: (axis_expr + var - 1) // var ----
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Name)
            and self._canonical(node.right.id) == var
        ):
            # numerator should contain var (or its alias) and an axis key
            divisor_id = node.right.id
            if (
                self.contains_target_var(node.left, var)
                or self.contains_target_var(node.left, divisor_id)
            ):
                for k, v in self.keys.items():
                    if self.contains_target_var(node.left, v):
                        return k

        return None

    def _axis_from_expr(self, expr: ast.AST) -> Optional[str]:
        """Return the axis name whose key variable appears in *expr*."""
        if expr is None:
            return None
        for k, v in self.keys.items():
            if self.contains_target_var(expr, v):
                return k
        return None

    def _candidate_from_divisor(
        self, node: ast.AST, candidates: List[str]
    ) -> Optional[str]:
        """
        If *node* is ``T.ceildiv(axis_expr, param)`` with ``param`` (or an alias
        of ``param``) in *candidates*, return the **canonical** candidate name.
        """
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in self.CEILDIV_ATTRS
            and len(node.args) == 2
        ):
            divisor = node.args[1]
            if isinstance(divisor, ast.Name):
                canonical = self._canonical(divisor.id)
                if canonical in candidates:
                    return canonical
        # floor-div shorthand: (axis + param - 1) // param
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Name)
        ):
            canonical = self._canonical(node.right.id)
            if canonical in candidates:
                return canonical
        return None


# ---------------------------------------------------------------------------
# Split-axes parser
# ---------------------------------------------------------------------------

class SplitAxesParser(AxesKeyParser):
    """
    Extracts the *split-axis* parameters from a TileLang kernel generator.

    A split axis parameter (``block_M``, ``block_N``, …) divides a problem
    dimension into grid tiles.  The canonical pattern is::

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                      threads=128) as (bx, by):

    This parser locates every positional argument of ``T.Kernel`` that looks
    like ``T.ceildiv(axis_size, block_param)`` and records the mapping.

    Grid variables computed before the ``T.Kernel`` call are also supported::

        grid_m = T.ceildiv(M, block_M)
        with T.Kernel(grid_m, T.ceildiv(N, block_N), ...) as (bx, by):

    Note
    ----
    1. Split-axis parameters **must** appear as the divisor in ``T.ceildiv`` (or
       equivalent) inside a ``T.Kernel`` grid argument.
    2. Only parameters in *candidates_params* are considered.
    3. A parameter already identified as a tiling-only (grid-stride) param is
       excluded from split axes.
    """

    def __init__(
        self,
        func_ast: ast.AST,
        keys: Dict[str, str],
        candidates_params: List[str],
    ):
        """
        :param func_ast:         AST of the kernel generator function.
        :param keys:             ``{axis_name: size_variable_name}``.
        :param candidates_params: Parameter names that the user did **not** supply
                                  when calling the kernel (auto-tune candidates).
        """
        super().__init__(func_ast, keys)
        self.split_axes: Dict[str, str] = {}
        # axis_name -> index in T.Kernel grid (0-based)
        self.axis_grid_dims: Dict[str, int] = {}
        self.candidates_params = candidates_params

    def parse(self) -> Dict[str, str]:
        super().parse()
        # Fallback: when T.Kernel args don't directly encode ceildiv (e.g. the
        # Ascend multi-kernel / grid-stride pattern where T.Kernel receives a
        # constant physical-kernel count), scan the function body for index-
        # decomposition assignments that still reveal the mapping.
        if len(self.split_axes) < len(self.candidates_params):
            self._scan_body_for_split_hints()
        return self.split_axes

    # ------------------------------------------------------------------
    # Visitor
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        if _is_t_call(node, "Kernel"):
            self._process_kernel_call(node)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_kernel_call(self, node: ast.Call):
        """Inspect each positional grid argument of ``T.Kernel(...)``."""
        # Keyword args (threads=, …) are not grid dimensions.
        for dim_idx, arg in enumerate(node.args):
            self._try_extract_split_axis(arg, dim_idx)

    def _try_extract_split_axis(self, arg: ast.AST, dim_idx: int):
        # ---- product form: T.ceildiv(M, block_M) * T.ceildiv(N, block_N) ----
        # Ascend NPU kernels often flatten the 2-D grid into a single linearised
        # dimension.  Recurse into both operands so each ceildiv is found.
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
            self._try_extract_split_axis(arg.left, dim_idx)
            self._try_extract_split_axis(arg.right, dim_idx)
            return

        # ---- direct T.ceildiv(axis, block_param) ----
        param = self._candidate_from_divisor(arg, self.candidates_params)
        if param is not None:
            axis_expr = arg.args[0] if isinstance(arg, ast.Call) else arg.left
            axis = self._axis_from_expr(axis_expr)
            if axis and param not in self.split_axes.values():
                self.split_axes[axis] = param
                self.axis_grid_dims[axis] = dim_idx
            return

        # ---- variable reference: grid_m = T.ceildiv(M, block_M) ----
        if isinstance(arg, ast.Name):
            var_name = arg.id
            param = self._find_split_param_for_assigned_var(var_name)
            if param is not None and param not in self.split_axes.values():
                axis = self.get_axis(param)
                if axis:
                    self.split_axes[axis] = param
                    self.axis_grid_dims[axis] = dim_idx

    def _scan_body_for_split_hints(self):
        """
        Fallback scan for kernels where ``T.Kernel`` receives a constant
        grid size (e.g. a physical-kernel count) instead of ``T.ceildiv``
        expressions.

        Recognises two patterns anywhere in the function body:

        **Pattern A – div/mod by ceildiv** (explicit 2-D index decomposition)::

            by = cid // T.ceildiv(N, block_N)   →  block_N splits N
            bx = cid %  T.ceildiv(N, block_N)   →  block_N splits N

        **Pattern B – simple floor-div tile count**::

            num_logical = (N // block_N) * (M // block_M)
            #              ^^^^^^^^^^^                       block_N splits N
            #                            ^^^^^^^^^^^        block_M splits M

        Pattern A is preferred; Pattern B is a secondary signal used only
        when Pattern A has not yet identified all candidate parameters.
        """
        # ── Pass 1: Pattern A – div / mod by T.ceildiv(..., block_param) ──
        for node in ast.walk(self.func_ast):
            if not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, (ast.FloorDiv, ast.Mod))
            ):
                continue
            rhs = node.right
            param = self._candidate_from_divisor(rhs, self.candidates_params)
            if param is None or param in self.split_axes.values():
                continue
            # rhs is T.ceildiv(axis_expr, param) – extract axis from numerator
            if isinstance(rhs, ast.Call) and len(rhs.args) == 2:
                axis = self._axis_from_expr(rhs.args[0])
            elif isinstance(rhs, ast.BinOp):   # floor-div shorthand form
                axis = self._axis_from_expr(rhs.left)
            else:
                continue
            if axis:
                self.split_axes[axis] = param

        # ── Pass 2: Pattern B – axis_expr // block_param (tile-count form) ──
        # Only fires for params still unresolved after Pass 1.
        remaining = [p for p in self.candidates_params if p not in self.split_axes.values()]
        if not remaining:
            return
        for node in ast.walk(self.func_ast):
            if not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.FloorDiv)
                and isinstance(node.right, ast.Name)
                and node.right.id in remaining
            ):
                continue
            param = node.right.id
            if param in self.split_axes.values():
                continue
            axis = self._axis_from_expr(node.left)
            if axis:
                self.split_axes[axis] = param

    def _find_split_param_for_assigned_var(self, var: str) -> Optional[str]:
        """
        Walk assignments to find ``var = T.ceildiv(axis_expr, block_param)``
        and return ``block_param`` if it is a candidate.
        """
        for node in ast.walk(self.func_ast):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == var
            ):
                continue
            param = self._candidate_from_divisor(node.value, self.candidates_params)
            if param is not None:
                return param
        return None


# ---------------------------------------------------------------------------
# Tiling-axes parser
# ---------------------------------------------------------------------------

class TilingAxesParser(AxesKeyParser):
    """
    Extracts the *tiling-axis* parameters from a TileLang kernel generator.

    Tiling axis parameters control the tile size used inside the kernel body
    (block-level tiling).  They appear in two main patterns:

    1. **Pipelined reduction loop**::

           for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):

    2. **Plain range loop**::

           for k in range(T.ceildiv(K, block_K)):

    3. **Shared-memory allocation shape** (secondary confirmation)::

           A_shared = T.alloc_shared((block_M, block_K), dtype)

    Note
    ----
    1. A tiling parameter must appear as the divisor in ``T.ceildiv`` inside a
       loop range (``T.Pipelined`` or ``range``).
    2. Only parameters in *candidates_params* are considered.
    3. Parameters already captured as split axes are still recorded here if
       they also appear in tiling loops (one parameter can serve both roles in
       some kernels, though this is uncommon).
    """

    def __init__(
        self,
        func_ast: ast.AST,
        keys: Dict[str, str],
        candidates_params: List[str],
    ):
        """
        :param func_ast:         AST of the kernel generator function.
        :param keys:             ``{axis_name: size_variable_name}``.
        :param candidates_params: Auto-tune candidate parameter names.
        """
        super().__init__(func_ast, keys)
        self.tiling_axes: Dict[str, str] = {}
        self.candidates_params = candidates_params
        # Params confirmed as loop-step candidates (used in range step or alloc)
        self._loop_candidates: List[str] = []

    def parse(self) -> Dict[str, str]:
        super().parse()
        return self.tiling_axes

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_For(self, node: ast.For):
        if isinstance(node.iter, ast.Call):
            self._process_loop_iter(node.iter)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Detect ``T.alloc_shared / T.alloc_fragment`` to confirm tiling params."""
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            call = node.value
            if _is_t_call(call, "alloc_shared") or _is_t_call(call, "alloc_fragment"):
                self._process_alloc_shape(call)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_loop_iter(self, call: ast.Call):
        """
        Handle ``T.Pipelined(range_expr, ...)`` and ``range(range_expr)`` iters.
        """
        is_pipelined = _is_t_call(call, "Pipelined")
        is_range = (
            isinstance(call.func, ast.Name) and call.func.id in ("range", "tl_range")
        )
        if not (is_pipelined or is_range):
            return

        # The loop bound is always the first positional argument.
        if not call.args:
            return
        range_expr = call.args[0]
        self._extract_tiling_axis(range_expr)

    def _extract_tiling_axis(self, expr: ast.AST):
        """
        Map ``T.ceildiv(axis_expr, block_param)`` → ``{axis: block_param}``.
        """
        param = self._candidate_from_divisor(expr, self.candidates_params)
        if param is None:
            return
        # Determine axis from the numerator
        if isinstance(expr, ast.Call) and len(expr.args) == 2:
            axis_expr = expr.args[0]
        elif isinstance(expr, ast.BinOp):
            axis_expr = expr.left  # (axis + param - 1) // param
        else:
            return
        axis = self._axis_from_expr(axis_expr)
        if axis and axis not in self.tiling_axes:
            self.tiling_axes[axis] = param

    def _process_alloc_shape(self, call: ast.Call):
        """
        Inspect the shape argument of ``T.alloc_shared`` / ``T.alloc_fragment``
        (tuple or list literal) to add secondary confirmation of tiling params.
        """
        if not call.args:
            return
        shape_arg = call.args[0]
        if not isinstance(shape_arg, (ast.Tuple, ast.List)):
            return
        for elt in shape_arg.elts:
            if isinstance(elt, ast.Name) and elt.id in self.candidates_params:
                if elt.id not in self._loop_candidates:
                    self._loop_candidates.append(elt.id)


# ---------------------------------------------------------------------------
# Reduction-axes parser
# ---------------------------------------------------------------------------

class ReductionAxesParser(AxesKeyParser):
    """
    Extracts the *reduction-axis* name(s) from a TileLang kernel generator.

    In TileLang the reduction axis is iterated by either:

    * ``T.Pipelined(T.ceildiv(K, block_K), num_stages=…)`` – the canonical
      software-pipelined reduction loop.
    * A plain ``for k in range(T.ceildiv(K, block_K)):`` loop that contains a
      ``T.gemm`` or ``T.reduce`` call in its body.

    The parser records every axis whose size variable appears as the numerator
    in a pipelined-loop bound.  Axes already recorded are deduplicated.

    Note
    ----
    1. ``T.Pipelined`` loops are *always* treated as reduction loops.
    2. A plain ``range`` loop is only treated as a reduction loop when its body
       contains a recognised reduction primitive (``T.gemm``, ``T.reduce``, …).
    3. The identified axes are taken directly from *keys*; no candidate filter
       is applied (unlike split/tiling parsers).
    """

    #: TileLang primitives that imply a reduction.
    REDUCTION_CALLS = frozenset(("gemm", "reduce", "dot", "matmul"))

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: AST of the kernel generator function.
        :param keys:     ``{axis_name: size_variable_name}``.
        """
        super().__init__(func_ast, keys)
        self.reduction_axes: List[str] = []
        # var_name → list of shape AST elements from T.alloc_shared/fragment
        self._alloc_shapes: Dict[str, List[ast.AST]] = {}

    def parse(self) -> List[str]:
        super().parse()
        return self.reduction_axes

    # ------------------------------------------------------------------
    # Visitors
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign):
        """
        Record ``T.alloc_shared`` / ``T.alloc_fragment`` shape tuples so that
        ``_process_t_reduce`` can map a ``dims=`` index to a key-axis name.
        """
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            call = node.value
            if _is_t_call(call, "alloc_shared") or _is_t_call(call, "alloc_fragment"):
                if call.args and isinstance(call.args[0], (ast.Tuple, ast.List)):
                    self._alloc_shapes[node.targets[0].id] = call.args[0].elts
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Detect direct reduction-primitive calls:

        * ``T.reduce(src, dst, dims=k, …)``   – uses ``dims`` kwarg
        * ``T.gemm(A, B, C)``                  – always a K-axis reduction
        """
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr in self.REDUCTION_CALLS
        ):
            self.generic_visit(node)
            return

        if node.func.attr == "reduce":
            self._process_t_reduce(node)
        # T.gemm / T.dot / T.matmul: reduction axis is implicit (K dim).
        # These are handled through T.Pipelined loop detection in visit_For;
        # no additional action needed here.
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if not isinstance(node.iter, ast.Call):
            self.generic_visit(node)
            return

        iter_call = node.iter
        is_pipelined = _is_t_call(iter_call, "Pipelined")
        is_range = (
            isinstance(iter_call.func, ast.Name)
            and iter_call.func.id in ("range", "tl_range")
        )

        if is_pipelined:
            # All T.Pipelined loops are reduction loops.
            axis = self._axis_from_loop_bound(iter_call)
            self._record(axis)

        elif is_range:
            # Only record if the loop body contains a reduction primitive.
            if self._body_has_reduction(node.body):
                axis = self._axis_from_loop_bound(iter_call)
                self._record(axis)

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _axis_from_loop_bound(self, iter_call: ast.Call) -> Optional[str]:
        """Return the axis name encoded in the loop-bound argument."""
        if not iter_call.args:
            return None
        bound_expr = iter_call.args[0]

        # Direct T.ceildiv(axis_expr, block_param)
        if (
            isinstance(bound_expr, ast.Call)
            and isinstance(bound_expr.func, ast.Attribute)
            and bound_expr.func.attr in self.CEILDIV_ATTRS
            and len(bound_expr.args) == 2
        ):
            return self._axis_from_expr(bound_expr.args[0])

        # Variable reference: num_k = T.ceildiv(K, block_K)
        if isinstance(bound_expr, ast.Name):
            return self._trace_var_to_axis(bound_expr.id)

        return None

    def _trace_var_to_axis(self, var: str) -> Optional[str]:
        """Follow an assignment to recover the axis encoded in its rhs."""
        for node in ast.walk(self.func_ast):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == var
            ):
                continue
            rhs = node.value
            if (
                isinstance(rhs, ast.Call)
                and isinstance(rhs.func, ast.Attribute)
                and rhs.func.attr in self.CEILDIV_ATTRS
                and len(rhs.args) == 2
            ):
                return self._axis_from_expr(rhs.args[0])
        return None

    def _body_has_reduction(self, stmts: list) -> bool:
        """Check whether *stmts* contain a T.gemm / T.reduce / … call."""
        for stmt in stmts:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if (
                        isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "T"
                        and child.func.attr in self.REDUCTION_CALLS
                    ):
                        return True
        return False

    def _process_t_reduce(self, node: ast.Call):
        """
        Handle ``T.reduce(src, dst, dims=k, reduce_mode=…, clear=…)``.

        Strategy:
        1. Extract the ``dims`` kwarg (or 3rd positional arg) as an integer.
        2. Look up the source tensor's recorded alloc shape.
        3. The shape element at index ``dims`` identifies the axis being reduced.

        Example::

            A_shared = T.alloc_shared((block_M, N), "float16")  # shape[1] = N
            T.reduce(A_shared, B_local, dims=1, …)              # reduces axis N
        """
        # -- extract dims value --
        dims_val = self._kwarg_int(node, "dims")
        if dims_val is None and len(node.args) >= 3:
            dims_val = self._const_int(node.args[2])
        if dims_val is None:
            return

        # -- get source tensor name --
        if not node.args or not isinstance(node.args[0], ast.Name):
            return
        src_name = node.args[0].id

        # -- look up its shape --
        shape_elts = self._alloc_shapes.get(src_name)
        if not shape_elts or dims_val >= len(shape_elts):
            return

        # -- map shape element at dims_val to an axis --
        axis = self._axis_from_shape_elt(shape_elts[dims_val])
        self._record(axis)

    def _axis_from_shape_elt(self, elt: ast.AST) -> Optional[str]:
        """
        Map a single shape-tuple element to an axis name.

        Two cases:
        * ``block_K`` (block param) → find via ``T.ceildiv(K, block_K)``
        * ``N``       (raw key var) → look up directly in ``keys``
        """
        if not isinstance(elt, ast.Name):
            return self._axis_from_expr(elt)
        var = elt.id
        # Case 1: var is a block param → trace through ceildiv
        axis = self.get_axis(var)
        if axis:
            return axis
        # Case 2: var is the raw problem-size variable itself (e.g. shape = (block_M, N))
        for k, v in self.keys.items():
            if var == v:
                return k
        return None

    @staticmethod
    def _kwarg_int(node: ast.Call, name: str) -> Optional[int]:
        for kw in node.keywords:
            if kw.arg == name:
                return ReductionAxesParser._const_int(kw.value)
        return None

    @staticmethod
    def _const_int(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    def _record(self, axis: Optional[str]):
        if axis and axis not in self.reduction_axes:
            self.reduction_axes.append(axis)


# ---------------------------------------------------------------------------
# Low-dims axes parser
# ---------------------------------------------------------------------------

class LowDimsAxesParser(AxesKeyParser):
    """
    Identifies the *innermost* (lowest) tiling dimension axes in a TileLang
    kernel generator.

    In TileLang, the innermost axes are those whose block parameters are used
    as the last (innermost) dimension in shared-memory or fragment allocations::

        A_shared = T.alloc_shared((block_M, block_K), dtype)
        # block_K is the innermost (low) dimension → K-axis

    The parser additionally falls back to checking which tiling-block parameters
    appear in the *last* position of any slice used to copy into shared memory::

        T.copy(A[bx * block_M : (bx+1)*block_M, k*block_K : (k+1)*block_K],
               A_shared)

    Note
    ----
    1. A block parameter must appear as the *last* shape dimension of a
       ``T.alloc_shared`` or ``T.alloc_fragment`` call to be identified here.
    2. Without an allocation or copy pattern, the axis cannot be confirmed.
    """

    def __init__(self, func_ast: ast.AST, keys: Dict[str, str]):
        """
        :param func_ast: AST of the kernel generator function.
        :param keys:     ``{axis_name: size_variable_name}``.
        """
        super().__init__(func_ast, keys)
        self.low_dims_axes: List[str] = []
        self._checked_alloc_vars: List[str] = []

    def parse(self) -> List[str]:
        super().parse()
        return self.low_dims_axes

    # ------------------------------------------------------------------
    # Visitor
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign):
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            self.generic_visit(node)
            return

        call = node.value
        if _is_t_call(call, "alloc_shared") or _is_t_call(call, "alloc_fragment"):
            self._process_alloc(call)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_alloc(self, call: ast.Call):
        """
        Inspect the shape argument of ``T.alloc_shared((dim0, dim1, …), dtype)``
        or ``T.alloc_shared([dim0, dim1, …], dtype)`` (tuple or list literal).
        The *last* element identifies the innermost / low-dimensional axis.
        """
        if not call.args:
            return
        shape_arg = call.args[0]
        # Accept both tuple (block_M, block_N) and list [block_M, block_N] literals
        if isinstance(shape_arg, (ast.Tuple, ast.List)) and shape_arg.elts:
            last_elt = shape_arg.elts[-1]
            self._try_record_low_dim(last_elt)

    def _try_record_low_dim(self, expr: ast.AST):
        """
        Map an innermost-dimension expression to an axis and record it.

        Two cases:
        * ``block_N`` (block param) → find via ``T.ceildiv(N, block_N)``
        * ``N``       (raw key var) → look up directly in ``keys``
          This happens when the full axis fits in shared memory without tiling,
          e.g. ``T.alloc_shared((block_M, N), dtype)`` in a reduce-sum kernel.
        """
        axis = None

        if isinstance(expr, ast.Name):
            var = expr.id
            # Case 1: block param → trace through ceildiv pattern
            axis = self.get_axis(var)
            # Case 2: raw problem-size variable used directly as a dim
            if axis is None:
                for k, v in self.keys.items():
                    if var == v:
                        axis = k
                        break

        elif isinstance(expr, ast.Call):
            # e.g. T.ceildiv(N, block_N) used directly as a shape element
            for k, v in self.keys.items():
                if self.contains_target_var(expr, v):
                    axis = k
                    break

        if axis and axis not in self.low_dims_axes:
            self.low_dims_axes.append(axis)


# ---------------------------------------------------------------------------
# Buffer (pointer) count parser
# ---------------------------------------------------------------------------

class BufferNumsParser(AutoParser):
    """
    Counts the number of tensor-buffer parameters in a TileLang kernel.

    In TileLang the inner ``@T.prim_func`` function declares its tensor
    arguments with ``T.Buffer(shape, dtype)`` annotations::

        @T.prim_func
        def main(
            A: T.Buffer((M, K), "float16"),
            B: T.Buffer((K, N), "float16"),
            C: T.Buffer((M, N), "float32"),
        ):

    These are the pointer-equivalent parameters.  All other arguments (plain
    integer scalars) are excluded.

    Note
    ----
    1. Only parameters of functions decorated with ``@T.prim_func`` are counted.
    2. Parameters whose names match values in *keys* (i.e. problem-size
       variables that happen to share a name with a buffer) are excluded.
    3. A warning is printed for any *miss_params* not declared as scalar/constexpr
       (they should be ``T.constexpr``-equivalent scalars).
    """

    def __init__(
        self,
        func_ast: ast.AST,
        keys: Dict[str, str],
        miss_params: List[str],
    ):
        """
        :param func_ast:    AST of the kernel generator function.
        :param keys:        ``{axis_name: size_variable_name}`` (excluded from count).
        :param miss_params: Parameters the user did not supply (should be scalars).
        """
        super().__init__(func_ast)
        self.keys = keys
        self.miss_params = miss_params
        self.buf_nums: int = 0
        self.buf_params: List[str] = []
        self.scalar_params: List[str] = []
        self._checked_vars: List[str] = []

    def parse(self) -> Tuple[int, List[str]]:
        super().parse()
        return self.buf_nums, self.buf_params

    # ------------------------------------------------------------------
    # Visitor
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Inspect every ``@T.prim_func``-decorated inner function."""
        if self._is_prim_func(node):
            self._parse_prim_func_params(node)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_prim_func(node: ast.FunctionDef) -> bool:
        """Return True when the function carries a ``@T.prim_func`` decorator."""
        for dec in node.decorator_list:
            if _is_t_attr(dec, "prim_func"):
                return True
            # @T.prim_func()  – call form
            if isinstance(dec, ast.Call) and _is_t_attr(dec.func, "prim_func"):
                return True
        return False

    def _parse_prim_func_params(self, node: ast.FunctionDef):
        if not isinstance(node.args, ast.arguments):
            return

        for arg in node.args.args:
            if not isinstance(arg, ast.arg):
                continue
            if arg.annotation is None:
                # No annotation – treat as scalar.
                self.scalar_params.append(arg.arg)
                continue

            if self._is_buffer_annotation(arg.annotation):
                # Buffer param – excluded if it's a key (size variable).
                if arg.arg not in self.keys.values():
                    self.buf_params.append(arg.arg)
                    self.buf_nums += 1
            else:
                self.scalar_params.append(arg.arg)

        # Warn about miss_params that look like they should be scalars but
        # were not declared at all in the prim_func (they should be constexpr
        # equivalents in the outer generator signature).
        all_prim_params = {a.arg for a in node.args.args if isinstance(a, ast.arg)}
        for mp in self.miss_params:
            if mp not in all_prim_params and mp not in self._checked_vars:
                print(
                    f"[WARNING] The parameter '{mp}' is not declared in the "
                    f"@T.prim_func signature. "
                    f"Ensure it is a scalar/constexpr in the generator function."
                )
                self._checked_vars.append(mp)

    @staticmethod
    def _is_buffer_annotation(annotation: ast.AST) -> bool:
        """
        Return True for any of::

            T.Buffer(shape, dtype)   # ast.Call
            T.Tensor(shape, dtype)   # ast.Call  (alias used in some TileLang versions)
            T.Buffer                 # ast.Attribute (bare reference)
            T.Tensor                 # ast.Attribute (bare reference)
        """
        #: Both names are accepted; T.Tensor is an alias for T.Buffer in TileLang.
        BUFFER_NAMES = ("Buffer", "Tensor")

        # T.Buffer(...) / T.Tensor(...) call
        if isinstance(annotation, ast.Call) and isinstance(annotation.func, ast.Attribute):
            if (
                isinstance(annotation.func.value, ast.Name)
                and annotation.func.value.id == "T"
                and annotation.func.attr in BUFFER_NAMES
            ):
                return True
        # Bare T.Buffer / T.Tensor attribute reference
        if _is_t_attr(annotation, "Buffer") or _is_t_attr(annotation, "Tensor"):
            return True
        # Unqualified Buffer(...) / Tensor(...)
        if isinstance(annotation, ast.Call) and isinstance(annotation.func, ast.Name):
            return annotation.func.id in BUFFER_NAMES
        if isinstance(annotation, ast.Name):
            return annotation.id in BUFFER_NAMES
        return False

    def is_in_buffer_access(self, var: str) -> bool:
        """
        Return True if *var* directly participates in a ``T.copy``, ``T.load``,
        ``T.store``, or ``T.gemm`` call as the first argument (address operand).
        """
        for node in ast.walk(self.func_ast):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "T"
                and func.attr in ("copy", "load", "store", "gemm")
            ):
                continue
            # Check first argument only (the address/buffer operand).
            if node.args and self.contains_target_var(node.args[0], var):
                return True
        return False
