"""
TileLang Axis Semantic Parser  (VV-style)
==========================================

Automatically infers split / tiling / reduction / low-dim / extent information
from a TileLang kernel generator function without requiring user-supplied
axis-to-variable-name mappings (``keys``).

Design mirrors the Triton VV parser architecture
-------------------------------------------------
* **Evidence accumulation** – every pattern that reveals axis information adds
  a candidate with a confidence score instead of a hard match/no-match decision.
* **Multi-source evidence** – ``T.Kernel``, ``T.Pipelined``, ``T.alloc_shared``,
  ``T.reduce``, body index decompositions, and buffer annotations all contribute.
* **Best-candidate selection** – conflicting signals are resolved by ranking
  candidates by confidence.
* **Automatic axis naming** – no ``keys`` dict needed.  Axis names are derived
  from the block-parameter names (``block_M`` → ``"M"``) or from the raw size
  variable when no block param exists (``dim`` → ``"D_dim"``).  Reduction axes
  are prefixed with ``"r"`` (``rK``).

Confidence reference
--------------------
+-----------------------------------------------+-----------+
| Source                                        | Confidence|
+===============================================+===========+
| T.Kernel(T.ceildiv(M, block_M))               | 1.00      |
| T.Pipelined(T.ceildiv(K, block_K))            | 1.00      |
| T.reduce dims= + alloc shape lookup           | 1.00      |
| T.Kernel product form ceildiv                 | 0.90      |
| buffer annotation T.Tensor((M,N), dtype)      | 0.90      |
| T.copy size= argument                         | 0.85      |
| body decomposition  cid // T.ceildiv(N, bN)   | 0.80      |
| T.alloc_shared shape element                  | 0.75      |
| floor-div  N // block_N                       | 0.70      |
+-----------------------------------------------+-----------+

Drop-in replacement for the five autoparser classes
----------------------------------------------------
    result = parse_tilelang_axes(func_ast)

    result.split_params    ↔  SplitAxesParser.parse()
    result.tiling_params   ↔  TilingAxesParser.parse()
    result.reduction_axes  ↔  ReductionAxesParser.parse()
    result.low_dim_axes    ↔  LowDimsAxesParser.parse()
    result.buffer_params,
    result.buf_count       ↔  BufferNumsParser.parse()
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CEILDIV_ATTRS   = ("ceildiv", "cdiv")
_ALLOC_CALLS     = ("alloc_shared", "alloc_fragment")
_BUFFER_ATTRS    = ("Buffer", "Tensor")
_REDUCE_CALLS    = ("reduce",)
_GEMM_CALLS      = ("gemm", "dot", "matmul")
_BLOCK_PREFIXES  = ("block_", "BLOCK_", "Block_")

# Confidence levels
_CONF_KERNEL_CEILDIV   = 1.00   # T.Kernel(T.ceildiv(M, block_M))
_CONF_PIPELINED        = 1.00   # T.Pipelined(T.ceildiv(K, block_K))
_CONF_REDUCE_DIMS      = 1.00   # T.reduce + alloc shape
_CONF_KERNEL_PRODUCT   = 0.90   # T.ceildiv inside product arg
_CONF_BUFFER_ANNOT     = 0.90   # T.Tensor((M, N), dtype)
_CONF_COPY_SIZE        = 0.85   # T.copy(..., size=[bM, N])
_CONF_BODY_DECOMP      = 0.80   # cid // T.ceildiv(N, block_N)
_CONF_ALLOC_SHAPE      = 0.75   # T.alloc_shared((block_M, K))
_CONF_FLOOR_DIV        = 0.70   # N // block_N


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TLAxisExtent:
    """Size / length of an axis as inferred from source patterns."""
    expr:       Optional[str]   # e.g. ``"M"``, ``"seq_len"``, ``"512"``
    source:     str             # pattern that produced this evidence
    confidence: float


@dataclass(frozen=True)
class TLAxisSplit:
    """Grid-split (T.Kernel tile) info for an axis."""
    param:      Optional[str]   # e.g. ``"block_M"``
    grid_dim:   Optional[int]   # position in T.Kernel arg list
    source:     str
    confidence: float


@dataclass(frozen=True)
class TLAxisTiling:
    """Inner-loop tiling info for an axis."""
    param:      Optional[str]   # e.g. ``"block_K"``
    loop_var:   Optional[str]   # loop variable name, e.g. ``"k"``
    source:     str
    confidence: float


@dataclass
class TLAxisInfo:
    """Complete semantic information for one axis."""
    axis_name:    str
    extent:       TLAxisExtent
    split:        TLAxisSplit
    tiling:       TLAxisTiling
    is_low_dim:   bool
    is_reduction: bool
    diagnostics:  List[str] = field(default_factory=list)


@dataclass
class TLAxisParseResult:
    """
    Complete result of TileLang VV-style axis semantic parsing.

    Attributes
    ----------
    axes:
        Per-axis semantic information keyed by axis name.
    split_params:
        ``{axis_name: block_param}`` – drop-in for ``SplitAxesParser.parse()``.
    tiling_params:
        ``{axis_name: block_param}`` – drop-in for ``TilingAxesParser.parse()``.
    reduction_axes:
        List of axis names – drop-in for ``ReductionAxesParser.parse()``.
    low_dim_axes:
        List of axis names – drop-in for ``LowDimsAxesParser.parse()``.
    buffer_params:
        List of ``@T.prim_func`` buffer parameter names.
    buf_count:
        Length of ``buffer_params`` – drop-in for ``BufferNumsParser.parse()``.
    inferred_keys:
        ``{axis_name: size_expr}`` – equivalent to the ``keys`` dict the old
        parsers required the user to supply.
    status:
        ``"ok"`` | ``"partial"`` | ``"failed"``
    diagnostics:
        Human-readable notes about what was resolved and what was ambiguous.
    """
    axes:           Dict[str, TLAxisInfo]
    split_params:   Dict[str, str]
    tiling_params:  Dict[str, str]
    low_dim_axes:   List[str]
    reduction_axes: List[str]
    buffer_params:  List[str]
    buf_count:      int
    inferred_keys:  Dict[str, str]
    status:         str
    diagnostics:    List[str]


# ---------------------------------------------------------------------------
# Internal evidence accumulator
# ---------------------------------------------------------------------------

@dataclass
class _AxisEvidence:
    """
    Raw evidence for a single axis, accumulated from multiple source patterns.

    Keyed either by a canonical block-param name (``"block_M"``) or by a
    fixed-size sentinel (``"__fixed__dim"`` for a raw ``dim`` variable).
    """
    canonical_param: Optional[str]              = None  # block_M / block_K / …
    canonical_size:  Optional[str]              = None  # M / seq_len / dim / …

    # (size_expr, source_tag, confidence)
    extent_candidates:  List[Tuple[str, str, float]]                  = field(default_factory=list)
    # (param, grid_dim_or_None, source_tag, confidence)
    split_candidates:   List[Tuple[str, Optional[int], str, float]]   = field(default_factory=list)
    # (param, loop_var_or_None, source_tag, confidence)
    tiling_candidates:  List[Tuple[str, Optional[str], str, float]]   = field(default_factory=list)

    is_low_dim:   bool       = False
    is_reduction: bool       = False
    diagnostics:  List[str]  = field(default_factory=list)

    # ---- helpers -----------------------------------------------------------

    def add_extent(self, expr: str, source: str, conf: float) -> None:
        if not expr:
            return
        if not any(e == expr and s == source for e, s, _ in self.extent_candidates):
            self.extent_candidates.append((expr, source, conf))

    def add_split(self, param: str, grid_dim: Optional[int],
                  source: str, conf: float) -> None:
        if not any(p == param and d == grid_dim and s == source
                   for p, d, s, _ in self.split_candidates):
            self.split_candidates.append((param, grid_dim, source, conf))

    def add_tiling(self, param: str, loop_var: Optional[str],
                   source: str, conf: float) -> None:
        if not any(p == param and v == loop_var and s == source
                   for p, v, s, _ in self.tiling_candidates):
            self.tiling_candidates.append((param, loop_var, source, conf))


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------

def _is_t_call(node: ast.AST, attr: str) -> bool:
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr == attr)


def _is_t_attr(node: ast.AST, attr: str) -> bool:
    return (isinstance(node, ast.Attribute)
            and node.attr == attr
            and isinstance(node.value, ast.Name)
            and node.value.id == "T")


def _ast_str(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        return "<expr>"


def _elts_of(shape_arg: ast.AST) -> List[ast.AST]:
    """Return elements of a Tuple or List AST node (or empty list)."""
    if isinstance(shape_arg, (ast.Tuple, ast.List)):
        return list(shape_arg.elts)
    return []


def _kwarg_int(call: ast.Call, name: str) -> Optional[int]:
    for kw in call.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                return kw.value.value
    return None


def _is_buffer_annotation(ann: ast.AST) -> bool:
    if isinstance(ann, ast.Call):
        f = ann.func
        if isinstance(f, ast.Attribute) and f.attr in _BUFFER_ATTRS:
            return True
        if isinstance(f, ast.Name) and f.id in _BUFFER_ATTRS:
            return True
    if isinstance(ann, ast.Attribute) and ann.attr in _BUFFER_ATTRS:
        return True
    if isinstance(ann, ast.Name) and ann.id in _BUFFER_ATTRS:
        return True
    return False


def _is_prim_func_decorator(func_node: ast.FunctionDef) -> bool:
    for dec in func_node.decorator_list:
        if _is_t_attr(dec, "prim_func"):
            return True
        if isinstance(dec, ast.Call) and _is_t_attr(dec.func, "prim_func"):
            return True
    return False


def _collect_generator_params(func_ast: ast.AST) -> set:
    """
    Return parameter names of the outermost non-``@T.prim_func`` function —
    i.e. the user-facing kernel generator.

    These are the only names that can be tunable block params.  Local
    variables such as ``num_physical = 48`` are excluded automatically.
    """
    for node in ast.walk(func_ast):
        if isinstance(node, ast.FunctionDef) and not _is_prim_func_decorator(node):
            return {arg.arg for arg in node.args.args}
    return set()


def _strip_block_prefix(name: str) -> Optional[str]:
    """
    ``block_M`` → ``"M"``,  ``BLOCK_K`` → ``"K"``.
    Returns *None* if no recognised prefix is found.
    """
    for prefix in _BLOCK_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
    return None


# ---------------------------------------------------------------------------
# Pre-pass: alias map
# ---------------------------------------------------------------------------

def _build_alias_map(func_ast: ast.AST) -> Dict[str, str]:
    """
    Collect all ``alias = name`` assignments and resolve chains transitively.

    Example::

        block_m = block_M   →  {"block_m": "block_M"}
        bm      = block_m   →  {"bm":      "block_M", "block_m": "block_M"}
    """
    direct: Dict[str, str] = {}
    for node in ast.walk(func_ast):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
                and node.targets[0].id != node.value.id):
            direct[node.targets[0].id] = node.value.id

    def _resolve(name: str, seen: set) -> str:
        if name in seen:
            return name
        seen.add(name)
        t = direct.get(name)
        return _resolve(t, seen) if t else name

    return {alias: _resolve(alias, set()) for alias in direct}


def _canon(name: str, alias_map: Dict[str, str]) -> str:
    return alias_map.get(name, name)


# ---------------------------------------------------------------------------
# Evidence collectors
# ---------------------------------------------------------------------------

def _process_ceildiv_node(
    node: ast.AST,
    grid_dim: int,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    source: str,
    base_conf: float,
    is_tiling: bool,
    loop_var: Optional[str],
    generator_params: set,
) -> None:
    """
    Attempt to extract split or tiling evidence from *node*.

    Only creates evidence entries for params that belong to the outer
    generator function (``generator_params``), filtering out local
    constants such as ``num_physical = 48``.
    """
    # ── product: recurse into both operands ──────────────────────────────────
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        _process_ceildiv_node(node.left,  grid_dim, alias_map, ev_map,
                               source, base_conf * 0.90, is_tiling, loop_var, generator_params)
        _process_ceildiv_node(node.right, grid_dim, alias_map, ev_map,
                               source, base_conf * 0.90, is_tiling, loop_var, generator_params)
        return

    # ── T.ceildiv(size_expr, param) ──────────────────────────────────────────
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _CEILDIV_ATTRS
            and len(node.args) == 2):
        size_node, div_node = node.args
        if isinstance(div_node, ast.Name):
            param = _canon(div_node.id, alias_map)
            if param not in generator_params:
                return
            size_str = _ast_str(size_node)
            ev = ev_map.setdefault(param, _AxisEvidence(canonical_param=param))
            ev.add_extent(size_str, source, base_conf)
            if ev.canonical_size is None:
                ev.canonical_size = size_str
            if is_tiling:
                ev.add_tiling(param, loop_var, source, base_conf)
            else:
                ev.add_split(param, grid_dim, source, base_conf)
        return

    # ── floor-div forms ───────────────────────────────────────────────────────
    if (isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Name)):
        param = _canon(node.right.id, alias_map)
        if param not in generator_params:
            return
        size_str = _ast_str(node.left)
        conf     = min(base_conf, _CONF_FLOOR_DIV)
        ev = ev_map.setdefault(param, _AxisEvidence(canonical_param=param))
        ev.add_extent(size_str, source + ".floordiv", conf)
        if ev.canonical_size is None:
            ev.canonical_size = size_str
        if is_tiling:
            ev.add_tiling(param, loop_var, source + ".floordiv", conf)
        else:
            ev.add_split(param, grid_dim, source + ".floordiv", conf)
        return


# --- T.Kernel -----------------------------------------------------------------

def _collect_kernel_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    generator_params: set,
) -> None:
    """Walk ``T.Kernel(arg0, arg1, …)`` and record split evidence."""
    for node in ast.walk(func_ast):
        if not _is_t_call(node, "Kernel"):
            continue
        for dim_idx, arg in enumerate(node.args):
            _process_ceildiv_node(
                arg, dim_idx, alias_map, ev_map,
                "T.Kernel", _CONF_KERNEL_CEILDIV,
                is_tiling=False, loop_var=None,
                generator_params=generator_params,
            )


def _collect_loop_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    generator_params: set,
) -> None:
    """Walk ``for`` loops and record tiling / reduction evidence."""
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue

        iter_call = node.iter
        loop_var  = node.target.id if isinstance(node.target, ast.Name) else None

        is_pipelined = _is_t_call(iter_call, "Pipelined")
        is_serial    = _is_t_call(iter_call, "serial")
        is_range     = (isinstance(iter_call.func, ast.Name)
                        and iter_call.func.id in ("range", "tl_range"))

        if not (is_pipelined or is_serial or is_range):
            continue
        if not iter_call.args:
            continue

        bound_expr = iter_call.args[0]
        conf       = _CONF_PIPELINED if is_pipelined else 0.85
        tag        = ("T.Pipelined" if is_pipelined
                      else "T.serial" if is_serial else "range")

        before = set(ev_map.keys())
        _process_ceildiv_node(
            bound_expr, 0, alias_map, ev_map,
            tag, conf,
            is_tiling=True, loop_var=loop_var,
            generator_params=generator_params,
        )

        # T.Pipelined → every param registered from this bound is a reduction
        if is_pipelined:
            after = set(ev_map.keys())
            for param in (before | after):
                ev = ev_map.get(param)
                if ev and ev.tiling_candidates and any(
                    s == tag for _, _, s, _ in ev.tiling_candidates
                ):
                    ev.is_reduction = True


# --- T.alloc_shared / T.alloc_fragment ----------------------------------------

def _collect_alloc_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    fixed_ev: Dict[str, _AxisEvidence],
) -> None:
    """
    Walk ``T.alloc_shared`` and ``T.alloc_fragment`` calls.

    For each shape element:
    * If it resolves to a known block param  → add to that param's evidence.
    * If it is a raw size variable            → add to ``fixed_ev`` (axes
      without a tunable block param, e.g. the ``dim`` head-dimension in
      flash-attention).

    The **last** element of every shape marks the innermost (low) dimension.
    """
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (_is_t_call(call, "alloc_shared")
                or _is_t_call(call, "alloc_fragment")):
            continue
        if not call.args:
            continue

        elts = _elts_of(call.args[0])
        if not elts:
            continue

        for dim_idx, elt in enumerate(elts):
            is_last = dim_idx == len(elts) - 1
            if not isinstance(elt, ast.Name):
                continue

            canonical = _canon(elt.id, alias_map)

            if canonical in ev_map:
                # Known block param → add shape evidence
                ev = ev_map[canonical]
                ev.add_extent(elt.id, "alloc_shape", _CONF_ALLOC_SHAPE)
                if is_last:
                    ev.is_low_dim = True
            else:
                # Raw size variable (e.g. K, dim, seq_len)
                ev = fixed_ev.setdefault(canonical, _AxisEvidence(canonical_size=canonical))
                ev.add_extent(canonical, "alloc_shape.fixed", _CONF_ALLOC_SHAPE)
                if is_last:
                    ev.is_low_dim = True


# --- T.copy size= argument ----------------------------------------------------

def _collect_copy_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    fixed_ev: Dict[str, _AxisEvidence],
) -> None:
    """
    Extract extent evidence from ``T.copy(src, dst, size=[block_M, N])``.

    The ``size`` argument lists the actual dimensions being transferred and is
    a reliable source of axis size information.
    """
    for node in ast.walk(func_ast):
        if not _is_t_call(node, "copy"):
            continue
        # size= keyword argument (list or tuple)
        size_node = None
        for kw in node.keywords:
            if kw.arg == "size":
                size_node = kw.value
                break
        if size_node is None:
            # positional: T.copy(src, dst, size_list)
            if len(node.args) >= 3:
                size_node = node.args[2]
        if size_node is None:
            continue

        elts = _elts_of(size_node)
        for dim_idx, elt in enumerate(elts):
            if not isinstance(elt, ast.Name):
                continue
            canonical = _canon(elt.id, alias_map)
            if canonical in ev_map:
                ev_map[canonical].add_extent(elt.id, "T.copy.size", _CONF_COPY_SIZE)
            else:
                ev = fixed_ev.setdefault(canonical, _AxisEvidence(canonical_size=canonical))
                ev.add_extent(canonical, "T.copy.size.fixed", _CONF_COPY_SIZE)


# --- T.reduce dims= -----------------------------------------------------------

def _collect_reduce_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    fixed_ev: Dict[str, _AxisEvidence],
    alloc_shapes: Dict[str, List[ast.AST]],
) -> None:
    """
    Detect reduction axes from ``T.reduce(src, dst, dims=k, …)``.

    Looks up ``src``'s recorded allocation shape, reads the element at
    index ``dims``, and marks that axis (block param or fixed size) as a
    reduction axis.
    """
    for node in ast.walk(func_ast):
        if not _is_t_call(node, "reduce"):
            continue

        # Extract dims value
        dims_val = _kwarg_int(node, "dims")
        if dims_val is None and len(node.args) >= 3:
            a = node.args[2]
            if isinstance(a, ast.Constant) and isinstance(a.value, int):
                dims_val = a.value
        if dims_val is None:
            continue

        if not node.args or not isinstance(node.args[0], ast.Name):
            continue
        src_name = node.args[0].id

        shape_elts = alloc_shapes.get(src_name)
        if not shape_elts or dims_val >= len(shape_elts):
            continue

        dim_elt = shape_elts[dims_val]
        if not isinstance(dim_elt, ast.Name):
            continue

        canonical = _canon(dim_elt.id, alias_map)
        if canonical in ev_map:
            ev = ev_map[canonical]
            ev.is_reduction = True
            ev.add_extent(dim_elt.id, "T.reduce.dims", _CONF_REDUCE_DIMS)
        else:
            ev = fixed_ev.setdefault(canonical, _AxisEvidence(canonical_size=canonical))
            ev.is_reduction = True
            ev.add_extent(canonical, "T.reduce.dims.fixed", _CONF_REDUCE_DIMS)


# --- Body index decomposition fallback ----------------------------------------

def _collect_body_decomp_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    generator_params: set,
) -> None:
    """
    Fallback split-evidence pass for constant-grid kernels.

    Pattern A: ``expr // T.ceildiv(N, block_N)``  or  ``expr % T.ceildiv(…)``
    Pattern B: ``N // block_N``  (simple tile-count floor-div)
    """
    already_split = {
        param for param, ev in ev_map.items() if ev.split_candidates
    }

    for node in ast.walk(func_ast):
        if not isinstance(node, ast.BinOp):
            continue
        if not isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            continue
        rhs = node.right

        # Pattern A: expr {// | %} T.ceildiv(size, param) ──────────────────────
        if (isinstance(rhs, ast.Call)
                and isinstance(rhs.func, ast.Attribute)
                and rhs.func.attr in _CEILDIV_ATTRS
                and len(rhs.args) == 2
                and isinstance(rhs.args[1], ast.Name)):
            param = _canon(rhs.args[1].id, alias_map)
            if param not in generator_params or param in already_split:
                continue
            size_str = _ast_str(rhs.args[0])
            ev = ev_map.setdefault(param, _AxisEvidence(canonical_param=param))
            ev.add_extent(size_str, "body_decomp.ceildiv", _CONF_BODY_DECOMP)
            ev.add_split(param, None, "body_decomp.ceildiv", _CONF_BODY_DECOMP)
            if ev.canonical_size is None:
                ev.canonical_size = size_str
            continue

        # Pattern B: size // param  (plain tile-count) ──────────────────────────
        if isinstance(node.op, ast.FloorDiv) and isinstance(rhs, ast.Name):
            param = _canon(rhs.id, alias_map)
            if param not in generator_params or param in already_split:
                continue
            size_str = _ast_str(node.left)
            ev = ev_map.setdefault(param, _AxisEvidence(canonical_param=param))
            ev.add_extent(size_str, "body_decomp.floordiv", _CONF_FLOOR_DIV)
            ev.add_split(param, None, "body_decomp.floordiv", _CONF_FLOOR_DIV)
            if ev.canonical_size is None:
                ev.canonical_size = size_str


# --- Buffer annotations (@T.prim_func parameters) ----------------------------

def _collect_buffer_evidence(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
    ev_map: Dict[str, _AxisEvidence],
    fixed_ev: Dict[str, _AxisEvidence],
) -> Tuple[List[str], int]:
    """
    Find ``@T.prim_func`` inner functions and collect:

    * Buffer/tensor parameter names (for ``buffer_params`` / ``buf_count``).
    * Extent evidence from ``T.Tensor((M, K), dtype)`` shape annotations.
    """
    buf_params: List[str] = []

    for node in ast.walk(func_ast):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _is_prim_func_decorator(node):
            continue
        if not isinstance(node.args, ast.arguments):
            continue

        for arg in node.args.args:
            if not isinstance(arg, ast.arg) or arg.annotation is None:
                continue
            ann = arg.annotation
            if not _is_buffer_annotation(ann):
                continue
            if arg.arg in buf_params:
                continue
            buf_params.append(arg.arg)

            # Extract shape from T.Tensor((M, K), dtype) annotation
            if not (isinstance(ann, ast.Call) and ann.args):
                continue
            elts = _elts_of(ann.args[0])
            for dim_idx, elt in enumerate(elts):
                if not isinstance(elt, ast.Name):
                    continue
                canonical = _canon(elt.id, alias_map)
                if canonical in ev_map:
                    ev_map[canonical].add_extent(
                        elt.id, "buffer_annotation", _CONF_BUFFER_ANNOT
                    )
                else:
                    ev = fixed_ev.setdefault(
                        canonical, _AxisEvidence(canonical_size=canonical)
                    )
                    ev.add_extent(canonical, "buffer_annotation.fixed", _CONF_BUFFER_ANNOT)

    return buf_params, len(buf_params)


# --- Alloc shape registry (needed by T.reduce pass) --------------------------

def _build_alloc_shapes(
    func_ast: ast.AST,
    alias_map: Dict[str, str],
) -> Dict[str, List[ast.AST]]:
    """Return ``{var_name: shape_elts}`` for every alloc_shared/fragment."""
    result: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(func_ast):
        if not (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (_is_t_call(call, "alloc_shared")
                or _is_t_call(call, "alloc_fragment")):
            continue
        if call.args:
            elts = _elts_of(call.args[0])
            if elts:
                result[node.targets[0].id] = elts
    return result


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _best_extent(candidates: List[Tuple[str, str, float]]) -> TLAxisExtent:
    if not candidates:
        return TLAxisExtent(expr=None, source="none", confidence=0.0)
    best = max(candidates, key=lambda x: x[2])
    return TLAxisExtent(expr=best[0], source=best[1], confidence=best[2])


def _best_split(candidates: List[Tuple[str, Optional[int], str, float]]) -> TLAxisSplit:
    if not candidates:
        return TLAxisSplit(param=None, grid_dim=None, source="none", confidence=0.0)
    best = max(candidates, key=lambda x: x[3])
    return TLAxisSplit(param=best[0], grid_dim=best[1], source=best[2], confidence=best[3])


def _best_tiling(
    candidates: List[Tuple[str, Optional[str], str, float]],
    split_param: Optional[str],
) -> TLAxisTiling:
    if not candidates:
        return TLAxisTiling(param=None, loop_var=None, source="none", confidence=0.0)
    # Prefer a tiling param that differs from the split param
    filtered = [c for c in candidates if c[0] != split_param]
    best = max(filtered or candidates, key=lambda x: x[3])
    return TLAxisTiling(param=best[0], loop_var=best[1], source=best[2], confidence=best[3])


# ---------------------------------------------------------------------------
# Axis naming
# ---------------------------------------------------------------------------

def _assign_axis_name(
    canonical_param: Optional[str],
    canonical_size: Optional[str],
    is_reduction: bool,
    used_names: set,
) -> str:
    """
    Derive a human-readable axis name.

    Priority:
    1. Strip ``block_`` prefix from the canonical param: ``block_M`` → ``"M"``.
    2. Fall back to the canonical size expression: ``"seq_len"``, ``"dim"``.
    3. Fall back to ``"axis_{n}"``.

    Reduction axes are prefixed with ``"r"`` (``"rK"``, ``"r_seq_len"``).
    """
    base: Optional[str] = None

    if canonical_param:
        stripped = _strip_block_prefix(canonical_param)
        if stripped:
            base = stripped

    if base is None and canonical_size:
        base = canonical_size

    if base is None:
        base = "axis"

    candidate = ("r" + base) if is_reduction else base

    # Dedup
    if candidate not in used_names:
        return candidate
    for i in range(2, 100):
        alt = f"{candidate}_{i}"
        if alt not in used_names:
            return alt
    return candidate + "_x"


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_tilelang_axes(func_ast: ast.AST) -> TLAxisParseResult:
    """
    Parse a TileLang kernel generator AST and return full axis semantic info.

    No user-supplied ``keys`` or ``candidates_params`` are needed.

    Parameters
    ----------
    func_ast:
        AST of the outer Python generator function (obtained via
        ``ast.parse(textwrap.dedent(inspect.getsource(fn)))``).

    Returns
    -------
    TLAxisParseResult
        See class docstring for field descriptions.

    Example
    -------
    ::

        import ast, inspect, textwrap
        from tilelang.autotuner.tilelang_vv_parser import parse_tilelang_axes

        source   = textwrap.dedent(inspect.getsource(my_kernel))
        func_ast = ast.parse(source)
        result   = parse_tilelang_axes(func_ast)

        print(result.split_params)    # {'M': 'block_M', 'N': 'block_N'}
        print(result.tiling_params)   # {'rK': 'block_K'}
        print(result.reduction_axes)  # ['rK']
        print(result.low_dim_axes)    # ['N']
        print(result.buf_count)       # 3
    """
    diagnostics: List[str] = []

    # ── 1. Pre-passes ────────────────────────────────────────────────────────
    alias_map         = _build_alias_map(func_ast)
    generator_params  = _collect_generator_params(func_ast)
    alloc_shapes      = _build_alloc_shapes(func_ast, alias_map)

    ev_map:   Dict[str, _AxisEvidence] = {}
    fixed_ev: Dict[str, _AxisEvidence] = {}

    # ── 2. Evidence collection ───────────────────────────────────────────────
    _collect_kernel_evidence(func_ast, alias_map, ev_map, generator_params)
    _collect_loop_evidence(func_ast, alias_map, ev_map, generator_params)
    _collect_alloc_evidence(func_ast, alias_map, ev_map, fixed_ev)
    _collect_copy_evidence(func_ast, alias_map, ev_map, fixed_ev)
    _collect_reduce_evidence(func_ast, alias_map, ev_map, fixed_ev, alloc_shapes)
    _collect_body_decomp_evidence(func_ast, alias_map, ev_map, generator_params)
    buf_params, buf_count = _collect_buffer_evidence(
        func_ast, alias_map, ev_map, fixed_ev
    )

    # ── 3. Deduplicate fixed_ev against ev_map ───────────────────────────────
    # If a fixed-size variable (e.g. "M", "seq_len") is already captured as
    # the canonical_size of a block-param entry in ev_map, drop it from
    # fixed_ev to avoid creating a redundant duplicate axis.
    covered_sizes = {ev.canonical_size for ev in ev_map.values()
                     if ev.canonical_size}
    fixed_ev = {k: v for k, v in fixed_ev.items()
                if k not in covered_sizes}

    if not ev_map and not fixed_ev:
        return TLAxisParseResult(
            axes={}, split_params={}, tiling_params={},
            low_dim_axes=[], reduction_axes=[],
            buffer_params=buf_params, buf_count=buf_count,
            inferred_keys={},
            status="failed",
            diagnostics=["no axis evidence found in kernel source"],
        )

    # ── 3. Build per-axis info ───────────────────────────────────────────────
    axes:           Dict[str, TLAxisInfo] = {}
    split_params:   Dict[str, str]        = {}
    tiling_params:  Dict[str, str]        = {}
    low_dim_axes:   List[str]             = []
    reduction_axes: List[str]             = []
    inferred_keys:  Dict[str, str]        = {}
    used_names:     set                   = set()

    # Sort: split axes first (have split_candidates), then tiling, then fixed
    def _sort_key(item):
        _, ev = item
        has_split  = bool(ev.split_candidates)
        has_tiling = bool(ev.tiling_candidates)
        return (0 if has_split else (1 if has_tiling else 2), ev.canonical_param or "")

    all_evidence = (
        list(ev_map.items()) + list(fixed_ev.items())
    )
    all_evidence.sort(key=_sort_key)

    for key, ev in all_evidence:
        # Skip if no real evidence at all
        if not (ev.extent_candidates or ev.split_candidates or ev.tiling_candidates):
            continue

        extent  = _best_extent(ev.extent_candidates)
        split   = _best_split(ev.split_candidates)
        tiling  = _best_tiling(ev.tiling_candidates, split.param)

        # Decide axis name
        axis_name = _assign_axis_name(
            ev.canonical_param,
            ev.canonical_size or extent.expr,
            ev.is_reduction,
            used_names,
        )
        used_names.add(axis_name)

        # Clean up self-referential extents like {"block_M": "block_M"}
        # that arise when a block param is used as its own size expression
        # in alloc shapes (e.g. T.alloc_shared((block_M, block_N))).
        # They add noise to inferred_keys; drop them here.
        clean_extent_expr = extent.expr
        if (clean_extent_expr is not None
                and clean_extent_expr == ev.canonical_param
                and ev.split_candidates):
            # Prefer a non-self extent from other sources if available
            better = [
                e for e, s, c in ev.extent_candidates
                if e != ev.canonical_param
            ]
            if better:
                clean_extent_expr = max(
                    better,
                    key=lambda e: max(
                        c for ex, s, c in ev.extent_candidates if ex == e
                    ),
                )
            else:
                clean_extent_expr = None  # suppress self-referential key

        # Collect per-axis diagnostics
        ax_diag: List[str] = list(ev.diagnostics)
        if not ev.extent_candidates:
            ax_diag.append("no extent resolved")
        if ev.split_candidates and ev.tiling_candidates:
            ax_diag.append(
                f"axis has both split ({split.param}) and tiling ({tiling.param}) "
                f"candidates — check kernel structure"
            )

        ax_info = TLAxisInfo(
            axis_name=axis_name,
            extent=extent,
            split=split,
            tiling=tiling,
            is_low_dim=ev.is_low_dim,
            is_reduction=ev.is_reduction,
            diagnostics=ax_diag,
        )
        axes[axis_name] = ax_info

        if clean_extent_expr:
            inferred_keys[axis_name] = clean_extent_expr
        if split.param:
            split_params[axis_name] = split.param
        if tiling.param:
            tiling_params[axis_name] = tiling.param
        if ev.is_low_dim:
            low_dim_axes.append(axis_name)
        if ev.is_reduction:
            reduction_axes.append(axis_name)

        if ax_diag:
            diagnostics.extend(f"{axis_name}: {d}" for d in ax_diag)

    # ── 4. Determine status ──────────────────────────────────────────────────
    unresolved = [n for n, ax in axes.items() if ax.extent.expr is None]
    if not axes:
        status = "failed"
        diagnostics.append("no axes resolved")
    elif unresolved:
        status = "partial"
        diagnostics.append(f"unresolved extents: {unresolved}")
    else:
        status = "ok"

    return TLAxisParseResult(
        axes=axes,
        split_params=split_params,
        tiling_params=tiling_params,
        low_dim_axes=low_dim_axes,
        reduction_axes=reduction_axes,
        buffer_params=buf_params,
        buf_count=buf_count,
        inferred_keys=inferred_keys,
        status=status,
        diagnostics=diagnostics,
    )