# ---- numpy_trace: core tracer (sys.monitoring, Python 3.12+) ----
import ast
import json
import os
import sys
import time
import types
import linecache

import numpy as np

MAX_NODES_DEFAULT = 30000
MAX_LINES_PER_NODE = 3000      # 1 ノードあたりに記録する実行行（順序）の上限
MAX_LINES_TOTAL = 400000       # トレース全体の実行行記録の上限

_NP_DIR = os.path.dirname(os.path.abspath(np.__file__))
_NP_PARENT = os.path.dirname(_NP_DIR)
_THIS_FILE = os.path.abspath(__file__)

# 演算子 -> ufunc 名
_BINOPS = {
    ast.Add: ("add", "+"), ast.Sub: ("subtract", "-"), ast.Mult: ("multiply", "*"),
    ast.Div: ("divide", "/"), ast.FloorDiv: ("floor_divide", "//"), ast.Mod: ("remainder", "%"),
    ast.Pow: ("power", "**"), ast.MatMult: ("matmul", "@"), ast.LShift: ("left_shift", "<<"),
    ast.RShift: ("right_shift", ">>"), ast.BitAnd: ("bitwise_and", "&"),
    ast.BitOr: ("bitwise_or", "|"), ast.BitXor: ("bitwise_xor", "^"),
}
_CMPOPS = {
    ast.Lt: ("less", "<"), ast.LtE: ("less_equal", "<="), ast.Gt: ("greater", ">"),
    ast.GtE: ("greater_equal", ">="), ast.Eq: ("equal", "=="), ast.NotEq: ("not_equal", "!="),
}
_UNOPS = {ast.USub: ("negative", "-"), ast.UAdd: ("positive", "+"), ast.Invert: ("invert", "~")}
_OPFUNC = {}
import operator as _op
for _k, _v in {"add": _op.add, "subtract": _op.sub, "multiply": _op.mul, "divide": _op.truediv,
               "floor_divide": _op.floordiv, "remainder": _op.mod, "power": _op.pow,
               "matmul": _op.matmul, "left_shift": _op.lshift, "right_shift": _op.rshift,
               "bitwise_and": _op.and_, "bitwise_or": _op.or_, "bitwise_xor": _op.xor,
               "less": _op.lt, "less_equal": _op.le, "greater": _op.gt, "greater_equal": _op.ge,
               "equal": _op.eq, "not_equal": _op.ne, "negative": _op.neg,
               "positive": _op.pos, "invert": _op.invert}.items():
    _OPFUNC[_k] = _v

_DUNDER = {"add": "__add__", "subtract": "__sub__", "multiply": "__mul__", "divide": "__truediv__",
           "floor_divide": "__floordiv__", "remainder": "__mod__", "power": "__pow__",
           "matmul": "__matmul__", "left_shift": "__lshift__", "right_shift": "__rshift__",
           "bitwise_and": "__and__", "bitwise_or": "__or__", "bitwise_xor": "__xor__",
           "less": "__lt__", "less_equal": "__le__", "greater": "__gt__",
           "greater_equal": "__ge__", "equal": "__eq__", "not_equal": "__ne__",
           "negative": "__neg__", "positive": "__pos__", "invert": "__invert__"}


def _ntop_(_nt_name, _nt_sym, *_nt_args):
    """ユーザーコード内の演算子 (a @ b など) を可視化するためのラッパー"""
    return _OPFUNC[_nt_name](*_nt_args)


class _OpRewriter(ast.NodeTransformer):
    def _call(self, node, name, sym, args):
        new = ast.Call(func=ast.Name(id="_ntop_", ctx=ast.Load()),
                       args=[ast.Constant(name), ast.Constant(sym)] + args, keywords=[])
        return ast.copy_location(new, node)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        t = _BINOPS.get(type(node.op))
        return self._call(node, t[0], t[1], [node.left, node.right]) if t else node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        t = _UNOPS.get(type(node.op))
        return self._call(node, t[0], t[1], [node.operand]) if t else node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
            t = _CMPOPS[type(node.ops[0])]
            return self._call(node, t[0], t[1], [node.left, node.comparators[0]])
        return node


# ---------------------------------------------------------------------------
def _short(v, n=70):
    try:
        if isinstance(v, np.ndarray):
            s = f"ndarray{tuple(v.shape)} {v.dtype}"
            if not v.flags.c_contiguous:
                s += " F" if v.flags.f_contiguous else " 非連続"
            return s
        if isinstance(v, np.generic):
            return f"np.{type(v).__name__}({str(v)[:24]})"
        if isinstance(v, np.dtype):
            return f"dtype('{v}')"
        if v is None or isinstance(v, (bool, int, float, complex)):
            return repr(v)[:n]
        if isinstance(v, str):
            r = repr(v)
            return r if len(r) <= n else r[:n - 1] + "…'"
        if isinstance(v, (tuple, list)):
            inner = ", ".join(_short(x, 24) for x in list(v)[:4])
            if len(v) > 4:
                inner += ", …"
            o, c = ("(", ")") if isinstance(v, tuple) else ("[", "]")
            return f"{o}{inner}{c}"[:n]
        if isinstance(v, type):
            return f"<class {v.__module__}.{v.__qualname__}>"[:n]
        if isinstance(v, dict):
            return f"dict(len={len(v)})"
        if callable(v):
            return f"<{type(v).__name__} {getattr(v, '__qualname__', getattr(v, '__name__', '?'))}>"[:n]
        return f"<{type(v).__module__}.{type(v).__qualname__}>"[:n]
    except Exception:
        return "<?>"


def _blas_info():
    out = {"blas": None, "lapack": None}
    try:
        cfg = np.show_config(mode="dicts")
        bd = cfg.get("Build Dependencies", {})
        for k in ("blas", "lapack"):
            d = bd.get(k) or {}
            if d:
                out[k] = {"name": d.get("name"), "version": d.get("version"),
                          "config": d.get("openblas configuration")}
    except Exception:
        pass
    return out


def _dt_prefix(arr):
    try:
        c = np.asarray(arr).dtype.char
    except Exception:
        return None
    return {"f": "s", "d": "d", "F": "c", "D": "z"}.get(c)


_LAPACK = {
    "solve": "?gesv", "solve1": "?gesv", "inv": "?gesv",
    "det": "?getrf", "slogdet": "?getrf",
    "eigh_lo": "?syevd/?heevd", "eigh_up": "?syevd/?heevd",
    "eigvalsh_lo": "?syevd/?heevd", "eigvalsh_up": "?syevd/?heevd",
    "eig": "?geev", "eigvals": "?geev",
    "svd": "?gesdd", "svd_f": "?gesdd", "svd_s": "?gesdd",
    "cholesky_lo": "?potrf", "cholesky_up": "?potrf",
    "qr_r_raw": "?geqrf", "qr_reduced": "?orgqr/?ungqr", "qr_complete": "?orgqr/?ungqr",
    "lstsq": "?gelsd",
}
_BLAS_FUNCS = {"matmul", "dot", "vdot", "inner", "vecdot", "matvec", "vecmat"}


def _fill_prefix(routine, pre):
    if not pre:
        return routine.replace("?", "x")
    parts = []
    for r in routine.split("/"):
        if r.startswith("?sy") and pre in "cz":
            continue
        if r.startswith("?he") and pre in "sd":
            continue
        if r.startswith("?org") and pre in "cz":
            continue
        if r.startswith("?ung") and pre in "sd":
            continue
        parts.append(r.replace("?", pre))
    return "/".join(parts) or routine.replace("?", pre)


# ---------------------------------------------------------------------------
class Tracer:
    TOOL_ID = 4

    def __init__(self, user_files, max_nodes=MAX_NODES_DEFAULT, user_dirs=(),
                 embed_stdlib=True, record_lines=True):
        self.user_dirs = [os.path.abspath(d) + os.sep for d in user_dirs]
        self.embed_stdlib = embed_stdlib
        self.record_lines = record_lines
        self.qtotal = 0
        self._line_ok = {}
        self.user_files = set(os.path.abspath(f) if not f.startswith("<") else f for f in user_files)
        self.max_nodes = max_nodes
        self.nodes = []
        self.files = {}       # filename -> id
        self.file_list = []   # [{path, src, kind}]
        self.cinfo = {}       # key -> info
        self.stack = []       # entries: [etype, key, node, ctx]
        self.truncated = False
        self.blas = _blas_info()
        self._pos_cache = {}
        self.gen_nodes = {}
        self._ext_map = self._build_ext_map()
        self._file_kind_cache = {}

    # --- helpers -----------------------------------------------------------
    def _build_ext_map(self):
        import importlib
        for sub in ("numpy.fft", "numpy.linalg", "numpy.random"):
            try:
                importlib.import_module(sub)
            except Exception:
                pass
        m = {}
        for name, mod in list(sys.modules.items()):
            if not name.startswith("numpy") or mod is None:
                continue
            f = getattr(mod, "__file__", "") or ""
            if not f.endswith((".so", ".pyd")):
                continue
            try:
                for v in vars(mod).values():
                    m.setdefault(id(v), name)
            except Exception:
                pass
        return m

    def _file_kind(self, fn):
        k = self._file_kind_cache.get(fn)
        if k is not None:
            return k
        if fn.startswith("<frozen importlib") or fn.startswith("<frozen zipimport"):
            k = "mute"
        elif os.path.abspath(fn) == _THIS_FILE if not fn.startswith("<") else False:
            k = "tracer"
        elif (fn if fn.startswith("<") else os.path.abspath(fn)) in self.user_files:
            k = "user"
        elif not fn.startswith("<") and os.path.abspath(fn).startswith(_NP_DIR + os.sep):
            k = "np"
        elif (not fn.startswith("<") and self.user_dirs
              and any(os.path.abspath(fn).startswith(d) for d in self.user_dirs)
              and "site-packages" not in fn and "dist-packages" not in fn):
            k = "user"
        else:
            k = "other"
        self._file_kind_cache[fn] = k
        return k

    def _fid(self, fn, kind):
        i = self.files.get(fn)
        if i is not None:
            return i
        if kind == "np":
            disp = os.path.relpath(os.path.abspath(fn), _NP_PARENT).replace(os.sep, "/")
        elif kind == "user":
            disp = "user/" + (os.path.basename(fn) if not fn.startswith("<") else fn.strip("<>") + ".py")
            for d in self.user_dirs:
                if os.path.abspath(fn).startswith(d):
                    disp = "user/" + os.path.relpath(fn, d).replace(os.sep, "/")
        else:
            disp = fn
            try:
                import sysconfig
                std = sysconfig.get_paths()["stdlib"]
                if os.path.abspath(fn).startswith(std + os.sep):
                    disp = "python-stdlib/" + os.path.relpath(fn, std).replace(os.sep, "/")
            except Exception:
                pass
        src = ""
        if kind != "stdlib" or self.embed_stdlib:
            try:
                src = "".join(linecache.getlines(fn))
                if not src and not fn.startswith("<"):
                    with open(fn, encoding="utf-8", errors="replace") as fh:
                        src = fh.read()
            except Exception:
                src = ""
        i = len(self.file_list)
        self.files[fn] = i
        self.file_list.append({"path": disp, "src": src, "kind": kind})
        return i

    def _line_of(self, code, off):
        pos = self._pos_cache.get(code)
        if pos is None:
            pos = list(code.co_positions())
            self._pos_cache[code] = pos
        try:
            ln = pos[off // 2][0]
            return ln if ln is not None else code.co_firstlineno
        except Exception:
            return code.co_firstlineno

    def _top(self):
        return self.stack[-1] if self.stack else None

    def _ctx(self):
        t = self._top()
        return t[3] if t else "user"

    def _parent_node(self):
        for e in reversed(self.stack):
            if e[2] >= 0:
                return e[2]
        return -1

    def _now(self):
        return time.perf_counter_ns() - self.t_start

    def _new(self, **kw):
        if len(self.nodes) >= self.max_nodes:
            self.truncated = True
            return -1
        kw["i"] = len(self.nodes)
        kw.setdefault("p", self._parent_node())
        kw["t0"] = self._now()
        kw["t1"] = None
        self.nodes.append(kw)
        p = kw["p"]
        if p >= 0 and self.record_lines:
            pq = self.nodes[p].get("q")
            if pq is not None:
                if len(pq) < MAX_LINES_PER_NODE and self.qtotal < MAX_LINES_TOTAL:
                    pq.append(-(kw["i"] + 1))
                    self.qtotal += 1
                else:
                    self.nodes[p]["qt"] = 1
        return kw["i"]

    def _close(self, idx):
        if idx >= 0 and self.nodes[idx]["t1"] is None:
            self.nodes[idx]["t1"] = self._now()
            for ch in self.nodes[idx].get("_est", []):
                self.nodes[ch]["t1"] = self.nodes[idx]["t1"]

    # --- C callable classification -------------------------------------
    def _c_describe(self, c):
        """returns (is_numpy, key, info) for a non-Python callable"""
        tname = type(c).__name__
        name = getattr(c, "__name__", None) or tname
        impl = self._ext_map.get(id(c))
        qual = name
        is_np = False
        kind_label = tname
        if isinstance(c, np.ufunc):
            is_np = True
            impl = impl or "numpy._core._multiarray_umath"
            qual = f"ufunc {name}" if not c.signature else f"gufunc {name}"
        elif tname == "_ArrayFunctionDispatcher":
            is_np = True
            impl = "numpy._core._multiarray_umath"
            mod = getattr(c, "__module__", "numpy") or "numpy"
            qual = f"{mod}.{name}"
            kind_label = "__array_function__ ディスパッチャ"
        elif isinstance(c, (types.BuiltinFunctionType, types.BuiltinMethodType)) or tname in (
                "cython_function_or_method", "method-wrapper", "fused_cython_function"):
            slf = getattr(c, "__self__", None)
            if isinstance(slf, np.ufunc):
                is_np = True
                impl = "numpy._core._multiarray_umath"
                qual = f"ufunc {slf.__name__}.{name}"
                kind_label = "ufunc メソッド"
            elif isinstance(slf, types.ModuleType) or slf is None:
                mod = getattr(slf, "__name__", None) or getattr(c, "__module__", None) or ""
                if mod.startswith("numpy") or (getattr(c, "__module__", "") or "").startswith("numpy"):
                    is_np = True
                    impl = impl or mod
                    qual = f"{getattr(c, '__module__', mod) or mod}.{name}"
                else:
                    qual = f"{mod}.{name}" if mod and mod != "builtins" else name
            else:
                t = slf if isinstance(slf, type) else type(slf)
                tm = t.__module__ or ""
                qual = f"{t.__qualname__}.{name}"
                if tm.startswith("numpy"):
                    is_np = True
                    impl = impl or (tm if tm != "numpy" else "numpy._core._multiarray_umath")
                    qual = f"{tm}.{t.__qualname__}.{name}" if tm != "numpy" else f"numpy.{t.__qualname__}.{name}"
        elif isinstance(c, (types.MethodDescriptorType, types.WrapperDescriptorType,
                            types.ClassMethodDescriptorType)):
            oc = getattr(c, "__objclass__", object)
            tm = oc.__module__ or ""
            qual = f"{tm}.{oc.__qualname__}.{name}" if tm != "builtins" else f"{oc.__qualname__}.{name}"
            kind_label = "メソッド (C 実装)"
            if tm.startswith("numpy"):
                is_np = True
                impl = "numpy._core._multiarray_umath" if tm == "numpy" else tm
        elif isinstance(c, type):
            mod = c.__module__ or ""
            qual = f"{mod}.{c.__qualname__}"
            kind_label = "type (インスタンス生成)"
            if mod.startswith("numpy"):
                is_np = True
                impl = impl or "numpy._core._multiarray_umath"
        elif isinstance(c, types.MethodWrapperType) or tname == "wrapper_descriptor":
            slf = getattr(c, "__self__", None)
            t = type(slf)
            qual = f"{t.__qualname__}.{name}"
            if (t.__module__ or "").startswith("numpy"):
                is_np = True
                impl = "numpy._core._multiarray_umath"
        else:
            mod = getattr(c, "__module__", "") or ""
            tmod = type(c).__module__ or ""
            if tmod.startswith("numpy"):
                is_np = True
                impl = impl or tmod
            qual = f"{mod}.{name}" if mod else name
            if not is_np:
                qual = f"{qual}  [{tmod}.{type(c).__qualname__}]"
        key = qual
        if key not in self.cinfo:
            self.cinfo[key] = self._c_info(c, qual, impl, kind_label, is_np)
        return is_np, key

    def _c_info(self, c, qual, impl, kind_label, is_np):
        info = {"qual": qual, "type": kind_label, "impl": impl, "so": None, "doc": None,
                "hint": None, "ufunc": None, "np": is_np}
        if impl and impl in sys.modules:
            f = getattr(sys.modules[impl], "__file__", None)
            if f:
                try:
                    info["so"] = os.path.relpath(f, _NP_PARENT).replace(os.sep, "/")
                except Exception:
                    info["so"] = f
        try:
            d = getattr(c, "__doc__", None) if is_np else None
            if isinstance(d, str):
                import inspect as _insp
                info["doc"] = "\n".join(_insp.cleandoc(d).splitlines()[:40])
        except Exception:
            pass
        u = c if isinstance(c, np.ufunc) else getattr(c, "__self__", None)
        if isinstance(u, np.ufunc):
            c = u
            try:
                info["ufunc"] = {"nin": c.nin, "nout": c.nout, "ntypes": c.ntypes,
                                 "signature": c.signature, "types": c.types[:48],
                                 "identity": repr(c.identity)}
            except Exception:
                pass
        hint = None
        if impl:
            if impl.startswith("numpy.linalg._umath_linalg"):
                hint = "numpy/linalg/umath_linalg.cpp"
            elif impl.startswith("numpy.fft._pocketfft_umath"):
                hint = "numpy/fft/_pocketfft_umath.cpp"
            elif impl.startswith("numpy.random"):
                hint = "numpy/random/" + impl.split(".")[-1] + ".pyx (Cython)"
            elif impl.startswith("numpy._core"):
                hint = ("numpy/_core/src/umath/" if isinstance(c, np.ufunc)
                        else "numpy/_core/src/multiarray/")
            elif impl.startswith("numpy.linalg.lapack_lite"):
                hint = "numpy/linalg/lapack_litemodule.c"
        info["hint"] = hint
        return info

    def _add_native_children(self, parent_idx, c, ufname, args):
        """L3: BLAS/LAPACK/pocketfft/random C 層の推定ノード"""
        if parent_idx < 0:
            return
        impl = self._ext_map.get(id(c), "")
        name = ufname
        kids = []
        lib = self.blas.get("lapack") or {}
        blib = self.blas.get("blas") or {}
        a0 = args[0] if args else None
        pre = _dt_prefix(a0) if a0 is not None else None
        if impl.startswith("numpy.linalg._umath_linalg") and name in _LAPACK:
            r = _fill_prefix(_LAPACK[name], pre)
            kids.append((f"LAPACK {r}", "lapack",
                         {"lib": f"{lib.get('name')} {lib.get('version')}", "routine": r,
                          "note": "numpy/linalg/umath_linalg.cpp が LAPACK を呼ぶ。ルーチン名は gufunc 名と dtype からの推定。"}))
        elif name in _BLAS_FUNCS:
            if pre:
                nd = [np.ndim(a) for a in args[:2]] if args else []
                if len(nd) == 2:
                    if nd == [2, 2]:
                        cand = f"cblas_{pre}gemm (A@A.T なら {pre}syrk)"
                    elif 1 in nd and 2 in nd:
                        cand = f"cblas_{pre}gemv"
                    elif nd == [1, 1]:
                        cand = f"cblas_{pre}dot" if pre in "sd" else f"cblas_{pre}dotu_sub"
                    else:
                        cand = f"cblas_{pre}gemm (バッチ毎) / 内蔵ループ"
                else:
                    cand = f"cblas_{pre}gemm / {pre}gemv / {pre}dot"
                kids.append((f"BLAS {cand}", "blas",
                             {"lib": f"{blib.get('name')} {blib.get('version')}", "routine": cand,
                              "note": "float/complex かつメモリ配置が BLAS 互換なら CBLAS、そうでなければ numpy 内蔵ループ。形状が分からない場合は候補を列挙（推定）。"}))
            elif a0 is not None:
                kids.append(("numpy 内蔵ループ (BLAS 非使用)", "blas",
                             {"lib": "numpy", "routine": "-", "note": "整数/bool/object dtype は BLAS を使わない（推定）。"}))
        elif impl.startswith("numpy.fft._pocketfft_umath"):
            kids.append(("pocketfft (C++ / numpy 同梱)", "blas",
                         {"lib": "pocketfft", "routine": name,
                          "note": "numpy/fft/pocketfft/ のヘッダオンリー C++ 実装。外部ライブラリではない。"}))
        elif (isinstance(getattr(c, "__self__", None), np.random.Generator) or isinstance(a0, np.random.Generator)) \
                and not name.startswith(("spawn", "bit_generator", "__")):
            kids.append(("C 分布関数 + BitGenerator", "blas",
                         {"lib": "numpy.random (C)", "routine": name,
                          "note": "numpy/random/src/distributions/*.c と BitGenerator (既定 PCG64) の C 実装を経由（推定）。"}))
        for (nm, kd, inf) in kids:
            key = "native:" + nm
            self.cinfo.setdefault(key, {"qual": nm, "type": kd, "native": inf, "np": True})
            idx = self._new(k="blas", n=nm, m=inf["lib"], f=-1, c=key, p=parent_idx,
                            cf=-1, cl=0, est=True)
            if idx >= 0:
                self.nodes[parent_idx].setdefault("_est", []).append(idx)

    # --- callbacks ---------------------------------------------------------
    def on_py_start(self, code, off, _depth=1):
        fn = code.co_filename
        fk = self._file_kind(fn)
        pctx = self._ctx()
        if pctx == "mute" or fk == "mute":
            self.stack.append(["py", code, -1, "mute"])
            return
        if fk == "tracer":
            if code.co_name == "_ntop_":
                self._start_op(code)
            else:
                self.stack.append(["py", code, -1, "other"])
            return
        if fk == "np":
            kind, ctx = "np", "np"
        elif fk == "user":
            kind, ctx = "user", "user"
        elif pctx in ("np", "stdlib_np"):
            kind, ctx = "stdlib", "stdlib_np"
        else:
            self.stack.append(["py", code, -1, "other"])
            return
        fid = self._fid(fn, kind)
        frame = sys._getframe(_depth)
        caller = frame.f_back
        cf, cl = -1, 0
        if caller is not None:
            ck = self._file_kind(caller.f_code.co_filename)
            if ck in ("np", "user"):
                cf = self._fid(caller.f_code.co_filename, ck)
                cl = caller.f_lineno or 0
            elif ck == "tracer" and caller.f_back is not None:
                c2 = caller.f_back
                k2 = self._file_kind(c2.f_code.co_filename)
                if k2 in ("np", "user"):
                    cf = self._fid(c2.f_code.co_filename, k2)
                    cl = c2.f_lineno or 0
        args = []
        try:
            nargs = code.co_argcount + code.co_kwonlyargcount
            flags = code.co_flags
            if flags & 0x04:
                nargs += 1
            if flags & 0x08:
                nargs += 1
            loc = frame.f_locals
            for vn in code.co_varnames[:nargs]:
                if vn in loc:
                    args.append([vn, _short(loc[vn])])
        except Exception:
            pass
        end = code.co_firstlineno
        try:
            for _, _, ln in code.co_lines():
                if ln and ln > end:
                    end = ln
        except Exception:
            pass
        mod = frame.f_globals.get("__name__", "") if kind != "user" else "__main__"
        idx = self._new(k=kind, n=code.co_qualname, m=mod, f=fid, l=code.co_firstlineno,
                        e=end, cf=cf, cl=cl, a=args)
        if idx >= 0 and self.record_lines and kind in ("np", "user"):
            self.nodes[idx]["q"] = []
            self.nodes[idx]["h"] = {}
        self.stack.append(["py", code, idx, ctx])
        if code.co_flags & 0x2A0 and idx >= 0:
            self.gen_nodes[id(frame)] = (frame, idx, ctx)

    def _start_op(self, code):
        frame = sys._getframe(2)
        loc = frame.f_locals
        name = loc.get("_nt_name")
        sym = loc.get("_nt_sym")
        args = loc.get("_nt_args", ())
        is_np = any(isinstance(a, (np.ndarray, np.generic)) or hasattr(type(a), "__array_ufunc__")
                    for a in args)
        if not is_np:
            self.stack.append(["py", code, -1, "other"])
            return
        caller = frame.f_back
        cf, cl = -1, 0
        if caller is not None:
            ck = self._file_kind(caller.f_code.co_filename)
            if ck in ("np", "user"):
                cf = self._fid(caller.f_code.co_filename, ck)
                cl = caller.f_lineno or 0
        uf = getattr(np, name, None)
        tn = next((type(a).__name__ for a in args if isinstance(a, (np.ndarray, np.generic))), "ndarray")
        dn = _DUNDER.get(name, name)
        qual = f"演算子 {sym}  →  {tn}.{dn}  →  ufunc np.{name}"
        key = f"op:{name}"
        if key not in self.cinfo:
            info = self._c_info(uf, qual, "numpy._core._multiarray_umath", "演算子 (ndarray の数値プロトコル)", True)
            info["hint"] = "numpy/_core/src/multiarray/number.c → numpy/_core/src/umath/"
            self.cinfo[key] = info
        idx = self._new(k="c", n=f"{tn}.{dn}  ( {sym} )", m="numpy", f=-1, c=key, cf=cf, cl=cl,
                        a=[[f"x{i}", _short(a)] for i, a in enumerate(args)])
        self.stack.append(["py", code, idx, "op", (uf, name, list(args)) if uf is not None else None])

    def on_py_end(self, code, off, *rest):
        st = self.stack
        if st and st[-1][0] == "py" and st[-1][1] is code:
            e = st.pop()
            if len(e) > 4 and e[4] and e[2] >= 0:
                self._add_native_children(e[2], *e[4])
            if e[2] >= 0:
                if rest and e[3] != "op":
                    try:
                        self.nodes[e[2]]["r"] = _short(rest[0])
                    except Exception:
                        pass
                self._close(e[2])
            return
        for j in range(len(st) - 1, -1, -1):
            if st[j][0] == "py" and st[j][1] is code:
                while len(st) > j:
                    e = st.pop()
                    self._close(e[2])
                return

    def on_py_resume(self, code, off, *rest):
        frame = sys._getframe(1)
        g = self.gen_nodes.get(id(frame))
        if g is not None and g[0] is frame:
            _, idx, ctx = g
            nd = self.nodes[idx]
            nd["rs"] = nd.get("rs", 0) + 1
            nd["t1"] = None
            self.stack.append(["py", code, idx, ctx])
            return
        self.on_py_start(code, off, _depth=2)

    def on_call(self, code, off, c, arg0):
        ctx = self._ctx()
        if ctx == "mute":
            return
        if isinstance(c, types.FunctionType):
            return
        if isinstance(c, types.MethodType) and isinstance(c.__func__, types.FunctionType):
            return
        if ctx == "op":
            # _ntop_ 内の operator.* 呼び出しは記録しない（ノードは既に作成済み）
            return
        is_np, key = self._c_describe(c)
        if is_np:
            kind = "c"
        elif ctx in ("np", "stdlib_np"):
            kind = "builtin"
        else:
            return
        fk = self._file_kind(code.co_filename)
        cf = self._fid(code.co_filename, fk) if fk in ("np", "user") else -1
        cl = self._line_of(code, off)
        a = [["arg0", _short(arg0)]] if arg0 is not sys.monitoring.MISSING else []
        info = self.cinfo[key]
        idx = self._new(k=kind, n=info["qual"].split(" ")[-1] if kind == "c" else info["qual"],
                        m=info.get("impl") or "", f=-1, c=key, cf=cf, cl=cl, a=a)
        pend = None
        if kind == "c" and idx >= 0:
            nm = getattr(c, "__name__", "")
            if isinstance(c, np.ufunc) or nm in _BLAS_FUNCS or (self._ext_map.get(id(c), "").startswith(("numpy.random", "numpy.fft"))) \
                    or isinstance(getattr(c, "__self__", None), np.random.Generator) or isinstance(arg0, np.random.Generator):
                args = [arg0] if arg0 is not sys.monitoring.MISSING else []
                pend = (c, nm, args)
        self.stack.append(["c", id(c), idx, "np" if kind == "c" else "stdlib_np", pend])

    def on_line(self, code, line):
        ok = self._line_ok.get(code)
        if ok is None:
            ok = self._file_kind(code.co_filename) in ("np", "user")
            self._line_ok[code] = ok
        if not ok:
            return sys.monitoring.DISABLE
        st = self.stack
        e = None
        if st and st[-1][0] == "py" and st[-1][1] is code:
            e = st[-1]
        else:
            for j in range(len(st) - 1, max(-1, len(st) - 6), -1):
                if st[j][0] == "py" and st[j][1] is code:
                    e = st[j]
                    break
        if e is None or e[2] < 0:
            return
        nd = self.nodes[e[2]]
        h = nd.get("h")
        if h is None:
            return
        h[line] = h.get(line, 0) + 1
        q = nd["q"]
        if len(q) < MAX_LINES_PER_NODE and self.qtotal < MAX_LINES_TOTAL:
            q.append(line)
            self.qtotal += 1
        else:
            nd["qt"] = 1

    def on_c_end(self, code, off, c, arg0):
        st = self.stack
        if st and st[-1][0] == "c" and st[-1][1] == id(c):
            e = st.pop()
            if e[4] and e[2] >= 0:
                self._add_native_children(e[2], *e[4])
            self._close(e[2])

    # --- run ---------------------------------------------------------------
    def run(self, fn):
        mon = sys.monitoring
        E = mon.events
        tid = self.TOOL_ID
        while mon.get_tool(tid) is not None and tid < 6:
            tid += 1
        mon.use_tool_id(tid, "numpy_trace")
        cbs = {E.PY_START: self.on_py_start, E.PY_RETURN: self.on_py_end,
               E.PY_UNWIND: self.on_py_end, E.PY_YIELD: self.on_py_end,
               E.PY_RESUME: self.on_py_resume, E.PY_THROW: self.on_py_resume,
               E.CALL: self.on_call, E.C_RETURN: self.on_c_end, E.C_RAISE: self.on_c_end}
        if self.record_lines:
            cbs[E.LINE] = self.on_line
        for ev, cb in cbs.items():
            mon.register_callback(tid, ev, cb)
        err = None
        self.t_start = time.perf_counter_ns()
        mon.restart_events()
        mon.set_events(tid, E.PY_START | E.PY_RETURN | E.PY_UNWIND | E.PY_YIELD |
                       E.PY_RESUME | E.PY_THROW | E.CALL | (E.LINE if self.record_lines else 0))
        try:
            fn()
        except BaseException as ex:  # noqa
            err = f"{type(ex).__name__}: {ex}"
        finally:
            mon.set_events(tid, 0)
            for ev in cbs:
                mon.register_callback(tid, ev, None)
            mon.free_tool_id(tid)
            mon.restart_events()
        end = self._now()
        self.gen_nodes.clear()
        for nd in self.nodes:
            if nd["t1"] is None:
                nd["t1"] = end
            nd.pop("_est", None)
        return err


def trace_source(src, filename="<user>", title=None, max_nodes=MAX_NODES_DEFAULT, user_dirs=(),
                 embed_stdlib=True, record_lines=True):
    """ソース文字列を実行してトレース結果(dict)を返す"""
    tree = ast.parse(src, filename=filename)
    tree = ast.fix_missing_locations(_OpRewriter().visit(tree))
    code = compile(tree, filename, "exec")
    if filename.startswith("<"):
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    g = {"__name__": "__main__", "__file__": filename, "_ntop_": _ntop_}
    import io
    import contextlib
    out = io.StringIO()
    tr = Tracer([filename], max_nodes=max_nodes, user_dirs=user_dirs,
                embed_stdlib=embed_stdlib, record_lines=record_lines)

    def body():
        with contextlib.redirect_stdout(out):
            exec(code, g)

    err = tr.run(body)
    # ユーザーファイルが呼ばれていない場合でも source を登録
    tr._fid(filename, "user")
    root_children = [n for n in tr.nodes if n["p"] == -1]
    return {
        "title": title or os.path.basename(filename),
        "code": src,
        "stdout": out.getvalue()[-20000:],
        "error": err,
        "nodes": tr.nodes,
        "files": tr.file_list,
        "cinfo": tr.cinfo,
        "truncated": tr.truncated,
        "roots": [n["i"] for n in root_children],
    }


def env_info():
    import platform
    b = _blas_info()
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "blas": b.get("blas"), "lapack": b.get("lapack"),
            "numpy_dir": _NP_DIR}


def merge_traces(traces):
    """複数トレースのファイルを共有テーブルにまとめる"""
    files, fmap = [], {}
    out = []
    for t in traces:
        remap = []
        for f in t["files"]:
            k = f["path"] if f["kind"] == "np" else (t["title"] + "::" + f["path"])
            if k not in fmap:
                fmap[k] = len(files)
                files.append(f)
            remap.append(fmap[k])
        nodes = []
        for n in t["nodes"]:
            n = dict(n)
            if n.get("f", -1) >= 0:
                n["f"] = remap[n["f"]]
            if n.get("cf", -1) >= 0:
                n["cf"] = remap[n["cf"]]
            nodes.append(n)
        t2 = {k: v for k, v in t.items() if k != "files"}
        t2["nodes"] = nodes
        out.append(t2)
    return {"files": files, "traces": out}


def numpy_license_notices(paths):
    """埋め込んだ NumPy ソースに適用されるライセンス文を、インストール済み NumPy から集める"""
    out = []
    try:
        import importlib.metadata as md
        files = md.files("numpy") or []
    except Exception:
        return out
    lic = []
    for f in files:
        sf = str(f).replace(os.sep, "/")
        name = sf.rsplit("/", 1)[-1].upper()
        if not (name.startswith("LICENSE") or name.startswith("COPYING")):
            continue
        if ".dist-info/licenses/" in sf:
            rel = sf.split(".dist-info/licenses/", 1)[1]
        elif sf.startswith("numpy/"):
            rel = sf
        else:
            continue
        d = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
        lic.append((d, f))
    seen = set()
    for d, f in sorted(lic, key=lambda x: len(x[0])):
        if d and not any(p.startswith(d) for p in paths):
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if not d:
            # ホイール同梱の他ソフト (OpenBLAS 等) の節は除き、NumPy 本体の条文だけ
            txt = txt.split("\n----", 1)[0].rstrip() + "\n"
        key = txt.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({"scope": d or "numpy/ (本体)", "file": str(f).replace(os.sep, "/"), "text": txt})
    return out


def build_html(traces, template):
    data = merge_traces(traces)
    np_paths = [f["path"] for f in data["files"] if f["kind"] == "np" and f["src"]]
    data["notices"] = numpy_license_notices(np_paths) if np_paths else []
    data["env"] = env_info()
    data["generated"] = time.strftime("%Y-%m-%d %H:%M")
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("/*__TRACE_DATA__*/null", js)
