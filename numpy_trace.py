#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 shin2225
"""
numpy_trace.py — NumPy がどこを参照して計算しているかを、層ごと・行ごとに VS Code 風ビューアで見るツール
https://github.com/shin22s-lab/py_structure1

  使い方:
    python numpy_trace.py my_script.py -o trace.html        # スクリプトをトレース
    python numpy_trace.py -c "import numpy as np; np.linalg.svd(np.ones((4,4)))" -o t.html
    python numpy_trace.py --demo -o demo.html               # 組み込みデモ

  要件: Python 3.12+（sys.monitoring）, NumPy 2.x 推奨。生成される HTML は単体で開けます。
  生成 HTML には NumPy のソースが含まれます（BSD-3-Clause。条文は HTML 内に埋め込み）。
  自分で再配布するときはその表記を消さないでください。
"""
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


# ---------------------------------------------------------------------------
TEMPLATE = r'''<!DOCTYPE html>
<!--
  NumPy Trace Explorer — generated by numpy_trace (MIT License)
  https://github.com/shin22s-lab/py_structure1
  This file may embed source code and docstrings from NumPy, which are
  Copyright (c) NumPy Developers and distributed under the BSD 3-Clause License
  (and, for some subpackages, additional permissive licenses). The applicable
  license texts are embedded in this file: open "?" (help) or 環境 > ライセンス.
-->
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NumPy Trace Explorer</title>
<style>
:root{
  --bg:#1e1e1e; --side:#252526; --act:#333333; --title:#3c3c3c; --tab:#2d2d2d; --border:#2b2b2b;
  --fg:#cccccc; --dim:#8b8b8b; --sel:#04395e; --selb:#0078d4; --hover:#2a2d2e; --status:#007acc;
  --L0:#e2c08d; --L1:#4fc1ff; --L2:#c586c0; --L3:#f0a35e; --Lb:#8a8a8a; --Ls:#7fb069;
  --mono: "Cascadia Code","JetBrains Mono",Consolas,"SF Mono",Menlo,"Courier New",monospace;
  --ui: -apple-system,"Segoe UI","Hiragino Kaku Gothic ProN","Hiragino Sans","Noto Sans JP",Meiryo,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:var(--ui);font-size:13px;overflow:hidden}
button{font-family:inherit}
#app{display:grid;height:100vh;height:100dvh;grid-template-rows:34px 1fr 22px;grid-template-columns:48px var(--sbw,330px) 4px 1fr}
#titlebar{grid-column:1/-1;background:var(--title);display:flex;align-items:center;gap:10px;padding:0 10px;border-bottom:1px solid #252526;user-select:none;min-width:0}
#titlebar .logo{display:flex;align-items:center;gap:6px;font-weight:600;color:#e7e7e7;white-space:nowrap}
#titlebar .logo i{display:inline-block;width:14px;height:14px;border-radius:3px;background:linear-gradient(135deg,var(--L1),var(--L2))}
.cmd{flex:1;display:flex;justify-content:center;min-width:0}
.cmd select{max-width:520px;width:100%;background:#252526;color:#ddd;border:1px solid #4a4a4a;border-radius:5px;height:24px;padding:0 8px;font-size:12px;font-family:var(--ui)}
.tbtn{background:transparent;border:1px solid transparent;color:#ccc;border-radius:4px;height:24px;padding:0 8px;cursor:pointer;font-size:12px;white-space:nowrap}
.tbtn:hover{background:#505050}
#activity{background:var(--act);display:flex;flex-direction:column;align-items:center;padding-top:4px}
.abtn{width:48px;height:48px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#858585;border-left:2px solid transparent;position:relative}
.abtn:hover{color:#e7e7e7}
.abtn.on{color:#e7e7e7;border-left-color:#e7e7e7}
.abtn svg{width:24px;height:24px}
#sidebar{background:var(--side);display:flex;flex-direction:column;min-width:0;overflow:hidden}
.side-h{height:35px;display:flex;align-items:center;padding:0 12px 0 20px;font-size:11px;letter-spacing:.04em;color:#bbb;text-transform:uppercase;flex:none;justify-content:space-between}
.side-h .acts{display:flex;gap:2px}
.ibtn{width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;border-radius:4px;cursor:pointer;color:#c5c5c5;background:transparent;border:0;font-size:13px}
.ibtn:hover{background:#3a3d41}
.ibtn.on{background:#3a3d41;outline:1px solid #555}
.filters{display:flex;flex-wrap:wrap;gap:4px;padding:2px 10px 6px 12px;flex:none}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:11px;padding:2px 7px;border-radius:10px;border:1px solid #444;cursor:pointer;color:#bbb;user-select:none;background:#2b2b2c}
.chip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.chip.off{opacity:.38;text-decoration:line-through}
.search{margin:0 10px 6px 12px;flex:none;display:flex}
.search input{flex:1;background:#3c3c3c;border:1px solid #3c3c3c;color:#ddd;height:24px;padding:0 6px;font-size:12px;outline:none;font-family:var(--ui);min-width:0}
.search input:focus{border-color:var(--selb)}
.sec-h{height:22px;display:flex;align-items:center;padding:0 8px;font-size:11px;font-weight:700;color:#ccc;cursor:pointer;flex:none;border-top:1px solid #333;user-select:none;text-transform:uppercase;gap:4px}
.sec-h .cnt{margin-left:auto;font-weight:400;color:var(--dim);text-transform:none}
#tree{flex:1;overflow:auto;position:relative;outline:none;min-height:60px}
#treeInner{position:relative}
.row{position:absolute;left:0;right:0;height:22px;display:flex;align-items:center;white-space:nowrap;cursor:pointer;padding-right:8px;font-size:13px}
.row:hover{background:var(--hover)}
.row.sel{background:var(--sel);outline:1px solid var(--selb);outline-offset:-1px}
.row.hit .nm{background:rgba(234,92,0,.35);border-radius:2px}
.ind{display:inline-block;width:12px;height:22px;flex:none;border-left:1px solid transparent}
#tree:hover .ind{border-left-color:#585858}
.tw{width:16px;flex:none;display:inline-flex;align-items:center;justify-content:center;color:#c5c5c5;font-size:10px}
.ico{flex:none;font:700 9px/1 var(--mono);width:22px;height:14px;display:inline-flex;align-items:center;justify-content:center;border-radius:3px;margin-right:5px;color:#1e1e1e}
.nm{overflow:hidden;text-overflow:ellipsis;min-width:0}
.ds{color:var(--dim);font-size:12px;margin-left:7px;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:1 1 0}
.tm{color:#9d9d9d;font-size:11px;margin-left:6px;font-family:var(--mono);flex:none}
.est{font-style:italic}
.kind-user .ico{background:var(--L0)} .kind-np .ico{background:var(--L1)} .kind-c .ico{background:var(--L2)}
.kind-blas .ico{background:var(--L3)} .kind-builtin .ico{background:var(--Lb)} .kind-stdlib .ico{background:var(--Ls)}
#files{max-height:30%;overflow:auto;flex:none}
.frow{height:22px;display:flex;align-items:center;padding:0 10px 0 22px;cursor:pointer;white-space:nowrap;font-size:13px;gap:6px}
.frow:hover{background:var(--hover)}
.frow .pth{color:var(--dim);font-size:11.5px;overflow:hidden;text-overflow:ellipsis}
.frow .n{margin-left:auto;color:var(--dim);font-size:11px;font-family:var(--mono)}
.pyico{color:#3d7bc3;font:700 10px var(--mono)}
#vsplit{cursor:col-resize;background:transparent}
#vsplit:hover{background:var(--selb)}
#main{display:grid;grid-template-rows:35px 22px 1fr 4px var(--pnh,240px);min-width:0;min-height:0;background:var(--bg)}
#tabs{display:flex;background:#252526;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin}
.tab{height:35px;display:flex;align-items:center;gap:6px;padding:0 6px 0 12px;background:var(--tab);color:#969696;border-right:1px solid #252526;cursor:pointer;white-space:nowrap;font-size:13px;flex:none;max-width:260px}
.tab .tn{overflow:hidden;text-overflow:ellipsis}
.tab.on{background:var(--bg);color:#fff;border-top:1px solid var(--selb)}
.tab .x{width:20px;height:20px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;opacity:0;font-size:14px}
.tab:hover .x,.tab.on .x{opacity:1}
.tab .x:hover{background:#444}
#cpath{display:flex;align-items:center;gap:2px;overflow:hidden;min-width:0;flex:1 1 auto}
#stepbar{display:flex;align-items:center;gap:10px;flex:none;margin-left:10px;font-size:12px;color:#bbb}
#stepbar .sb{display:inline-flex;align-items:center;gap:2px;background:#2a2a2a;border:1px solid #3c3c3c;border-radius:4px;padding:0 2px}
#stepbar .stl{padding:0 6px;white-space:nowrap} #stepbar .stl b{color:#ffd866}
#stepbar .cv-sum{color:#73c991;white-space:nowrap} #stepbar .stc{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
#crumbs{display:flex;align-items:center;gap:2px;padding:0 12px;color:#a9a9a9;font-size:12.5px;white-space:nowrap;overflow:hidden;background:var(--bg);box-shadow:0 3px 3px -3px #000 inset}
#crumbs span.s{color:#666;margin:0 3px}
#crumbs .c{cursor:pointer;border-radius:3px;padding:0 2px} #crumbs .c:hover{color:#fff;background:#333}
#editor{position:relative;overflow:hidden;min-height:0}
#code{position:absolute;inset:0 14px 0 0;overflow:auto;font-family:var(--mono);font-size:13px;line-height:19px;tab-size:4}
#ruler{position:absolute;top:0;right:0;bottom:0;width:14px;border-left:1px solid #333;background:#1e1e1e}
#ruler b{position:absolute;left:2px;right:2px;height:3px;border-radius:1px}
#ruler .rng{position:absolute;left:0;right:0;background:rgba(121,121,121,.25)}
.lines{padding-bottom:40vh;min-width:max-content}
.ln{display:flex;height:19px;white-space:pre}
.ln .g{width:18px;flex:none;display:flex;align-items:center;justify-content:center}
.ln .num{width:48px;flex:none;text-align:right;padding-right:16px;color:#6e7681;user-select:none}
.ln .hc{width:34px;flex:none;text-align:right;color:#73c991;font-size:10.5px;user-select:none;padding-right:2px}
.ln.cov .g{box-shadow:inset -2px 0 0 rgba(115,201,145,.35)}
.ln.ex .g{box-shadow:inset -3px 0 0 #73c991}
.ln.ex .num{color:#c8e6c9}
.ln.ex{cursor:pointer}
.ln.cur{background:rgba(255,216,102,.18)!important;outline:1px solid rgba(255,216,102,.45);outline-offset:-1px}
.ln.cur .num{color:#ffd866}
#ruler i.cv{position:absolute;left:0;width:3px;height:2px;background:rgba(115,201,145,.6)}
.row .lno{font-family:var(--mono);font-size:11px;color:#73c991;min-width:38px;text-align:right;margin-right:8px;flex:none}
.row .lsrc{font-family:var(--mono);font-size:12px;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:1 1 0;opacity:.9}
.row .rep{font-family:var(--mono);font-size:10.5px;color:#ffd866;margin-left:6px;flex:none}
.row.lrow_{color:#bbb}
.fsum{padding:2px 10px 2px 38px;font-size:11px;color:#73c991}
.ffn{height:21px;display:flex;align-items:center;padding:0 10px 0 38px;cursor:pointer;font-size:12px;white-space:nowrap;gap:6px}
.ffn:hover{background:var(--hover)} .ffn .nm{font-family:var(--mono);overflow:hidden;text-overflow:ellipsis} .ffn .n{margin-left:auto;color:var(--dim);font-size:11px}
.ffn .lno{font-family:var(--mono);font-size:10.5px;color:#73c991;min-width:64px}
.ln .src{padding-right:30px}
.ln.inr{background:rgba(255,255,255,.035)}
.ln.inr .num{color:#8f99a3}
.ln.def{background:rgba(79,193,255,.10)}
.ln.def .num{color:#c6c6c6}
.ln.site{background:rgba(255,213,0,.08)}
.ln.flash{animation:fl 1.6s ease-out}
@keyframes fl{0%{background:rgba(255,213,0,.45)}100%{background:transparent}}
.gm{width:9px;height:9px;border-radius:50%;cursor:pointer;box-shadow:0 0 0 1px rgba(0,0,0,.5)}
.gm.multi{border-radius:2px}
.ln.has .num{cursor:pointer;color:#c8c8c8}
.cmt{color:#6a9955}.str{color:#ce9178}.kw{color:#569cd6}.ctl{color:#c586c0}.num_{color:#b5cea8}.fn{color:#dcdcaa}.cls{color:#4ec9b0}.bi{color:#dcdcaa}.var{color:#9cdcfe}.slf{color:#569cd6;font-style:italic}.dec{color:#dcdcaa}.op{color:#d4d4d4}.ty{color:#4ec9b0}
.vdoc .src a{color:#3794ff}
#hsplit{cursor:row-resize;background:#2b2b2b}
#hsplit:hover{background:var(--selb)}
#panel{display:flex;flex-direction:column;min-height:0;border-top:1px solid #3c3c3c}
.ptabs{height:35px;display:flex;align-items:center;gap:2px;padding:0 10px;flex:none;overflow-x:auto}
.ptab{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#969696;padding:6px 8px;cursor:pointer;border-bottom:1px solid transparent;white-space:nowrap}
.ptab.on{color:#e7e7e7;border-bottom-color:#e7e7e7}
.pbody{flex:1;overflow:auto;padding:4px 16px 16px;min-height:0}
.lanes{position:relative;min-width:620px}
.lhead{display:grid;grid-template-columns:repeat(5,1fr);position:sticky;top:0;background:var(--bg);z-index:2;padding:4px 0 6px}
.lhead div{font-size:11px;color:#aaa;text-align:center;border-bottom:2px solid;padding-bottom:3px;margin:0 4px}
.lgrid{position:relative}
.lrow{display:grid;grid-template-columns:repeat(5,1fr);height:44px;align-items:center;position:relative;z-index:1}
.lbox{grid-row:1;margin:0 6px;height:36px;border-radius:5px;display:grid;grid-template-columns:auto 1fr;grid-template-rows:18px 14px;column-gap:6px;align-content:center;padding:0 8px;font-size:12px;cursor:pointer;white-space:nowrap;overflow:hidden;border:1px solid rgba(255,255,255,.1);background:#2a2a2a}
.lbox:hover{filter:brightness(1.25)}
.lbox .ico{grid-row:1/3;align-self:center;margin:0}
.lbox .d{color:#a0a0a0;font-size:10.5px;overflow:hidden;text-overflow:ellipsis;line-height:14px}
.lbox .nn{overflow:hidden;text-overflow:ellipsis;font-family:var(--mono);font-size:12px;line-height:18px}
.lbox.cur{outline:2px solid #fff}
.lbox.child{opacity:.62;border-style:dashed}
.lanebg{position:absolute;inset:0;display:grid;grid-template-columns:repeat(5,1fr);pointer-events:none}
.lanebg div{border-right:1px dashed #333}
svg.conn{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:0;overflow:visible}
.sumline{font-size:12px;color:#bbb;margin:6px 0 4px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.pill{display:inline-flex;align-items:center;gap:4px;padding:1px 7px;border-radius:9px;font-size:11px;color:#1e1e1e;font-weight:600}
table.kv{border-collapse:collapse;font-size:12.5px;margin-top:6px;width:100%}
table.kv td{padding:3px 10px 3px 0;vertical-align:top;border-bottom:1px solid #2a2a2a}
table.kv td:first-child{color:var(--dim);white-space:nowrap;width:1%}
table.kv code,.mono{font-family:var(--mono);font-size:12px}
a.lnk{color:#3794ff;cursor:pointer;text-decoration:none}
a.lnk:hover{text-decoration:underline}
pre.out{font-family:var(--mono);font-size:12px;white-space:pre-wrap;margin:6px 0;color:#ccc}
.err{color:#f48771}
#status{grid-column:1/-1;background:var(--status);color:#fff;display:flex;align-items:center;gap:14px;padding:0 10px;font-size:12px;white-space:nowrap;overflow:hidden}
#status .r{margin-left:auto;display:flex;gap:14px}
.sres{padding:0 0 0 12px;overflow:auto;flex:1}
.sres .g{padding:3px 8px 3px 0;cursor:pointer;display:flex;gap:6px;align-items:center;font-size:13px;white-space:nowrap}
.sres .g:hover{background:var(--hover)}
.sres .g .c{margin-left:auto;background:#4d4d4d;border-radius:9px;padding:0 6px;font-size:11px}
.stat{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}
.stat td:first-child{overflow:hidden;text-overflow:ellipsis}
.stat th{position:sticky;top:0;background:var(--side);text-align:left;color:#aaa;font-weight:400;padding:4px 6px;border-bottom:1px solid #3c3c3c}
.stat td{padding:3px 6px;white-space:nowrap;border-bottom:1px solid #2c2c2c;cursor:pointer}
.stat tr:hover td{background:var(--hover)}
.stat td.n{text-align:right;font-family:var(--mono)}
.bar{height:6px;border-radius:3px;display:inline-block;vertical-align:middle}
#menu{position:fixed;z-index:50;background:#252526;border:1px solid #454545;box-shadow:0 4px 16px rgba(0,0,0,.5);border-radius:5px;padding:4px 0;display:none;max-height:50vh;overflow:auto;min-width:220px}
#menu div{padding:4px 14px;cursor:pointer;font-size:12.5px;white-space:nowrap;display:flex;gap:6px;align-items:center}
#menu div:hover{background:var(--sel)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:60;display:none;align-items:center;justify-content:center;padding:16px}
#modal .box{background:#252526;border:1px solid #454545;border-radius:8px;max-width:760px;width:100%;max-height:86vh;overflow:auto;padding:18px 22px;line-height:1.65;font-size:13px}
#modal h2{font-size:16px;margin:0 0 8px} #modal h3{font-size:13.5px;margin:14px 0 4px;color:#e7e7e7}
#modal code{font-family:var(--mono);background:#1e1e1e;padding:0 4px;border-radius:3px}
#modal ul{margin:4px 0;padding-left:20px}
#mtabs{display:none}
.empty{color:var(--dim);padding:20px;font-size:13px;line-height:1.7}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:rgba(121,121,121,.4)}::-webkit-scrollbar-corner{background:transparent}
@media (max-width:820px){
  #app{grid-template-columns:1fr;grid-template-rows:40px 38px 1fr 22px}
  #activity,#vsplit{display:none} #hsplit{visibility:hidden}
  .lanes{min-width:0}
  #titlebar .logo span{display:none}
  .cmd select{height:28px;font-size:13px}
  #mtabs{display:flex;grid-column:1/-1;background:#252526;border-bottom:1px solid #333}
  #mtabs div{flex:1;text-align:center;line-height:38px;font-size:13px;color:#999;cursor:pointer;border-bottom:2px solid transparent}
  #mtabs div.on{color:#fff;border-bottom-color:var(--selb)}
  #sidebar,#main{grid-column:1/-1;grid-row:3}
  #main{grid-template-rows:35px 22px 1fr 0 0}
  body.m-tree #main{display:none} body.m-code #sidebar,body.m-panel #sidebar{display:none}
  body.m-panel #main{grid-template-rows:0 0 0 0 1fr} body.m-panel #tabs,body.m-panel #crumbs,body.m-panel #editor{visibility:hidden}
  body.m-code #panel{display:none}
  .row{height:28px;font-size:14px} .ind{height:28px;width:10px}
  #status .hide-m{display:none}
  #files{max-height:22%}
  .ln .num{width:38px;padding-right:8px} .ln .g{width:14px} .ln .hc{width:26px}
  #stepbar .cv-sum,#stepbar .stc{display:none}
}
</style>
</head>
<body class="m-tree">
<div id="app">
  <div id="titlebar">
    <div class="logo"><i></i><span>NumPy Trace Explorer</span></div>
    <div class="cmd"><select id="scn" title="トレースを選択"></select></div>
    <button class="tbtn" id="helpBtn" title="使い方">?</button>
  </div>
  <div id="mtabs"><div data-m="tree" class="on">ツリー</div><div data-m="code">コード</div><div data-m="panel">層ビュー</div></div>
  <div id="activity">
    <div class="abtn on" data-v="tree" title="呼び出しツリー"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 5h6M7 5v14M7 12h6M7 19h6"/><rect x="13" y="9.5" width="7" height="5" rx="1"/><rect x="13" y="16.5" width="7" height="5" rx="1"/><rect x="2" y="2.5" width="9" height="5" rx="1"/></svg></div>
    <div class="abtn" data-v="search" title="関数検索"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l6 6"/></svg></div>
    <div class="abtn" data-v="stats" title="ホットスポット / 層の統計"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg></div>
  </div>
  <div id="sidebar"></div>
  <div id="vsplit"></div>
  <div id="main">
    <div id="tabs"></div>
    <div id="crumbs"><div id="cpath"></div><div id="stepbar"></div></div>
    <div id="editor"><div id="code"></div><div id="ruler"></div></div>
    <div id="hsplit"></div>
    <div id="panel">
      <div class="ptabs">
        <div class="ptab on" data-p="lanes">層ビュー</div>
        <div class="ptab" data-p="detail">詳細</div>
        <div class="ptab" data-p="out">出力</div>
        <div class="ptab" data-p="env">環境</div>
      </div>
      <div class="pbody" id="pbody"></div>
    </div>
  </div>
  <div id="status"></div>
</div>
<div id="menu"></div>
<div id="modal"><div class="box" id="modalBox"></div></div>
<script>
"use strict";
const DATA = /*__TRACE_DATA__*/null;

const LAYERS = [
  {k:"user", L:"L0", name:"ユーザーコード", col:"var(--L0)", ico:"py"},
  {k:"np", L:"L1", name:"NumPy Python層", col:"var(--L1)", ico:"np"},
  {k:"c", L:"L2", name:"NumPy C拡張", col:"var(--L2)", ico:"C"},
  {k:"blas", L:"L3", name:"ネイティブ (BLAS/LAPACK等)", col:"var(--L3)", ico:"⚙"},
  {k:"stdlib", L:"py", name:"標準ライブラリ", col:"var(--Ls)", ico:"std"},
  {k:"builtin", L:"py", name:"組み込み関数", col:"var(--Lb)", ico:"ƒ"},
];
const LK = Object.fromEntries(LAYERS.map(l=>[l.k,l]));
const LANE = {user:0, np:1, c:2, blas:3, stdlib:4, builtin:4};
const LANES = [["L0 ユーザー","var(--L0)"],["L1 NumPy(Python)","var(--L1)"],["L2 NumPy(C拡張)","var(--L2)"],["L3 ネイティブ","var(--L3)"],["標準/組み込み","var(--Lb)"]];

const $ = s=>document.querySelector(s);
const esc = s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmtT = ns=>{ if(ns==null) return ""; if(ns<1e3) return ns+"ns"; if(ns<1e6) return (ns/1e3).toFixed(ns<1e4?1:0)+"µs"; if(ns<1e9) return (ns/1e6).toFixed(ns<1e7?2:1)+"ms"; return (ns/1e9).toFixed(2)+"s"; };
const basename = p=>p.split("/").pop();
const isMobile = ()=>matchMedia("(max-width:820px)").matches;

const S = { ti:0, T:null, kids:null, hidden:new Set(["builtin","stdlib"]), expanded:new Set(), sel:-1, view:"tree",
  rows:[], vk:new Map(), ti_:new Map(), hl:new Map(), cov:new Map(), showLines:true, selLine:null, cursor:0, tabs:[], tab:null, ptab:"lanes", query:"", hits:null, fileCache:new Map(), marks:[] };

/* ---------------- data prep ---------------- */
function prepTrace(ti){
  const T = DATA.traces[ti];
  if(!T._prep){
    const N = T.nodes, kids = new Array(N.length);
    for(let i=0;i<N.length;i++) kids[i]=[];
    const roots=[];
    for(const n of N){ if(n.p>=0) kids[n.p].push(n.i); else roots.push(n.i); }
    const self = new Float64Array(N.length);
    for(const n of N){ const d=(n.t1??n.t0)-n.t0; n.dur=d; }
    for(const n of N){ let s=n.dur; if(!n.est) for(const c of kids[n.i]) if(!N[c].est) s-=N[c].dur; n.self=Math.max(0,s); }
    const depth = new Int32Array(N.length);
    for(const n of N) depth[n.i] = n.p>=0 ? depth[n.p]+1 : 0;
    T._prep={kids, roots, depth};
  }
  return T;
}
function visible(n){ return !S.hidden.has(n.k); }
function visKids(i){
  if(S.vk.has(i)) return S.vk.get(i);
  const T=S.T, N=T.nodes, src = i<0 ? T._prep.roots : T._prep.kids[i];
  const out=[];
  for(const c of src){ if(visible(N[c])) out.push(c); else out.push(...visKids(c)); }
  S.vk.set(i,out); return out;
}
function visParent(i){ const N=S.T.nodes; let p=N[i].p; while(p>=0 && !visible(N[p])) p=N[p].p; return p; }
function chain(i){ const out=[]; const N=S.T.nodes; while(i>=0){ out.push(i); i=N[i].p; } return out.reverse(); }

/* ---------------- python highlighter ---------------- */
const KW_CTL = new Set("if elif else for while return try except finally raise with break continue import from as yield await pass assert del match case".split(" "));
const KW = new Set("def class lambda and or not in is None True False global nonlocal async".split(" "));
const TYPES = new Set("int float str list dict tuple set frozenset bool object type bytes complex slice range Exception TypeError ValueError KeyError IndexError AttributeError RuntimeError NotImplementedError NotImplemented Ellipsis".split(" "));
const BUILTINS = new Set("len isinstance issubclass getattr setattr hasattr print min max sum abs any all map filter zip enumerate sorted reversed iter next super repr id hash callable round divmod pow vars locals globals open format".split(" "));
const PYRE = /(#[^\n]*)|([rRbBuUfF]{0,2}(?:"{3}[\s\S]*?"{3}|'{3}[\s\S]*?'{3}))|([rRbBuUfF]{0,2}(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(\b(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?j?)\b|\.\d+(?:[eE][+-]?\d+)?)|(@[A-Za-z_][\w.]*)|([A-Za-z_][A-Za-z0-9_]*)|(\n)|([^\S\n]+)|(.)/gy;
function highlightPy(src){
  const lines=[]; let cur=""; let prev=""; PYRE.lastIndex=0; let m;
  const push=(cls,text)=>{
    const parts=text.split("\n");
    parts.forEach((p,k)=>{ if(k>0){ lines.push(cur); cur=""; } if(p) cur += cls ? `<span class="${cls}">${esc(p)}</span>` : esc(p); });
  };
  while(PYRE.lastIndex < src.length && (m=PYRE.exec(src))){
    if(m[1]) push("cmt",m[1]);
    else if(m[2]) push("str",m[2]);
    else if(m[3]) push("str",m[3]);
    else if(m[4]) push("num_",m[4]);
    else if(m[5]) push("dec",m[5]);
    else if(m[6]){
      const w=m[6]; let c="var";
      if(KW_CTL.has(w)) c="ctl"; else if(KW.has(w)) c="kw";
      else if(prev==="def") c="fn"; else if(prev==="class") c="cls";
      else if(w==="self"||w==="cls") c="slf";
      else if(TYPES.has(w)) c="ty";
      else { let j=PYRE.lastIndex; while(src[j]===" ") j++; if(src[j]==="(") c = BUILTINS.has(w)?"bi":"fn"; else if(/^[A-Z][A-Z0-9_]+$/.test(w)) c="kw"; }
      push(c,w); prev=w; continue;
    }
    else if(m[7]){ push(null,"\n"); }
    else if(m[8]){ push(null,m[8]); continue; }
    else push("op",m[9]);
    prev="";
  }
  lines.push(cur);
  return lines;
}

/* ---------------- tree ---------------- */
const ROWH = ()=> isMobile()?28:22;
/* ツリーの子項目: 行表示 ON のときは「実行行」と「呼び出し」を実行順に並べる */
function treeItems(i){
  const key=(S.showLines?"L":"C")+i;
  if(S.ti_.has(key)) return S.ti_.get(key);
  const N=S.T.nodes, n=i>=0?N[i]:null; let out;
  if(S.showLines && n && n.q && n.q.length){
    out=[]; const seen=new Set();
    n.q.forEach((v,qp)=>{
      if(v>0){ const last=out[out.length-1]; if(last && last.t==="l" && last.L===v) last.rep++; else out.push({t:"l",i,L:v,rep:1,qp}); }
      else { const c=-v-1; if(visible(N[c])){ out.push({t:"n",i:c}); seen.add(c); } else for(const x of visKids(c)){ out.push({t:"n",i:x}); seen.add(x); } }
    });
    for(const c of visKids(i)) if(!seen.has(c)) out.push({t:"n",i:c});
  } else out=visKids(i).map(c=>({t:"n",i:c}));
  S.ti_.set(key,out); return out;
}
function rebuildRows(){
  const rows=[];
  const walk=(list,d)=>{ for(const it of list){ rows.push({...it,d}); if(it.t==="n" && S.expanded.has(it.i)){ const k=treeItems(it.i); if(k.length) walk(k,d+1);} } };
  walk(treeItems(-1),0);
  S.rows=rows; S.rowIndex=new Map(); S.lineRowIndex=new Map();
  rows.forEach((r,ix)=>{ if(r.t==="n") S.rowIndex.set(r.i,ix); else S.lineRowIndex.set(r.i+":"+r.qp,ix); });
  renderTree();
}
function hlLines(fid){ if(!S.hl.has(fid)) S.hl.set(fid, highlightPy(DATA.files[fid].src||"")); return S.hl.get(fid); }
function nodeDesc(n){
  const F=DATA.files;
  if(n.k==="np"||n.k==="user"||n.k==="stdlib"){ const f=F[n.f]; return f? basename(f.path)+":"+n.l : ""; }
  if(n.k==="blas") return (n.m||"")+"  (推定)";
  if(n.k==="c"){ const ci=S.T.cinfo[n.c]||{}; return (ci.type&&ci.type.includes("ufunc")?ci.type+" · ":"")+(n.m||"").replace("numpy.",""); }
  return n.m||"";
}
function nodeName(n){
  if(n.k==="c" && S.T.cinfo[n.c] && S.T.cinfo[n.c].type==="__array_function__ ディスパッチャ") return n.n.replace(/^numpy\./,"np.")+" ⟨dispatch⟩";
  return n.n + (n.rs? `  ×${n.rs+1}`:"");
}
function renderTree(){
  const el=$("#tree"); if(!el) return;
  const inner=$("#treeInner"); const h=ROWH();
  inner.style.height = (S.rows.length*h+8)+"px";
  const top=el.scrollTop, H=el.clientHeight||400;
  const a=Math.max(0,Math.floor(top/h)-10), b=Math.min(S.rows.length, Math.ceil((top+H)/h)+10);
  const N=S.T.nodes; let html="";
  for(let r=a;r<b;r++){
    const row=S.rows[r], i=row.i, d=row.d, n=N[i];
    let ind=""; for(let k=0;k<d;k++) ind+='<span class="ind"></span>';
    if(row.t==="l"){
      const src=(hlLines(n.f)[row.L-1]||"").trimStart();
      const on = S.selLine && S.selLine.i===i && S.selLine.qp===row.qp;
      html+=`<div class="row lrow_${on?" sel":""}" style="top:${r*h}px" data-i="${i}" data-qp="${row.qp}" data-l="${row.L}">${ind}<span class="tw"></span><span class="lno">${row.L}</span><span class="lsrc">${src||" "}</span>${row.rep>1?`<span class="rep">×${row.rep}</span>`:""}</div>`;
      continue;
    }
    const L=LK[n.k]||LK.builtin;
    const kc=treeItems(i).length, ex=S.expanded.has(i);
    const hit = S.hits && S.hits.has(i) ? " hit":"";
    html+=`<div class="row kind-${n.k}${i===S.sel&&!S.selLine?" sel":""}${hit}" style="top:${r*h}px" data-i="${i}">${ind}<span class="tw">${kc?(ex?"▾":"▸"):""}</span><span class="ico">${L.ico}</span><span class="nm${n.est?" est":""}">${esc(nodeName(n))}</span><span class="ds">${esc(nodeDesc(n))}</span><span class="tm">${n.est?"":fmtT(n.dur)}</span></div>`;
  }
  inner.innerHTML=html;
}
function toggle(i,force){
  const has=S.expanded.has(i); const want = force==null ? !has : force;
  if(want) S.expanded.add(i); else S.expanded.delete(i);
  rebuildRows();
}
function scrollRow(ix){ const el=$("#tree"); if(!el||ix==null) return; const y=ix*ROWH(); if(y<el.scrollTop || y>el.scrollTop+el.clientHeight-30) el.scrollTop=y-el.clientHeight/3; renderTree(); }
function reveal(i){
  const c=chain(i); const N=S.T.nodes;
  for(const a of c.slice(0,-1)) if(visible(N[a])) S.expanded.add(a);
  rebuildRows();
  let t=i; while(t>=0 && !S.rowIndex.has(t)) t=N[t].p;
  if(t>=0) scrollRow(S.rowIndex.get(t));
}
function defaultExpand(){
  S.expanded.clear(); const N=S.T.nodes;
  let any=false;
  for(const n of N) if(n.k==="blas"){ any=true; let p=n.p; while(p>=0){ S.expanded.add(p); p=N[p].p; } }
  const walk=(list,d)=>{ for(const i of list){ if(d<(any?1:2)) { S.expanded.add(i); walk(visKids(i),d+1);} } };
  walk(visKids(-1),0);
}

/* ---------------- sidebar views ---------------- */
function renderSidebar(){
  const sb=$("#sidebar"); const T=S.T;
  if(S.view==="tree"){
    const cnt={}; for(const n of T.nodes) cnt[n.k]=(cnt[n.k]||0)+1;
    sb.innerHTML=`<div class="side-h"><span>呼び出しツリー</span><span class="acts">
      <button class="ibtn${S.showLines?" on":""}" id="bLines" title="実行した行（何行目か）をツリーに表示">≡</button>
      <button class="ibtn" id="bPath" title="ネイティブ層への経路だけ展開">⤓</button>
      <button class="ibtn" id="bExpand" title="すべて展開">⊞</button>
      <button class="ibtn" id="bCollapse" title="すべて折りたたむ">⊟</button></span></div>
      <div class="filters">${LAYERS.map(l=>`<span class="chip${S.hidden.has(l.k)?" off":""}" data-k="${l.k}" title="${l.name} の表示切替"><i style="background:${l.col}"></i>${l.L!=="py"?l.L+" ":""}${l.name} <span style="color:#888">${cnt[l.k]||0}</span></span>`).join("")}</div>
      <div class="search"><input id="q" placeholder="ツリー内を検索 (例: solve, reduce, gemm)" value="${esc(S.query)}"></div>
      <div id="tree" tabindex="0"><div id="treeInner"></div></div>
      <div class="sec-h" id="fh">▾ 参照されたモジュール（実行行 / 全行） <span class="cnt" id="fcnt"></span></div>
      <div id="files"></div>`;
    $("#tree").addEventListener("scroll",()=>renderTree(),{passive:true});
    $("#tree").addEventListener("click",onTreeClick);
    $("#tree").addEventListener("keydown",onTreeKey);
    sb.querySelectorAll(".chip").forEach(c=>c.onclick=()=>{ const k=c.dataset.k; if(S.hidden.has(k)) S.hidden.delete(k); else S.hidden.add(k); S.vk.clear(); S.ti_.clear(); renderSidebar(); updateSel(false); });
    $("#bExpand").onclick=()=>{ T.nodes.forEach(n=>S.expanded.add(n.i)); rebuildRows(); };
    $("#bCollapse").onclick=()=>{ S.expanded.clear(); rebuildRows(); };
    $("#bPath").onclick=()=>{ defaultExpand(); rebuildRows(); };
    $("#bLines").onclick=()=>{ S.showLines=!S.showLines; S.ti_.clear(); $("#bLines").classList.toggle("on",S.showLines); rebuildRows(); };
    $("#q").oninput=e=>{ S.query=e.target.value; applyQuery(); };
    $("#fh").onclick=()=>{ const f=$("#files"); f.style.display=f.style.display==="none"?"":"none"; };
    renderFiles(); applyQuery(true); rebuildRows();
  } else if(S.view==="search"){
    sb.innerHTML=`<div class="side-h"><span>検索</span></div><div class="search"><input id="q2" placeholder="関数名・モジュール名で検索" value="${esc(S.query)}"></div><div class="sres" id="sres"></div>`;
    const run=()=>{ S.query=$("#q2").value; const q=S.query.trim().toLowerCase(); const box=$("#sres"); if(!q){ box.innerHTML='<div class="empty">名前の一部を入力すると、このトレース内で呼ばれた関数を種類ごとにまとめて表示します。</div>'; return; }
      const g=new Map(); for(const n of T.nodes){ const key=n.k+"|"+n.n+"|"+(n.m||""); if((n.n+" "+(n.m||"")).toLowerCase().includes(q)){ if(!g.has(key)) g.set(key,[]); g.get(key).push(n.i);} }
      box.innerHTML = g.size? [...g.entries()].sort((a,b)=>b[1].length-a[1].length).slice(0,400).map(([k,ids])=>{ const n=T.nodes[ids[0]]; return `<div class="g kind-${n.k}" data-i="${ids[0]}" data-all="${ids.slice(0,500).join(",")}"><span class="ico">${(LK[n.k]||LK.builtin).ico}</span><span>${esc(n.n)}</span><span class="ds">${esc(n.m||"")}</span><span class="c">${ids.length}</span></div>`; }).join("") : '<div class="empty">見つかりません（フィルタで非表示の層も検索対象です）</div>';
      box.querySelectorAll(".g").forEach(e=>e.onclick=()=>{ const i=+e.dataset.i; const k=T.nodes[i].k; if(S.hidden.has(k)){ S.hidden.delete(k); S.vk.clear(); } S.hits=new Set(e.dataset.all.split(",").map(Number)); setView("tree"); select(i,true); });
    };
    $("#q2").oninput=run; run(); setTimeout(()=>{ const q=$("#q2"); if(q) q.focus(); },0);
  } else {
    const agg=new Map(); const lay={};
    for(const n of T.nodes){ if(n.est) continue; lay[n.k]=(lay[n.k]||0)+n.self; const key=n.k+"|"+n.n+"|"+(n.m||""); let a=agg.get(key); if(!a){ a={k:n.k,n:n.n,m:n.m,c:0,self:0,tot:0,first:n.i}; agg.set(key,a);} a.c++; a.self+=n.self; a.tot+=n.dur; }
    const totalSelf=Object.values(lay).reduce((a,b)=>a+b,0)||1;
    const arr=[...agg.values()].sort((a,b)=>b.self-a.self).slice(0,300);
    const mx=arr.length?arr[0].self:1;
    sb.innerHTML=`<div class="side-h"><span>ホットスポット</span></div>
      <div style="padding:0 12px 8px">${LAYERS.map(l=>{const v=lay[l.k]||0; return `<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin:3px 0"><span style="width:118px;color:#bbb">${l.L!=="py"?l.L+" ":""}${l.name}</span><span class="bar" style="width:${Math.max(2,120*v/totalSelf)}px;background:${l.col}"></span><span class="mono" style="color:#999">${(100*v/totalSelf).toFixed(1)}%</span></div>`}).join("")}
      <div style="font-size:11px;color:#888;margin-top:4px">自己時間（子呼び出しを除く）の層別割合。トレースのオーバーヘッドを含む目安値です。</div></div>
      <div style="overflow:auto;flex:1"><table class="stat"><tr><th>関数</th><th style="width:42px">回数</th><th style="width:96px">自己時間</th></tr>${arr.map(a=>`<tr data-q="${esc(a.n)}" data-i="${a.first}" class="kind-${a.k}"><td><span class="ico" style="display:inline-flex">${(LK[a.k]||LK.builtin).ico}</span>${esc(a.n)}<div style="color:#777;font-size:11px;padding-left:27px">${esc(a.m||"")}</div></td><td class="n">${a.c}</td><td class="n"><span class="bar" style="width:${Math.max(2,50*a.self/mx)}px;background:${LK[a.k].col}"></span> ${fmtT(a.self)}</td></tr>`).join("")}</table></div>`;
    sb.querySelectorAll("tr[data-q]").forEach(tr=>tr.onclick=()=>{ const i=+tr.dataset.i; const k=T.nodes[i].k; if(S.hidden.has(k)){ S.hidden.delete(k); S.vk.clear(); } S.query=tr.dataset.q; setView("tree"); select(i,true); });
  }
}
function applyQuery(silent){
  const q=S.query.trim().toLowerCase(); const N=S.T.nodes;
  if(!q){ S.hits=null; if(!silent) renderTree(); return; }
  S.hits=new Set(); for(const n of N) if((n.n+" "+(n.m||"")).toLowerCase().includes(q)) S.hits.add(n.i);
  if(!silent){
    let c=0; for(const i of S.hits){ if(c++>200) break; for(const a of chain(i).slice(0,-1)) if(visible(N[a])) S.expanded.add(a); }
    rebuildRows();
  }
}
function renderFiles(){
  const F=DATA.files, N=S.T.nodes; const cnt=new Map();
  for(const n of N) if(n.f>=0 && (n.k==="np"||n.k==="user")) cnt.set(n.f,(cnt.get(n.f)||0)+1);
  const ids=[...cnt.keys()].sort((a,b)=>F[a].path.localeCompare(F[b].path));
  $("#fcnt").textContent=ids.length;
  S.fopen=S.fopen||new Set();
  let h="";
  for(const f of ids){
    const op=S.fopen.has(f), cov=fileCov(f), total=(F[f].src||"").split("\n").length;
    h+=`<div class="frow" data-f="${f}" title="${esc(F[f].path)}"><span class="tw ftw" data-t="${f}">${op?"▾":"▸"}</span><span class="pyico" style="color:${F[f].kind==="user"?"var(--L0)":"var(--L1)"}">py</span><span>${esc(basename(F[f].path))}</span><span class="pth">${esc(F[f].path.split("/").slice(0,-1).join("/"))}</span><span class="n">${cov.lines.size}/${total}行</span></div>`;
    if(op){
      h+=`<div class="fsum">実行された行 ${cov.lines.size} / ${total}（${(100*cov.lines.size/Math.max(1,total)).toFixed(1)}%）· 呼ばれた関数 ${cov.funcs.length}</div>`;
      for(const fn of cov.funcs){ const ex=[...cov.lines.keys()].filter(L=>L>=fn.l&&L<=fn.e).length;
        h+=`<div class="ffn" data-i="${fn.first}"><span class="lno">${fn.l}–${fn.e}</span><span class="nm">${esc(fn.n)}</span><span class="n">${fn.c}回 · ${ex}行</span></div>`; }
    }
  }
  $("#files").innerHTML=h;
  $("#files").querySelectorAll(".frow").forEach(r=>r.onclick=e=>{ const f=+r.dataset.f; if(e.target.classList.contains("ftw")){ if(S.fopen.has(f)) S.fopen.delete(f); else S.fopen.add(f); renderFiles(); return; } openFile(f); if(isMobile()) setMobile("code"); });
  $("#files").querySelectorAll(".ffn").forEach(r=>r.onclick=()=>{ select(+r.dataset.i,true); if(isMobile()) setMobile("code"); });
}
function onTreeClick(e){
  const r=e.target.closest(".row"); if(!r) return; const i=+r.dataset.i;
  if(r.dataset.qp!=null){ selectLine(i,+r.dataset.qp); return; }
  if(e.target.classList.contains("tw")){ toggle(i); return; }
  if(i===S.sel && !S.selLine && treeItems(i).length) toggle(i);
  select(i,false);
}
function curRowIx(){ return S.selLine ? S.lineRowIndex.get(S.selLine.i+":"+S.selLine.qp) : S.rowIndex.get(S.sel); }
function onTreeKey(e){
  if(!S.rows.length) return; let ix=curRowIx(); if(ix==null) ix=0;
  const go=j=>{ j=Math.max(0,Math.min(S.rows.length-1,j)); const r=S.rows[j]; if(r.t==="l") selectLine(r.i,r.qp); else select(r.i,true); scrollRow(j); e.preventDefault(); };
  if(S.selLine && (e.key==="ArrowRight"||e.key==="ArrowLeft"||e.key==="Enter")){ if(e.key==="ArrowLeft"){ select(S.selLine.i,true); e.preventDefault(); return; } go(ix+1); return; }
  if(e.key==="ArrowDown") go(ix+1); else if(e.key==="ArrowUp") go(ix-1);
  else if(e.key==="ArrowRight"){ if(treeItems(S.sel).length && !S.expanded.has(S.sel)) toggle(S.sel,true); else go(ix+1); e.preventDefault(); }
  else if(e.key==="ArrowLeft"){ if(S.expanded.has(S.sel)) toggle(S.sel,false); else { const p=visParent(S.sel); if(p>=0) select(p,true);} e.preventDefault(); }
  else if(e.key==="Enter"){ toggle(S.sel); e.preventDefault(); }
}

/* ---------------- editor ---------------- */
function fileView(fid){
  if(S.fileCache.has(fid)) return S.fileCache.get(fid);
  const f=DATA.files[fid]; const lines=highlightPy(f.src||(f.kind==="stdlib"?"# Python 標準ライブラリのソースはこの HTML に含めていません（--no-stdlib-source）":"# (ソースを取得できませんでした)"));
  const wrap=document.createElement("div"); wrap.className="lines";
  S.hl.set(fid,lines);
  wrap.innerHTML=lines.map((h,k)=>`<div class="ln" data-l="${k+1}"><span class="g"></span><span class="hc"></span><span class="num">${k+1}</span><span class="src">${h||" "}</span></div>`).join("");
  wrap.addEventListener("click",onCodeClick);
  const v={el:wrap, lines:wrap.children, n:lines.length, scroll:0};
  S.fileCache.set(fid,v); return v;
}
function openTab(tab){
  const key=tab.key; if(!S.tabs.find(t=>t.key===key)){ S.tabs.push(tab); if(S.tabs.length>10) S.tabs.shift(); }
  else { const t=S.tabs.find(t=>t.key===key); Object.assign(t,tab); }
  if(S.tab && S.tab.view) S.tab.view.scroll=$("#code").scrollTop;
  S.tab=S.tabs.find(t=>t.key===key);
  renderTabs(); renderEditor();
}
function renderTabs(){
  $("#tabs").innerHTML=S.tabs.map(t=>`<div class="tab${t===S.tab?" on":""}" data-k="${esc(t.key)}" title="${esc(t.title)}"><span class="${t.c?"":"pyico"}" style="${t.c?"color:var(--L2);font:700 10px var(--mono)":t.user?"color:var(--L0)":""}">${t.c?(t.blas?"⚙":"C"):"py"}</span><span class="tn">${esc(t.label)}</span><span class="x" data-x="1">×</span></div>`).join("");
  $("#tabs").querySelectorAll(".tab").forEach(el=>el.onclick=e=>{ const t=S.tabs.find(t=>t.key===el.dataset.k); if(e.target.dataset.x){ S.tabs=S.tabs.filter(x=>x!==t); if(S.tab===t){ S.tab=S.tabs[S.tabs.length-1]||null; } renderTabs(); renderEditor(); return; } openTab(t); });
  const on=$("#tabs .tab.on"); if(on) on.scrollIntoView({block:"nearest",inline:"nearest"});
}
function renderEditor(){
  const code=$("#code"); code.innerHTML=""; $("#ruler").innerHTML="";
  const t=S.tab;
  if(!t){ code.innerHTML='<div class="empty">左のツリーからノードを選ぶと、そのソースがここに開きます。</div>'; $("#cpath").innerHTML=""; $("#stepbar").innerHTML=""; return; }
  if(t.c){ code.appendChild(t.view.el); code.scrollTop=0; $("#stepbar").innerHTML=""; }
  else { const v=fileView(t.fid); t.view=v; code.appendChild(v.el); code.scrollTop=v.scroll||0; }
  renderCrumbs(); applyMarks();
}
function renderCrumbs(){
  const t=S.tab, N=S.T.nodes; let parts=[];
  if(t.c){ const n=N[t.node]; parts=[["C 拡張",null], [n.m||"",null], [n.n,t.node]]; }
  else { const f=DATA.files[t.fid]; parts=f.path.split("/").map(p=>[p,null]); const n=S.sel>=0?N[S.sel]:null; if(n && n.f===t.fid) n.n.split(".").forEach(p=>parts.push([p,S.sel])); }
  $("#cpath").innerHTML=parts.map((p,k)=>`${k?'<span class="s">›</span>':""}<span class="c" ${p[1]!=null?`data-i="${p[1]}"`:""}>${esc(p[0])}</span>`).join("");
  $("#cpath").querySelectorAll(".c[data-i]").forEach(c=>c.onclick=()=>gotoNodeLine(+c.dataset.i));
}
function clearMarks(){ for(const [el,cls] of S.marks){ if(cls==="g") el.innerHTML=""; else if(cls==="hc") el.textContent=""; else el.classList.remove(cls); } S.marks=[]; }
function fileCov(fid){
  if(S.cov.has(fid)) return S.cov.get(fid);
  const m=new Map(), fn=new Map();
  for(const n of S.T.nodes){ if(n.f!==fid) continue;
    if(n.h) for(const [L,c] of Object.entries(n.h)) m.set(+L,(m.get(+L)||0)+c);
    if(n.k==="np"||n.k==="user"){ const k=n.n+"@"+n.l; const o=fn.get(k)||{n:n.n,l:n.l,e:n.e,c:0,first:n.i,k:n.k}; o.c++; fn.set(k,o); } }
  const r={lines:m, funcs:[...fn.values()].sort((a,b)=>a.l-b.l)}; S.cov.set(fid,r); return r;
}
function applyMarks(){
  clearMarks(); const t=S.tab; if(!t || t.c) return;
  const v=t.view, N=S.T.nodes, ruler=$("#ruler"); ruler.innerHTML="";
  const n = S.sel>=0 ? N[S.sel] : null;
  const total=v.n||1, rh=ruler.clientHeight||300;
  const tick=(line,col)=>{ const b=document.createElement("b"); b.style.top=(rh*(line-1)/total)+"px"; b.style.background=col; ruler.appendChild(b); };
  if(n && n.f===t.fid && (n.k==="np"||n.k==="user"||n.k==="stdlib")){
    const a=n.l||1, e=Math.max(a,n.e||a);
    if(n.i!==S.T._prep.roots[0] || n.k!=="user") { for(let L=a;L<=e && L<=v.n;L++){ const el=v.lines[L-1]; el.classList.add("inr"); S.marks.push([el,"inr"]); } }
    const d=v.lines[a-1]; if(d){ d.classList.add("def"); S.marks.push([d,"def"]); }
    const rg=document.createElement("div"); rg.className="rng"; rg.style.top=(rh*(a-1)/total)+"px"; rg.style.height=Math.max(3,rh*(e-a+1)/total)+"px"; ruler.appendChild(rg);
    const byLine=new Map();
    for(const c of visKids(n.i)){ const cn=N[c]; if(cn.cf===t.fid && cn.cl){ if(!byLine.has(cn.cl)) byLine.set(cn.cl,[]); byLine.get(cn.cl).push(c);} }
    for(const [L,ids] of byLine){ const el=v.lines[L-1]; if(!el) continue; el.classList.add("has"); S.marks.push([el,"has"]);
      const g=el.firstElementChild; const k=N[ids[0]].k; const deep=ids.map(i=>LANE[N[i].k]).reduce((a,b)=>Math.max(a,b),0);
      const kk = ["user","np","c","blas","stdlib"][deep]||k;
      g.innerHTML=`<span class="gm${ids.length>1?" multi":""}" style="background:${LK[kk].col}" title="${esc(ids.map(i=>N[i].n).join("\n"))}"></span>`; S.marks.push([g,"g"]);
      tick(L,LK[kk].col); }
  }
  if(t.site){ const el=v.lines[t.site-1]; if(el){ el.classList.add("site"); S.marks.push([el,"site"]); tick(t.site,"#ffd500"); } }
  // このトレースでファイル内の実行された行（薄い緑）と、選択ノードが実行した行（濃い緑＋回数）
  const cov=fileCov(t.fid);
  for(const [L] of cov.lines){ const el=v.lines[L-1]; if(!el) continue; el.classList.add("cov"); S.marks.push([el,"cov"]); }
  if(n && n.f===t.fid && n.h){ for(const [Ls,c] of Object.entries(n.h)){ const L=+Ls; const el=v.lines[L-1]; if(!el) continue; el.classList.add("ex"); S.marks.push([el,"ex"]); const hc=el.children[1]; hc.textContent=c>1?"×"+c:"✓"; S.marks.push([hc,"hc"]); } }
  if(cov.lines.size){ const fr=document.createDocumentFragment(); let prev=-9; for(const L of [...cov.lines.keys()].sort((a,b)=>a-b)){ const y=Math.round(rh*(L-1)/total); if(y===prev) continue; prev=y; const b=document.createElement("i"); b.className="cv"; b.style.top=y+"px"; fr.appendChild(b);} ruler.appendChild(fr); }
  if(S.cursor && S.selLine && n && n.f===t.fid){ const el=v.lines[S.cursor-1]; if(el){ el.classList.add("cur"); S.marks.push([el,"cur"]); tick(S.cursor,"#ffffff"); } }
  renderStepBar();
}
function lineSteps(i){ const n=S.T.nodes[i]; if(!n||!n.q) return []; const out=[]; n.q.forEach((v,qp)=>{ if(v>0) out.push(qp); }); return out; }
function callsAfter(i,qp){ const n=S.T.nodes[i], N=S.T.nodes, out=[]; for(let k=qp+1;k<n.q.length && n.q[k]<0;k++){ const c=-n.q[k]-1; if(visible(N[c])) out.push(c); else out.push(...visKids(c)); } return out; }
function selectLine(i,qp,opt={}){
  const n=S.T.nodes[i]; const L=n.q[qp];
  S.sel=i; S.selLine={i,qp}; S.cursor=L;
  if(!S.tab || S.tab.c || S.tab.fid!==n.f) openFileFor(i); else applyMarks();
  scrollToLine(L,false); applyMarks();
  const ix=S.lineRowIndex.get(i+":"+qp); if(ix!=null) scrollRow(ix); else renderTree();
  updateSel();
}
function stepBy(d){
  const i=S.sel; if(i<0) return; const st=lineSteps(i); if(!st.length) return;
  let k = S.selLine && S.selLine.i===i ? st.indexOf(S.selLine.qp) : -1;
  k = k<0 ? (d>0?0:st.length-1) : Math.max(0,Math.min(st.length-1,k+d));
  selectLine(i,st[k]);
}
function renderStepBar(){
  const bar=$("#stepbar"); if(!bar) return; const t=S.tab;
  if(!t || t.c){ bar.innerHTML=""; return; }
  const cov=fileCov(t.fid); const f=DATA.files[t.fid]; const total=(f.src||"").split("\n").length;
  let h=`<span class="cv-sum" title="このトレースでこのファイル内で実行された行数 / ファイルの総行数">実行 ${cov.lines.size} / ${total} 行 · 関数 ${cov.funcs.length}</span>`;
  const n=S.sel>=0?S.T.nodes[S.sel]:null;
  if(n && n.f===t.fid && n.q && n.q.length){
    const st=lineSteps(n.i); const k = S.selLine && S.selLine.i===n.i ? st.indexOf(S.selLine.qp) : -1;
    const calls = k>=0 ? callsAfter(n.i, st[k]) : [];
    h+=`<span class="sb"><button class="ibtn" id="stPrev" title="前の実行行 (p)">◀</button><span class="stl">${k>=0?`ステップ ${k+1}/${st.length} · <b>${S.cursor}</b> 行目`:`${st.length} ステップ`}</span><button class="ibtn" id="stNext" title="次の実行行 (n)">▶</button></span>`;
    if(calls.length) h+=`<span class="stc">→ ${calls.slice(0,3).map(c=>`<a class="lnk" data-go="${c}">${esc(S.T.nodes[c].n.split(".").pop())}</a>`).join(", ")}${calls.length>3?` 他${calls.length-3}`:""}</span>`;
    if(n.qt) h+=`<span class="stc" title="記録上限に達したため途中まで">(途中まで)</span>`;
  }
  bar.innerHTML=h;
  const pv=$("#stPrev"), nx=$("#stNext"); if(pv) pv.onclick=()=>stepBy(-1); if(nx) nx.onclick=()=>stepBy(1);
  bar.querySelectorAll("a[data-go]").forEach(a=>a.onclick=()=>select(+a.dataset.go,true));
}
function scrollToLine(L,flash){
  const t=S.tab; if(!t||t.c) return; S.pendLine=L; const el=t.view.lines[L-1]; if(!el) return;
  const code=$("#code"); code.scrollTop=Math.max(0, el.offsetTop - code.clientHeight*0.28);
  if(flash){ el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
}
function onCodeClick(e){
  const ln=e.target.closest(".ln"); if(!ln) return;
  if(!ln.classList.contains("has")){
    const L=+ln.dataset.l, n=S.T.nodes[S.sel];
    if(n && n.q && ln.classList.contains("ex")){ const st=lineSteps(n.i).filter(qp=>n.q[qp]===L); if(st.length){ const cur=S.selLine&&S.selLine.i===n.i?S.selLine.qp:-1; selectLine(n.i, st.find(qp=>qp>cur) ?? st[0]); } }
    return;
  }
  const L=+ln.dataset.l, N=S.T.nodes, n=N[S.sel]; if(!n) return;
  const ids=visKids(n.i).filter(c=>N[c].cl===L && N[c].cf===S.tab.fid);
  if(ids.length===1) { select(ids[0],true); return; }
  showMenu(e.clientX,e.clientY, ids.map(i=>[`<span class="ico" style="background:${LK[N[i].k].col};color:#1e1e1e;display:inline-flex">${LK[N[i].k].ico}</span>${esc(nodeName(N[i]))}`, ()=>select(i,true)]));
}
function showMenu(x,y,items){
  const m=$("#menu"); m.innerHTML=items.map((it,k)=>`<div data-k="${k}">${it[0]}</div>`).join(""); m.style.display="block";
  const w=m.offsetWidth,h=m.offsetHeight; m.style.left=Math.min(x,innerWidth-w-8)+"px"; m.style.top=Math.min(y,innerHeight-h-8)+"px";
  m.querySelectorAll("div").forEach(d=>d.onclick=()=>{ m.style.display="none"; items[+d.dataset.k][1](); });
}
document.addEventListener("click",e=>{ if(!e.target.closest("#menu") && !e.target.closest(".ln")) $("#menu").style.display="none"; },true);

function cDoc(i){
  const n=S.T.nodes[i]; const ci=S.T.cinfo[n.c]||{}; const ver=DATA.env.numpy;
  const L=[]; const c=s=>`<span class="cmt">${s}</span>`;
  if(n.k==="blas"){
    const nv=ci.native||{};
    L.push(c("/*  ネイティブ層 (推定)  ────────────────────────────"));
    L.push(c(` *  ${esc(n.n)}`));
    L.push(c(" */"));
    L.push(c(`// ライブラリ   : ${esc(nv.lib||"")}`));
    L.push(c(`// ルーチン     : ${esc(nv.routine||"")}`));
    const bl=DATA.env.blas||{}; if(bl.config) L.push(c(`// ビルド設定   : ${esc(bl.config)}`));
    L.push(""); (nv.note||"").split("。").filter(Boolean).forEach(s=>L.push(c("// "+esc(s)+"。")));
    L.push(""); L.push(c("// ここから先は機械語（共有ライブラリ）なので Python からは追跡できません。"));
    L.push(c("// 呼び出し元の C 拡張ノードの dtype と gufunc 名から推定した名前です。"));
  } else {
    L.push(c("/*  NumPy C 拡張 ─ 配布物にCソースは含まれません  ───────────"));
    L.push(c(` *  ${esc(ci.qual||n.n)}`)); L.push(c(" */"));
    const row=(k,v)=>{ if(v) L.push(`<span class="kw">${esc(k.padEnd(14," "))}</span><span class="op">: </span><span class="str">${v}</span>`); };
    row("種類", esc(ci.type)); row("実装モジュール", esc(ci.impl)); row("共有ライブラリ", esc(ci.so));
    if(ci.hint){ const p=ci.hint.split(" ")[0].replace(/\/$/,""); const isDir=ci.hint.endsWith("/")||!/\.\w+$/.test(p); const url=`https://github.com/numpy/numpy/${isDir?"tree":"blob"}/v${ver}/${p.split("→")[0].trim()}`; row("Cソース(目安)", `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(ci.hint)}</a>`); }
    if(ci.ufunc){ const u=ci.ufunc; row("入出力", `nin=${u.nin}, nout=${u.nout}` + (u.signature?`, signature ${esc(u.signature)}`:"")); row("恒等元", esc(u.identity)); L.push(""); L.push(c(`// 型ループ (${u.ntypes} 種類) — 実行時に入力 dtype から 1 つが選ばれる`));
      for(let k=0;k<u.types.length;k+=8) L.push(`<span class="var">  ${u.types.slice(k,k+8).map(s=>esc(s).padEnd(9," ")).join("")}</span>`); }
    if(n.a&&n.a.length){ L.push(""); L.push(c("// この呼び出しの第1引数")); n.a.forEach(([k,v])=>L.push(`<span class="var">${esc(k)}</span><span class="op"> = </span><span class="cls">${esc(v)}</span>`)); }
    if(ci.doc){ L.push(""); L.push(c("/* docstring")); ci.doc.split("\n").forEach(s=>L.push(c(" * "+esc(s)))); L.push(c(" */")); }
  }
  const p=visParent(i); const N=S.T.nodes;
  if(n.cf>=0){ L.push(""); L.push(`${c("// 呼び出し元 → ")}<a class="lnk" data-site="${i}">${esc(DATA.files[n.cf].path)}:${n.cl}</a>`); }
  const kids=visKids(i); if(kids.length){ L.push(""); L.push(c(`// ここから呼ばれた処理 (${kids.length})`)); kids.slice(0,60).forEach(k=>L.push(`  <a class="lnk" data-go="${k}">${esc(LK[N[k].k].L)} ${esc(nodeName(N[k]))}</a>`)); }
  const wrap=document.createElement("div"); wrap.className="lines vdoc";
  wrap.innerHTML=L.map((h,k)=>`<div class="ln"><span class="g"></span><span class="num">${k+1}</span><span class="src">${h||" "}</span></div>`).join("");
  wrap.addEventListener("click",e=>{ const a=e.target.closest("a.lnk"); if(!a) return; if(a.dataset.go) select(+a.dataset.go,true); else if(a.dataset.site) gotoSite(+a.dataset.site); });
  return {el:wrap, lines:wrap.children, n:L.length};
}
function gotoNodeLine(i){ const n=S.T.nodes[i]; if(n.f>=0){ openFileFor(i); scrollToLine(n.l,true); } }
function openFileFor(i,site){
  const n=S.T.nodes[i]; const f=DATA.files[n.f];
  openTab({key:"f"+n.f, fid:n.f, label:basename(f.path), title:f.path, user:f.kind==="user", site:site||0});
}
function openFile(fid){ const f=DATA.files[fid]; openTab({key:"f"+fid, fid, label:basename(f.path), title:f.path, user:f.kind==="user", site:0}); }
function gotoSite(i){
  const n=S.T.nodes[i]; if(n.cf<0) return; const f=DATA.files[n.cf];
  const p=visParent(i);
  if(p>=0 && S.T.nodes[p].f===n.cf) { select(p,true,{noEditor:true}); }
  openTab({key:"f"+n.cf, fid:n.cf, label:basename(f.path), title:f.path, user:f.kind==="user", site:n.cl});
  scrollToLine(n.cl,true);
}

/* ---------------- selection ---------------- */
function select(i,doReveal,opt={}){
  S.sel=i; S.selLine=null; S.cursor=0; const n=S.T.nodes[i];
  if(doReveal) reveal(i); else renderTree();
  if(!opt.noEditor){
    if(n.f>=0 && (n.k==="np"||n.k==="user"||n.k==="stdlib")){
      openFileFor(i);
      let L = n.i===S.T._prep.roots[0]&&n.k==="user" ? 1 : n.l;
      const N=S.T.nodes; const sites=visKids(i).map(c=>N[c]).filter(c=>c.cf===n.f && c.cl>=n.l).map(c=>c.cl);
      if(sites.length){ const f=Math.min(...sites); if(f-L>18) L=f-5; }
      scrollToLine(L,false); }
    else { openTab({key:"c"+i, c:true, blas:n.k==="blas", node:i, label:n.n.split(".").pop(), title:n.n, view:cDoc(i)}); }
  } else applyMarks();
  updateSel(true);
}
function updateSel(){ renderPanel(); renderStatus(); }

/* ---------------- panel ---------------- */
function renderPanel(){
  document.querySelectorAll(".ptab").forEach(p=>p.classList.toggle("on",p.dataset.p===S.ptab));
  const b=$("#pbody"); const T=S.T, N=T.nodes;
  if(S.ptab==="out"){ b.innerHTML=`<pre class="out">${esc(T.stdout||"(標準出力なし)")}</pre>${T.error?`<pre class="out err">${esc(T.error)}</pre>`:""}${T.truncated?'<div class="err">ノード数が上限に達したため記録を打ち切りました (--max-nodes で変更可)</div>':""}`; return; }
  if(S.ptab==="env"){ const e=DATA.env; b.innerHTML=`<table class="kv"><tr><td>Python</td><td class="mono">${esc(e.python)}</td></tr><tr><td>NumPy</td><td class="mono">${esc(e.numpy)}</td></tr><tr><td>BLAS</td><td class="mono">${esc(e.blas?e.blas.name+" "+e.blas.version:"不明")}</td></tr><tr><td>LAPACK</td><td class="mono">${esc(e.lapack?e.lapack.name+" "+e.lapack.version:"不明")}</td></tr><tr><td>OpenBLAS設定</td><td class="mono">${esc(e.blas&&e.blas.config||"-")}</td></tr><tr><td>プラットフォーム</td><td class="mono">${esc(e.platform)}</td></tr><tr><td>numpyの場所</td><td class="mono">${esc(e.numpy_dir)}</td></tr><tr><td>生成日時</td><td class="mono">${esc(DATA.generated)}</td></tr><tr><td>ノード数</td><td class="mono">${N.length}</td></tr></table>${licenseHtml()}`; return; }
  if(S.sel<0){ b.innerHTML='<div class="empty">ノードを選ぶと、ユーザーコードから選択ノードまでの「層の降り方」がここに表示されます。</div>'; return; }
  const n=N[S.sel];
  if(S.ptab==="detail"){
    const ci=n.c?T.cinfo[n.c]||{}:null;
    const site = n.cf>=0 ? `<a class="lnk" id="dsite">${esc(DATA.files[n.cf].path)}:${n.cl}</a>` : "-";
    const loc = n.f>=0 ? `<a class="lnk" id="dloc">${esc(DATA.files[n.f].path)}:${n.l}–${n.e}</a>` : "-";
    b.innerHTML=`<table class="kv">
      <tr><td>名前</td><td class="mono">${esc(n.n)}</td></tr>
      <tr><td>層</td><td><span class="pill" style="background:${LK[n.k].col}">${LK[n.k].L} ${LK[n.k].name}</span> ${n.est?'<span class="err">推定ノード</span>':""}</td></tr>
      <tr><td>モジュール</td><td class="mono">${esc(n.m||"-")}</td></tr>
      <tr><td>定義位置</td><td class="mono">${loc}</td></tr>
      <tr><td>呼び出し元</td><td class="mono">${site}</td></tr>
      <tr><td>深さ</td><td class="mono">${T._prep.depth[n.i]}</td></tr>
      <tr><td>時間</td><td class="mono">${n.est?"-":`合計 ${fmtT(n.dur)} / 自己 ${fmtT(n.self)}`}${n.rs?` (ジェネレータ再開 ${n.rs} 回)`:""}</td></tr>
      ${(n.a||[]).length?`<tr><td>引数</td><td>${n.a.map(([k,v])=>`<div class="mono"><span class="var">${esc(k)}</span> = <span class="cls">${esc(v)}</span></div>`).join("")}</td></tr>`:""}
      ${n.r!=null?`<tr><td>戻り値</td><td class="mono cls">${esc(n.r)}</td></tr>`:""}
      ${ci&&ci.so?`<tr><td>共有ライブラリ</td><td class="mono">${esc(ci.so)}</td></tr>`:""}
      ${ci&&ci.type?`<tr><td>C 側の種類</td><td class="mono">${esc(ci.type)}</td></tr>`:""}
      <tr><td>子ノード</td><td class="mono">${T._prep.kids[n.i].length} (表示中 ${visKids(n.i).length})</td></tr>
      ${n.h?`<tr><td>実行した行</td><td class="mono" style="white-space:normal">${Object.entries(n.h).sort((a,b)=>a[0]-b[0]).map(([L,c])=>`<a class="lnk" data-l="${L}">${L}</a>${c>1?`<span style="color:#888">×${c}</span>`:""}`).join(" ")}<div style="color:#888;margin-top:3px">実行順の記録 ${n.q.length} 件${n.qt?"（上限で打ち切り）":""}</div></td></tr>`:""}</table>`;
    b.querySelectorAll("a[data-l]").forEach(a=>a.onclick=()=>{ const L=+a.dataset.l; const qp=n.q.indexOf(L); if(qp>=0) selectLine(n.i,qp); if(isMobile()) setMobile("code"); });
    const ds=$("#dsite"); if(ds) ds.onclick=()=>{ gotoSite(n.i); if(isMobile()) setMobile("code"); };
    const dl=$("#dloc"); if(dl) dl.onclick=()=>{ gotoNodeLine(n.i); if(isMobile()) setMobile("code"); };
    return;
  }
  // lanes
  const ch=chain(S.sel); const kids=visKids(S.sel).slice(0,40);
  const used=[]; for(const i of ch){ const L=LK[N[i].k].L; if(L!=="py" && used[used.length-1]!==L) used.push(L); }
  const deepest=ch.reduce((a,i)=>Math.max(a,LANE[N[i].k]),0);
  let rows=ch.map(i=>({i,cur:i===S.sel})).concat(kids.map(i=>({i,child:true})));
  const RH=44;
  let h=`<div class="sumline"><span>ユーザーコード → 選択ノードまで <b>${ch.length}</b> 段 / 通過した層:</span>${used.map(L=>{const l=LAYERS.find(x=>x.L===L); return `<span class="pill" style="background:${l.col}">${L}</span>`}).join("<span style='color:#777'>→</span>")}${kids.length?`<span style="color:#888">（点線: この下で呼ばれる ${visKids(S.sel).length} 件）</span>`:""}</div>`;
  h+=`<div class="lanes"><div class="lhead">${LANES.map(([t,c])=>`<div style="border-color:${c}">${t}</div>`).join("")}</div><div class="lgrid" style="height:${rows.length*RH}px"><div class="lanebg">${LANES.map(()=>"<div></div>").join("")}</div><svg class="conn" id="conn"></svg>`;
  rows.forEach((r,ix)=>{ const nn=N[r.i]; const lane=LANE[nn.k]; const L=LK[nn.k];
    let sub = nn.k==="c"||nn.k==="blas" ? (nn.m||"").replace("numpy.","") : (nn.f>=0? basename(DATA.files[nn.f].path)+":"+nn.l : "");
    if(nn.cf>=0 && nn.cl) sub += `  ← ${basename(DATA.files[nn.cf].path)} ${nn.cl}行目から`;
    h+=`<div class="lrow" style="position:absolute;left:0;right:0;top:${ix*RH}px"><div class="lbox${r.cur?" cur":""}${r.child?" child":""}" data-i="${r.i}" style="grid-column:${lane+1} / -1;background:color-mix(in srgb, ${L.col} 22%, #232323);border-color:${L.col}" title="${esc(nn.n)}"><span class="ico" style="background:${L.col}">${L.ico}</span><span class="nn">${esc(nodeName(nn))}</span><span class="d">${esc(sub)}</span></div></div>`; });
  h+=`</div></div>`;
  b.innerHTML=h;
  b.querySelectorAll(".lbox").forEach(x=>x.onclick=()=>select(+x.dataset.i,true));
  requestAnimationFrame(()=>{ const svg=$("#conn"); if(!svg) return; const W=svg.clientWidth; const lw=W/5; let p="";
    const pt=(ix)=>{ const nn=N[rows[ix].i]; return [lw*LANE[nn.k]+18, ix*RH+RH/2]; };
    for(let k=1;k<ch.length;k++){ const [x0,y0]=pt(k-1),[x1,y1]=pt(k); const col=LK[N[rows[k].i].k].col; p+=`<path d="M${x0},${y0+18} C${x0},${y0+RH*0.8} ${x1},${y1-RH*0.8} ${x1},${y1-18}" stroke="${col}" stroke-width="1.6" fill="none" opacity=".85"/>`; }
    const ci=ch.length-1; for(let k=ch.length;k<rows.length;k++){ const [x0,y0]=pt(ci),[x1,y1]=pt(k); p+=`<path d="M${x0},${y0+18} L${x0},${y1} L${x1-12},${y1}" stroke="#666" stroke-dasharray="3 3" fill="none"/>`; }
    svg.innerHTML=p; });
}

function renderStatus(){
  const e=DATA.env, T=S.T, n=S.sel>=0?T.nodes[S.sel]:null;
  $("#status").innerHTML=`<span>⎇ numpy ${esc(e.numpy)}</span><span class="hide-m">Python ${esc(e.python)}</span><span class="hide-m">BLAS: ${esc(e.blas?e.blas.name:"?")}</span><span>${T.nodes.length} ノード${T.truncated?" (打ち切り)":""}</span>
   <span class="r">${n?`<span>${LK[n.k].L} ${esc(LK[n.k].name)}</span><span class="hide-m">深さ ${T._prep.depth[n.i]}</span>`:""}<span class="hide-m">${T.error?"⚠ 例外あり":"✓ 実行完了"}</span></span>`;
}

/* ---------------- views / layout ---------------- */
function setView(v){ S.view=v; document.querySelectorAll(".abtn").forEach(a=>a.classList.toggle("on",a.dataset.v===v)); renderSidebar(); if(isMobile()) setMobile("tree"); }
function setMobile(m){ document.body.className="m-"+m; document.querySelectorAll("#mtabs div").forEach(d=>d.classList.toggle("on",d.dataset.m===m)); if(m==="tree") renderTree(); if(m==="panel") renderPanel(); if(m==="code"){ applyMarks(); if(S.pendLine) scrollToLine(S.pendLine,false); } }
function loadTrace(ti){
  S.ti=ti; S.T=prepTrace(ti); S.vk.clear(); S.ti_.clear(); S.hl.clear(); S.cov.clear(); S.fileCache.clear(); S.selLine=null; S.cursor=0; S.sel=-1; S.tabs=[]; S.tab=null; S.query=""; S.hits=null;
  defaultExpand();
  renderSidebar();
  const N=S.T.nodes;
  let first=-1; const lap=N.find(n=>n.k==="blas"&&n.n.startsWith("LAPACK"))||N.find(n=>n.k==="blas"&&n.n.startsWith("BLAS"))||N.find(n=>n.k==="blas"&&!n.n.startsWith("C "))||N.find(n=>n.k==="blas"); if(lap) first=lap.p;
  if(first<0){ const c=N.find(n=>n.k==="c"); first=c?c.i:(N.length?0:-1); }
  const root=S.T._prep.roots.find(r=>N[r].k==="user");
  if(root!=null) select(root,false);
  if(first>=0) select(first,true);
  if(isMobile()) setMobile("tree");
  renderStatus();
}
function licenseHtml(){
  const ns=DATA.notices||[];
  let h=`<h4 style="margin:16px 0 6px;font-size:12.5px">ライセンス</h4><div style="font-size:12px;color:#bbb;line-height:1.7">このビューア／トレーサーは MIT License です（<a class="lnk" href="https://github.com/shin22s-lab/py_structure1" target="_blank" rel="noopener">py_structure1</a>）。<br>エディタに表示している NumPy のソースコードと docstring は NumPy 開発者の著作物で、以下のライセンスに従って再配布しています。</div>`;
  if(!ns.length) h+=`<div style="font-size:12px;color:#888;margin-top:4px">（このファイルには NumPy のソースは含まれていません）</div>`;
  for(const x of ns) h+=`<details style="margin-top:6px"><summary style="cursor:pointer;font-size:12px">${esc(x.scope)} — ${esc(x.file.split("/").slice(-2).join("/"))}</summary><pre class="out" style="font-size:11px;max-height:280px;overflow:auto;background:#181818;padding:8px;border-radius:4px">${esc(x.text)}</pre></details>`;
  return h;
}
function help(){
  $("#modalBox").innerHTML=`<h2>NumPy Trace Explorer の見方</h2>
  <p>Python 3.12+ の <code>sys.monitoring</code> で実際に実行を追跡し、NumPy がどのファイルのどの関数を経由して、どの C 拡張・ネイティブライブラリまで降りていくかを木構造で記録したものです。</p>
  <h3>層 (Layer)</h3><ul>
  <li><b style="color:var(--L0)">L0 ユーザーコード</b> — トレース対象のスクリプト</li>
  <li><b style="color:var(--L1)">L1 NumPy Python層</b> — <code>numpy/**/*.py</code> の関数（ソースを実際に表示）</li>
  <li><b style="color:var(--L2)">L2 NumPy C拡張</b> — <code>_multiarray_umath</code> などの .so。ufunc・ndarray メソッド・<code>__array_function__</code> ディスパッチャ</li>
  <li><b style="color:var(--L3)">L3 ネイティブ</b> — OpenBLAS / LAPACK / pocketfft など（<i>斜体 = 推定</i>）</li>
  <li><b style="color:var(--Ls)">標準ライブラリ</b>・<b style="color:#aaa">組み込み関数</b> — NumPy 内部から呼ばれたもの。初期状態では非表示（チップで切替）</li></ul>
  <h3>操作</h3><ul>
  <li>ツリーのノードを選ぶとソースが開き、関数範囲がハイライト、そこから呼ばれた行に <b>●</b> が付きます（色は降りる層）。● の行番号をクリックするとその呼び出し先へ潜れます</li>
  <li>下パネル「層ビュー」はユーザーコード→選択ノードまでの経路を層のレーンで表示。点線はその下の呼び出し</li>
  <li><b>≡</b> ボタン（既定 ON）: 関数を開くと<b>実行した行が行番号つきで実行順に</b>並び、その行から呼ばれた関数がすぐ下に入ります。行をクリックするとエディタのその行へ</li>
  <li>エディタ: 薄い緑 = このトレースでそのモジュール内で実行された行、濃い緑 + <code>×回数</code> = 選択中の関数が実行した行。上部の ◀ ▶（キー <code>p</code>/<code>n</code>）で実行順に 1 行ずつ辿れます</li>
  <li>左下「参照されたモジュール」の ▸ を開くと、そのファイルで呼ばれた関数の行範囲と実行行数を一覧できます</li>
  <li>⤓ ボタン: ネイティブ層まで届く経路だけを展開 / 矢印キーでツリー移動</li>
  <li>左の 🔍 で関数検索、📊 で自己時間のホットスポットと層別割合</li></ul>
  <h3>制約</h3><ul>
  <li>C 拡張の<b>内部</b>は追跡できません（C 関数の入口までが観測可能）。L3 名は gufunc 名と dtype からの推定です</li>
  <li>ユーザーコード中の演算子 <code>+ - * / @ ** 比較 単項</code> は AST 書き換えで可視化。<code>a[i]</code>・<code>a += b</code> などは記録されません</li>
  <li>時間はトレースのオーバーヘッドを含む相対的な目安です</li>
  <li>実行行は関数あたり最大 3000 件まで記録します（ループが長いと途中まで）。<code>import</code> 時に実行されるモジュール直下のコードは対象外です</li></ul>
  <h3>自分のコードで使う</h3>
  <p><code>python numpy_trace.py my_script.py -o trace.html</code><br><code>python numpy_trace.py -c "import numpy as np; np.linalg.svd(np.ones((3,3)))" -o t.html</code></p>
  ${licenseHtml()}
  <p style="text-align:right"><button class="tbtn" style="border-color:#555" onclick="document.getElementById('modal').style.display='none'">閉じる</button></p>`;
  $("#modal").style.display="flex";
}
function initSplit(){
  const app=$("#app"), main=$("#main");
  $("#vsplit").onpointerdown=e=>{ const x0=e.clientX, w0=$("#sidebar").offsetWidth; const mv=ev=>{ app.style.setProperty("--sbw",Math.max(180,Math.min(innerWidth-300,w0+ev.clientX-x0))+"px"); renderTree(); }; const up=()=>{removeEventListener("pointermove",mv);removeEventListener("pointerup",up); applyMarks();}; addEventListener("pointermove",mv); addEventListener("pointerup",up); };
  $("#hsplit").onpointerdown=e=>{ const y0=e.clientY, h0=$("#panel").offsetHeight; const mv=ev=>{ main.style.setProperty("--pnh",Math.max(60,Math.min(innerHeight-160,h0-(ev.clientY-y0)))+"px"); }; const up=()=>{removeEventListener("pointermove",mv);removeEventListener("pointerup",up); applyMarks(); renderPanel();}; addEventListener("pointermove",mv); addEventListener("pointerup",up); };
}
function init(){
  if(!DATA || !DATA.traces || !DATA.traces.length){ document.body.innerHTML='<div class="empty">トレースデータがありません。</div>'; return; }
  const sel=$("#scn"); sel.innerHTML=DATA.traces.map((t,k)=>`<option value="${k}">${esc(t.title)}  —  ${t.nodes.length} ノード</option>`).join("");
  sel.onchange=()=>loadTrace(+sel.value);
  document.querySelectorAll(".abtn").forEach(a=>a.onclick=()=>setView(a.dataset.v));
  document.querySelectorAll(".ptab").forEach(p=>p.onclick=()=>{ S.ptab=p.dataset.p; renderPanel(); });
  document.querySelectorAll("#mtabs div").forEach(d=>d.onclick=()=>setMobile(d.dataset.m));
  $("#helpBtn").onclick=help; $("#modal").onclick=e=>{ if(e.target.id==="modal") e.currentTarget.style.display="none"; };
  addEventListener("resize",()=>{ renderTree(); applyMarks(); });
  addEventListener("keydown",e=>{ if(e.target.closest&&e.target.closest("input,select,textarea")) return; if(e.key==="n"){ stepBy(1); e.preventDefault(); } else if(e.key==="p"){ stepBy(-1); e.preventDefault(); } });
  initSplit();
  loadTrace(0);
  if(isMobile()) setMobile("tree");
}
init();
</script>
</body>
</html>
'''

DEMOS = [
    ("① 連立一次方程式  np.linalg.solve → LAPACK", '''import numpy as np

A = np.array([[4.0, 1.0, 0.0],
              [1.0, 3.0, 1.0],
              [0.0, 1.0, 2.0]])
b = np.array([1.0, 2.0, 3.0])

x = np.linalg.solve(A, b)          # _linalg.py → _umath_linalg(C) → LAPACK dgesv
res = np.linalg.norm(A @ x - b)    # 残差
'''),
    ("② 行列積  A @ B / np.dot → BLAS", '''import numpy as np

rng = np.random.default_rng(0)
A = rng.standard_normal((200, 300))
B = rng.standard_normal((300, 100))

C = A @ B                 # ndarray.__matmul__ → ufunc matmul → cblas_dgemm
v = np.dot(A, B[:, 0])    # dot → cblas_dgemv
G = A @ A.T               # 転置との積は dsyrk が使われることがある
i = np.arange(6).reshape(2, 3) @ np.arange(6).reshape(3, 2)   # 整数は BLAS を使わない
'''),
    ("③ リダクション  np.mean / sum / std", '''import numpy as np

a = np.arange(12.0).reshape(3, 4)
m = np.mean(a, axis=0)    # fromnumeric.mean → _methods._mean → add.reduce → divide
s = a.sum(axis=1)         # ndarray.sum (C) → _methods._sum
sd = np.std(a)            # _methods._std → _var
'''),
    ("④ 線形代数いろいろ  eigh / svd / det / inv / qr", '''import numpy as np

rng = np.random.default_rng(1)
M = rng.standard_normal((6, 6))
S = M + M.T

w, V = np.linalg.eigh(S)      # LAPACK dsyevd
U, s, Vt = np.linalg.svd(M)   # LAPACK dgesdd
d = np.linalg.det(M)          # LAPACK dgetrf
Mi = np.linalg.inv(M)         # LAPACK dgesv
Q, R = np.linalg.qr(M)        # dgeqrf + dorgqr
Z = np.linalg.eigvals(M.astype(np.complex128))   # zgeev
'''),
    ("⑤ FFT  np.fft.rfft / fft → pocketfft", '''import numpy as np

t = np.linspace(0.0, 1.0, 256, endpoint=False)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

X = np.fft.rfft(x)                 # _pocketfft.py → _pocketfft_umath(C++)
f = np.fft.rfftfreq(x.size, d=t[1] - t[0])
y = np.fft.irfft(X, n=x.size)
Y = np.fft.fft(x + 0j)
'''),
    ("⑥ 乱数  default_rng().normal (Cython)", '''import numpy as np

rng = np.random.default_rng(42)          # SeedSequence → PCG64 → Generator
x = rng.normal(0.0, 1.0, size=1000)      # Generator.normal (Cython) → C 分布関数
u = rng.uniform(size=(3, 3))
k = rng.integers(0, 10, size=5)
p = rng.permutation(10)
'''),
    ("⑦ 統計  percentile / median / histogram", '''import numpy as np

rng = np.random.default_rng(3)
data = rng.standard_normal(500)

q = np.percentile(data, [5, 50, 95])   # lib/_function_base_impl.py の深い Python 層
med = np.median(data)
hist, edges = np.histogram(data, bins=20)
c = np.corrcoef(data[:-1], data[1:])
'''),
    ("⑧ einsum(optimize=True) → bmm_einsum → matmul → BLAS", '''import numpy as np

rng = np.random.default_rng(5)
A = rng.standard_normal((40, 50))
B = rng.standard_normal((50, 60))
C = rng.standard_normal((60, 30))

P = np.einsum("ij,jk,kl->il", A, B, C, optimize=True)   # einsum_path で縮約順を決め → bmm_einsum → matmul
t = np.einsum("ii->", A[:40, :40])                       # optimize=False は c_einsum 直行
'''),
    ("⑨ 総合: 平滑化・微分・補間・条件抽出", '''import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x) + 0.1 * np.cos(7 * x)

kernel = np.ones(9) / 9
ys = np.convolve(y, kernel, mode="same")   # 移動平均
dy = np.gradient(ys, x)                    # 数値微分
yi = np.interp([1.5, 2.5, 7.25], x, ys)    # 線形補間
peaks = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0]
area = np.trapezoid(ys, x)
order = np.argsort(ys)[-5:]
'''),
]


def _run_demo(max_nodes, **kw):
    out = []
    for title, src in DEMOS:
        out.append(trace_source(src, filename="<demo>", title=title, max_nodes=max_nodes, **kw))
        # ユーザーファイル名をシナリオごとに区別
        for f in out[-1]["files"]:
            if f["kind"] == "user":
                f["path"] = "user/" + title.split()[0] + "_demo.py"
    return out


def main(argv=None):
    import argparse
    if sys.version_info < (3, 12):
        sys.exit("numpy_trace には Python 3.12 以上（sys.monitoring）が必要です")
    ap = argparse.ArgumentParser(
        prog="numpy_trace",
        description="Python コードを実行し、NumPy がどこを参照して計算しているかを層ごとに VS Code 風ビューアで表示する HTML を生成します。")
    ap.add_argument("script", nargs="?", help="トレースする .py ファイル")
    ap.add_argument("args", nargs="*", help="スクリプトに渡す引数（`--` 以降はすべてスクリプトへ）")
    ap.add_argument("-c", dest="code", help="コード文字列をトレース")
    ap.add_argument("-o", "--output", default="numpy_trace.html", help="出力 HTML（既定: numpy_trace.html）")
    ap.add_argument("--json", help="生データを JSON でも保存")
    ap.add_argument("--demo", action="store_true", help="組み込みのデモ 9 本をトレース")
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT, help=f"記録するノード数の上限（既定 {MAX_NODES_DEFAULT}）")
    ap.add_argument("--title", help="トレースのタイトル")
    ap.add_argument("--no-stdlib-source", action="store_true",
                    help="Python 標準ライブラリのソースを HTML に埋め込まない（公開用。ツリーには残る）")
    ap.add_argument("--no-lines", action="store_true", help="実行行（何行目を通ったか）を記録しない（軽量化）")
    ap.add_argument("--user-path", action="append", help="ユーザーコード(L0)として扱う追加ディレクトリ（スクリプトと同じフォルダは自動で対象）")
    argv = list(sys.argv[1:] if argv is None else argv)
    passthru = []
    if "--" in argv:
        k = argv.index("--")
        argv, passthru = argv[:k], argv[k + 1:]
    a, unknown = ap.parse_known_args(argv)
    a.args = list(a.args) + unknown + passthru

    traces = []
    kw = dict(embed_stdlib=not a.no_stdlib_source, record_lines=not a.no_lines)
    if a.demo:
        traces = _run_demo(a.max_nodes, **kw)
    elif a.code:
        traces = [trace_source(a.code, filename="<cmd>", title=a.title or "-c コード", max_nodes=a.max_nodes, **kw)]
    elif a.script:
        path = os.path.abspath(a.script)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        sys.argv = [path] + a.args
        sys.path.insert(0, os.path.dirname(path))
        udirs = [os.path.dirname(path)] + (a.user_path or [])
        traces = [trace_source(src, filename=path, title=a.title or os.path.basename(path),
                               max_nodes=a.max_nodes, user_dirs=udirs, **kw)]
    else:
        ap.print_help()
        return 1

    html = build_html(traces, TEMPLATE)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(merge_traces(traces), fh, ensure_ascii=False)
    for t in traces:
        print(f"[numpy_trace] {t['title']}: {len(t['nodes'])} nodes"
              + (" (truncated)" if t["truncated"] else "") + (f"  ⚠ {t['error']}" if t["error"] else ""))
    print(f"[numpy_trace] → {os.path.abspath(a.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
