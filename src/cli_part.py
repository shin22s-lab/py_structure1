
# ---------------------------------------------------------------------------
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
