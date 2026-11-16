#!/usr/bin/env python3
"""src/ の 3 ファイルから単一ファイルの numpy_trace.py を組み立てる。

    python tools/build.py           # numpy_trace.py を生成
    python tools/build.py --demo    # さらにデモを実行して index.html（GitHub Pages 用）を生成
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

HEADER = '''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 shin2225
"""
numpy_trace.py — NumPy がどこを参照して計算しているかを、層ごと・行ごとに VS Code 風ビューアで見るツール
https://github.com/shin2225/py_structure1

  使い方:
    python numpy_trace.py my_script.py -o trace.html        # スクリプトをトレース
    python numpy_trace.py -c "import numpy as np; np.linalg.svd(np.ones((4,4)))" -o t.html
    python numpy_trace.py --demo -o demo.html               # 組み込みデモ

  要件: Python 3.12+（sys.monitoring）, NumPy 2.x 推奨。生成される HTML は単体で開けます。
  生成 HTML には NumPy のソースが含まれます（BSD-3-Clause。条文は HTML 内に埋め込み）。
  自分で再配布するときはその表記を消さないでください。
"""
'''


def build():
    core = open(os.path.join(SRC, "tracer_core.py"), encoding="utf-8").read()
    cli = open(os.path.join(SRC, "cli_part.py"), encoding="utf-8").read()
    tpl = open(os.path.join(SRC, "viewer_template.html"), encoding="utf-8").read()
    if "'''" in tpl or tpl.endswith("\\"):
        sys.exit("viewer_template.html に ''' または末尾バックスラッシュがあります")
    core_body = core.split("\n", 1)[1]  # 先頭のコメント行を除く
    marker = "\nDEMOS = ["
    if marker not in cli:
        sys.exit("cli_part.py に DEMOS が見つかりません")
    cli = cli.replace(marker, "\nTEMPLATE = r'''" + tpl + "'''\n" + marker, 1)
    out = HEADER + core_body + "\n" + cli
    dst = os.path.join(ROOT, "numpy_trace.py")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"built {dst} ({len(out):,} bytes)")
    return dst


def write_notices(html_path):
    """index.html に埋め込んだ第三者ライセンスを THIRD_PARTY_NOTICES.md に書き出す"""
    import json
    import re
    s = open(html_path, encoding="utf-8").read()
    m = re.search(r"const DATA = (\{.*?\});\n", s, re.S)
    data = json.loads(m.group(1))
    np_files = sorted({f["path"] for f in data["files"] if f["kind"] == "np" and f["src"]})
    env = data["env"]
    out = [
        "# Third-party notices",
        "",
        "このリポジトリ自身のコード（`numpy_trace.py`, `src/`, `tools/`）は [MIT License](LICENSE) です。",
        "",
        "公開用デモ `index.html` は、ビューアで読めるように **NumPy のソースコードと docstring をそのまま埋め込んでいます**。",
        "これらは NumPy 開発者の著作物で、以下のライセンスに従って再配布しています（同じ条文は `index.html` 内にも埋め込み済み）。",
        "",
        f"- 取得元: NumPy {env['numpy']}（PyPI のバイナリホイールに同梱されたソース・ライセンスファイル）",
        "- 改変: なし（表示用に HTML/JSON へ格納しているだけ）",
        "- Python 標準ライブラリのソースは含めていません（`--no-stdlib-source`）。BLAS/LAPACK などはルーチン名を表示するだけで、コードは含みません。",
        "",
        "<details><summary>index.html に含まれる NumPy のファイル（" + str(len(np_files)) + " 件）</summary>",
        "",
    ] + [f"- `{p}`" for p in np_files] + ["", "</details>", ""]
    for n in data.get("notices", []):
        out += [f"## NumPy — {n['scope']}", "", f"出典: `{n['file']}`", "", "```text", n["text"].rstrip(), "```", ""]
    dst = os.path.join(ROOT, "THIRD_PARTY_NOTICES.md")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"wrote {dst}")


if __name__ == "__main__":
    dst = build()
    if "--demo" in sys.argv:
        html = os.path.join(ROOT, "index.html")
        subprocess.run([sys.executable, dst, "--demo", "--no-stdlib-source", "-o", html], check=True)
        write_notices(html)
