# Third-party notices

このリポジトリ自身のコード（`numpy_trace.py`, `src/`, `tools/`）は [MIT License](LICENSE) です。

公開用デモ `index.html` は、ビューアで読めるように **NumPy のソースコードと docstring をそのまま埋め込んでいます**。
これらは NumPy 開発者の著作物で、以下のライセンスに従って再配布しています（同じ条文は `index.html` 内にも埋め込み済み）。

- 取得元: NumPy 2.5.3（PyPI のバイナリホイールに同梱されたソース・ライセンスファイル）
- 改変: なし（表示用に HTML/JSON へ格納しているだけ）
- Python 標準ライブラリのソースは含めていません（`--no-stdlib-source`）。BLAS/LAPACK などはルーチン名を表示するだけで、コードは含みません。

<details><summary>index.html に含まれる NumPy のファイル（21 件）</summary>

- `numpy/__init__.py`
- `numpy/_core/_methods.py`
- `numpy/_core/_ufunc_config.py`
- `numpy/_core/einsumfunc.py`
- `numpy/_core/fromnumeric.py`
- `numpy/_core/function_base.py`
- `numpy/_core/getlimits.py`
- `numpy/_core/multiarray.py`
- `numpy/_core/numeric.py`
- `numpy/_core/numerictypes.py`
- `numpy/fft/_helper.py`
- `numpy/fft/_pocketfft.py`
- `numpy/lib/_arraysetops_impl.py`
- `numpy/lib/_function_base_impl.py`
- `numpy/lib/_histograms_impl.py`
- `numpy/lib/_stride_tricks_impl.py`
- `numpy/lib/_twodim_base_impl.py`
- `numpy/lib/_type_check_impl.py`
- `numpy/lib/_utils_impl.py`
- `numpy/linalg/_linalg.py`
- `numpy/ma/core.py`

</details>

## NumPy — numpy/ (本体)

出典: `numpy-2.5.3.dist-info/licenses/LICENSE.txt`

```text
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## NumPy — numpy/ma/

出典: `numpy-2.5.3.dist-info/licenses/numpy/ma/LICENSE`

```text
* Copyright (c) 2006, University of Georgia and Pierre G.F. Gerard-Marchant
* All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the University of Georgia nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
