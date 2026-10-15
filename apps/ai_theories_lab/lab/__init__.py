"""AI Theories Lab のアプリ固有のコード。

- ``core``: 画面に依存しない共通部分(Hub からの読み込み・デバイス・計測)。
- ``features``: 機能ごとのコード。各機能は、処理(``service.py``)と画面(``ui.py``)に分ける。
  ``service.py`` は Gradio を import しない。
- ``ui``: 共通の画面部品・テーマ・ページの配置。

理論に関わる計算(注意機構・KV キャッシュ・デコードのフィルタ・ビームサーチ・指示データ・
トークナイザ)は ``src/``(ai-theories)のものをそのまま使い、ここには画面の都合で必要な
部分だけを置く。
"""

import spaces  # noqa: F401  ZeroGPU: torch など CUDA を使うパッケージより前に import する(下記)

# ZeroGPU の ``spaces`` は、torch など CUDA を使うパッケージより前に import する必要がある。
# ``lab`` のどのモジュールを import しても、まずこのファイルが実行されるので、torch より前に
# ``spaces`` が読み込まれる(呼び出し側が import の順序を意識しなくてよい)。
