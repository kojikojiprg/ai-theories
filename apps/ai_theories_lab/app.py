"""AI Theories Lab(Hugging Face Spaces の ZeroGPU で動く Gradio アプリ)の入口。

各機能のページ(``lab/features/*/ui.py``)を組み立てるだけで、処理は ``lab/features/*/service.py``
にある。

**ZeroGPU の作法**

- ``spaces`` は、CUDA を使うパッケージ(torch など)より前に import する(``spaces`` が要求する)。
- モデルはモジュールの読み込み時に 1 回だけ読み込み、デバイスへ載せる(ZeroGPU の推奨)。
- GPU を使う処理は ``@spaces.GPU`` を付けた関数の中で行う(``lab.core.device.gpu``)。ZeroGPU の外
  (手元の MPS や CPU)では、この装飾子は何もせず、同じファイルがそのまま動く。

手元での起動: このディレクトリで ``uv run python app.py``。
"""

import spaces  # noqa: F401  ZeroGPU: CUDA を使うパッケージ(torch など)より前に import する

# isort: split

from lab.core.artifacts import load_resources
from lab.features.generation.ui import PAGE as GENERATION_PAGE
from lab.features.inspection.ui import PAGE as INSPECTION_PAGE
from lab.features.instruction.ui import PAGE as INSTRUCTION_PAGE
from lab.ui.layout import build_app
from lab.ui.theme import launch_kwargs

# 起動時に 1 回だけ読み込む(トークナイザと 3 ブランチのモデルを、同時にメモリへ載せる)
RESOURCES = load_resources()

demo = build_app(RESOURCES, [GENERATION_PAGE, INSPECTION_PAGE, INSTRUCTION_PAGE])

if __name__ == "__main__":
    demo.launch(**launch_kwargs())
