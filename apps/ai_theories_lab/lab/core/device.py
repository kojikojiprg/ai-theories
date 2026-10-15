"""デバイスの選択と、ZeroGPU の互換。

``spaces`` は、torch など CUDA を使うパッケージより前に import する必要がある
(``spaces`` が要求する)。その順序は ``lab/__init__.py`` が守る。

ZeroGPU の外(手元の MPS(Metal Performance Shaders)や CPU)では、``@spaces.GPU`` は何もしない
(関数をそのまま返す)ので、互換のための処理は要らない。ここでは、GPU を使う関数に付ける装飾子を
``gpu`` の名前で 1 か所にまとめるだけにする(ZeroGPU の仕様が変わったときの変更箇所を
1 つにするため)。
"""

import spaces
import torch

gpu = spaces.GPU
"""GPU を使う関数に付ける装飾子(``@gpu(duration=秒)``)。ZeroGPU の外では何もしない。"""


def select_device() -> torch.device:
    """CUDA → MPS(Metal Performance Shaders)→ CPU の順に、使えるデバイスを選ぶ。

    ZeroGPU では、``@spaces.GPU`` の関数の外(モジュールの読み込み時)でも
    ``torch.cuda.is_available()`` が真になる(CUDA のエミュレーション)。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
