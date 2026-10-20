"""常駐サーバ(手元専用): 常駐のエンジン・HTTP の API・負荷を掛ける道具。

画面のタブ「バッチ推論」(``lab/features/serving/``)は、1 回の ``@spaces.GPU`` の関数の中で
負荷を流して終わる(ZeroGPU と互換)。これに対して、本パッケージは **常駐する** エンジンが、
実行中に届く要求を受け付けて、複数の要求を同時に抱える。要求ごとに別プロセスで GPU を割り当てる
ZeroGPU とは合わないので、**手元専用** で、Hugging Face Spaces では動かない(Spaces は ``app.py``
だけを起動する)。起動は ``uv run python serve.py``。

- ``engine``: 反復を 1 つずつ進める常駐のエンジン(``src/serving/`` の部品を組み合わせたもの)。
- ``runtime``: エンジンを動かす 1 本のスレッドと、モデルの読み込み・起動と停止。
- ``contract``: API の要求の検査と、トークンの逐次復号。
- ``api``: FastAPI の端点(``/v1/generate``・``/v1/health``・``/v1/stats``)。
- ``loadgen``: API に同時に要求を送り、スループットと待ち時間を集計する道具。

このパッケージは Gradio と ``spaces`` を import しない。torch の呼び出しは、エンジンのスレッドに限る
(HTTP の処理は、要求を待ち行列に入れ、生成されたトークンを受け取るだけにする)。

使用した理論: 027(continuous batching と PagedAttention。実装は ``src/serving/``)、
010(KV キャッシュ)、008(デコードの方法)。API の層そのものは、システムの仕組みなので、
理論トピックはない。
"""
