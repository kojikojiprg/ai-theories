"""試験で共有する定数。"""

# 010 の PROD_CONFIG(本物の main の config.json の内容。010 のセル出力)
PRODUCTION_CONFIG = {
    "vocabulary_size": 8192,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 1024,
    "swiglu_d_ff": 683,
    "sequence_length": 256,
    "dropout": 0.0,
    "tie_embeddings": True,
    "positional_encoding": "rope",
    "normalization": "rmsnorm",
    "feed_forward": "swiglu",
    "norm_first": True,
}
GQA_KEY_VALUE_HEADS = 4  # 010 の実験 D の gqa_mean_pool

# アップロードしたチェックポイントの SHA-256(017 の 6.14 節、010 の実験 D のセル出力)
EXPECTED_SHA256 = {
    "sft-synthetic": "5814fbc024a27adc1624ee9a718fbbd5a7e28f5d20a0df40c8d4633126a91ab5",
    "gqa": "f23e68cd4a2865f01612f20af10482fe7146deb84d16df18a78489e43037b9ca",
}
