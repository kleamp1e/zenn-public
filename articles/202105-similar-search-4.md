---
title: "類似画像検索ツールを作ってみる (4) 類似画像検索"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# 目次

* [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)
* [類似画像検索ツールを作ってみる (2) 特徴化 その1](202105-similar-search-2)
* [類似画像検索ツールを作ってみる (3) 特徴化 その2](202105-similar-search-3)
* 類似画像検索ツールを作ってみる (4) 類似画像検索（本記事）
* [類似画像検索ツールを作ってみる (5) 類似画像検索サーバ](202105-similar-search-5)
* [類似画像検索ツールを作ってみる (6) Next.js + SVGで可視化](202106-similar-search-6)

# ついに類似画像検索へ

[前回](202105-similar-search-3)、約10万枚の画像に対して特徴量を取得し、1万枚ごとに結合した特徴量ファイルを作成しました。

今回は実際に、この特徴量ファイルを使って類似画像検索を行ってみたいと思います。

# 類似画像を検索する

類似画像を検索する大まかな流れは以下の通りです。

1. ONNXモデルを読み込む。
2. クエリ画像を読み込む。
3. クエリ画像を整形する。
4. クエリ画像の特徴量を取得する。
5. 各特徴量ファイルについて:
    1. 特徴量ファイルを読み込む。
    2. クエリ画像の特徴量との距離を求める。（今回はユークリッド距離）
    3. 距離が近い順にソートする。
    4. 上位n件を保存する。
6. すべての特徴量ファイルの結果を結合する。
7. 上位n件を出力する。

実際のスクリプトは以下の通りです。今回は`n=10`としています。

```py:search.py
#!/usr/bin/env python3

import pathlib
import sys

import numpy as np
import onnxruntime
import PIL.Image

FEATURE_DIR = pathlib.Path("/mnt/feature/mobilenet_v3_large_100_224_feature_vector_v5")
ONNX_MODEL_PATH = pathlib.Path(
    "/mnt/model/mobilenet_v3_large_100_224_feature_vector_v5.onnx"
)

onnx_session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH))

image_path = pathlib.Path(sys.argv[1])

image = PIL.Image.open(image_path)
image = image.convert("RGB")
image = image.resize((224, 224))
image = np.array(image, dtype=np.float32)
image = image / 255

query_feature = onnx_session.run(
    ["feature_vector"], {"inputs:0": np.expand_dims(image, 0)}
)[0][0]

results = []
limit = 10

for blob_index in range(20):
    object_ids_path = FEATURE_DIR / "{:04d}.object_ids.npy".format(blob_index)
    features_path = FEATURE_DIR / "{:04d}.features.npy".format(blob_index)
    if not object_ids_path.exists():
        break

    object_ids = np.load(object_ids_path)
    features = np.load(features_path)
    assert len(object_ids) == len(features)

    query_features = np.tile(query_feature, (len(features), 1))

    distances = np.linalg.norm(query_features - features, axis=1)
    distance_indexes = np.argsort(distances)[:limit]
    results.extend(zip(object_ids[distance_indexes], distances[distance_indexes]))

for object_id, distance in sorted(results, key=lambda item: item[1])[0:limit]:
    print("{} {}".format(object_id, distance))
```

コマンドライン引数にクエリ画像のファイルパスを指定して実行してみます。

```
$ time ./src/search.py /mnt/media/object/55/77/5577b06378df4cbf5fa04237ac767205a944a360.jpg
5577b06378df4cbf5fa04237ac767205a944a360.jpg 0.0
877803e4c546ce3548b088cf734ee83a0c722a5a.jpg 0.39354559779167175
d5d1e73cb23b79ec05743c97bedd2f770adcbcaf.jpg 8.174514770507812
b6031e0d869b7b5e397e82f5db9dfaacd384f1a0.jpg 8.18053913116455
29e6a6c5885ff27a99d358065d3abb7856c723f3.jpg 8.67011833190918
924beb3a39b67a6c7e582d093d46928188e4be29.jpg 8.68422794342041
8e3aa1aaaaceb2bad287585c766927e5f1f218ca.jpg 10.661015510559082
d96c72edcf3062732f8c37ff339772b070cd195f.jpg 10.730582237243652
21793eb39f11ec24f42a643dec695264955a2300.jpg 11.072492599487305
42b2a6040589e68e734c4d1499b263992a2c5956.jpg 11.31399154663086

real    0m1.660s
user    0m1.307s
sys     0m1.169s
```

検索結果が表示されました。『[けしからん画像分類器を作ってみる](202102-pornography-classifier-1)』シリーズで使用した「けしからん画像」データセットを使用しているため、実際の検索結果をお見せできないのが残念です。

1番目は距離`0.0`となっています。クエリ画像として指定した`5577b06378df4cbf5fa04237ac767205a944a360.jpg`はインデックスにも含まれているため、距離ゼロとなっています。ちゃんと動作しているようですね。

2番目の距離`0.39...`の画像は、確かに似た画像でした。というか、同じ画像の色調が調整された画像でした。

それ以降も、同じ人物が写っている、同じ構図、同じ色調など、確かに「似ている画像」が出力されています。
距離が`11`を越えた辺りから「似ている…かな？」となってきます。

# マルチスレッド化

何となく、それっぽい類似画像検索を行えるようになりました。ただ、目標としていた「1秒以内」が実現できておらず、1.6秒を要しています。
1秒以内を目指すために、少し高速化をしてみましょう。

上記のスクリプトでは、約50MBの特徴量ファイルの読み込みと検索を順次行っていますが、これをマルチスレッド化してみます。
面倒なので、まずは特徴量ファイルの数だけスレッドを生成してみます。

```py:search.py
# （省略）

import threading

def search(object_ids_path, features_path, query_feature, limit, results):
    object_ids = np.load(object_ids_path)
    features = np.load(features_path)
    assert len(object_ids) == len(features)

    query_features = np.tile(query_feature, (len(features), 1))
    distances = np.linalg.norm(query_features - features, axis=1)
    distance_indexes = np.argsort(distances)[:limit]
    results.extend(zip(object_ids[distance_indexes], distances[distance_indexes]))


results = []
search_threads = []
limit = 10

for blob_index in range(20):
    object_ids_path = FEATURE_DIR / "{:04d}.object_ids.npy".format(blob_index)
    features_path = FEATURE_DIR / "{:04d}.features.npy".format(blob_index)
    if not object_ids_path.exists():
        break

    search_thread = threading.Thread(
        target=search,
        args=(object_ids_path, features_path, query_feature, limit, results),
    )
    search_thread.start()
    search_threads.append(search_thread)

for search_thread in search_threads:
    search_thread.join()

# （省略）
```

実行してみます。

```
$ time ./src/search.py /mnt/media/object/55/77/5577b06378df4cbf5fa04237ac767205a944a360.jpg
...
real    0m1.482s
user    0m1.541s
sys     0m1.743s
```

・・・0.1秒しか早くなりませんでした。そもそも検索処理は十分に高速なようです。

各処理の時間を測定してみると、モデルのロードに0.4秒、推論に0.5秒掛かっているようです。
高速化、最適化の第1ステップは「測定」。やっぱり手を抜いてはダメですね。

# 今日はここまで

今回は実際に類似画像を検索するところまで実施できました。ただ、目標とする「1秒以内」は実現できませんでした。

次回は検索処理をサーバ化し、モデルファイル、特徴量ファイルの読み込みを起動時に1回だけ行うようにして、高速化を図ってみたいと思います。今日はここまで！

『[類似画像検索ツールを作ってみる (5) 類似画像検索サーバ](202105-similar-search-5)』に続く。
