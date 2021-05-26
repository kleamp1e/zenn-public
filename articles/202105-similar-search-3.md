---
title: "類似画像検索ツールを作ってみる (3) 特徴化 その2"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# 目次

* [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)
* [類似画像検索ツールを作ってみる (2) 特徴化 その1](202105-similar-search-2)
* 類似画像検索ツールを作ってみる (3) 特徴化 その2（本記事）

# 特徴化

[前回](202105-similar-search-2)、MobileNet V3の学習済みモデルをONNXモデルに変換するところまでを実施しました。

今回は、検索対象である約10万枚の画像について特徴量を取得してみたいと思います。

# 画像ファイル一覧の作成

まずは、特徴量を取得する画像ファイルの一覧を作成します。10万枚もあるとファイル一覧の取得に地味に時間が掛かるので、予め一覧を生成しておきます。

```
$ cd /mnt/media/
$ find object -type f -name "*.jpg" | sort > objects.txt
$ wc -l objects.txt
100381 objects.txt

$ head -n 5 objects.txt
object/00/00/0000bada96b0c1c131ebdf41c37964f308cb21d3.jpg
object/00/01/0001594e435e50bf781a99d1de4446fde176fc03.jpg
object/00/03/000313d295f268775cf28a6322272b93def124f2.jpg
object/00/03/00036628c1855fc8648649c61bff4edc3b9760c4.jpg
object/00/03/0003af0e37f9fbac7180fad7836a0223dd4ac4f0.jpg
```

# 特徴量の取得

今回は、以下のPythonスクリプトをで特徴量を取得しました。大まかな流れは以下の通りです。

1. `objects.txt`から画像ファイル一覧を読み込む。
2. 画像ファイル一覧をシャッフルする。（簡易並列化のため）
3. ONNXモデルを読み込む。
4. 各画像ファイルについて:
    1. 出力先の特徴量ファイルパスを生成する。
    2. 出力先の特徴量ファイルパスが既に存在したらスキップする。（簡易並列化のため）
    3. 画像を読み込む。
    4. 画像を整形する。
    5. 特徴量を取得する。（推論する）
    6. 特徴量をファイルに書き込む。

```py:extract_all.py
#!/usr/bin/env python3

import pathlib
import random

import numpy as np
import onnxruntime
import PIL.Image

MEDIA_DIR = pathlib.Path("/mnt/media")
FEATURE_DIR = pathlib.Path("/mnt/feature")
ONNX_MODEL_PATH = pathlib.Path("/mnt/model/mobilenet_v3_large_100_224_feature_vector_v5.onnx")


def make_nested_id_path(dir, id, ext=""):
    return dir / id[0:2] / id[2:4] / (id + ext)


image_list_path = MEDIA_DIR / "objects.txt"
feature_base_dir = FEATURE_DIR / ONNX_MODEL_PATH.stem

with image_list_path.open("r") as file:
    image_paths = [MEDIA_DIR / path.rstrip() for path in file.readlines()]
print(len(image_paths))

random.shuffle(image_paths)

onnx_session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH))

for image_path in image_paths:
    print(image_path)
    object_id = image_path.name
    feature_path = make_nested_id_path(feature_base_dir, object_id, ".npy")
    print(feature_path)
    if feature_path.exists():
        print("skip")
        continue

    image = PIL.Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255

    feature = onnx_session.run(["feature_vector"], {"inputs:0": np.expand_dim(image, 0)})[0][0]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature)
```

上記のスクリプトでは愚直に処理を行っており、特に並列化などは行っていません。
ただ、「一覧のシャッフル」、「結果が存在したらスキップ」という処理を組み込んでいるので、複数のプロセスを起動するだけで簡易的に並列化できます。
マルチスレッド化、推論のバッチ化などを行えばもっと高速に処理できそうですが、今回は簡単で確実な方法を選びました。

```
$ tmux
[1]$ ./extract_all.py
[2]$ ./extract_all.py
[3]$ ./extract_all.py
[4]$ ./extract_all.py
[5]$ ./extract_all.py
[6]$ ./extract_all.py
[7]$ ./extract_all.py
[8]$ ./extract_all.py
```

今回は徐々にプロセス数を増やし、最終的に8プロセスで並列実行しました。プロセスあたりのGPUメモリ使用量は350MBほどでした。
約10万枚の特徴量について、15分以内に処理を終えることができました。

# 特徴量の結合

上記のスクリプトで、画像1枚に対して1つの特徴量ファイルを得ることができました。
ただ、このままでは処理しづらいので、一定単位（ここでは1万枚）でまとめることにします。

特徴量ファイルを結合するスクリプトは以下の通りです。

```py:concat.py
#!/usr/bin/env python3

import pathlib

import numpy as np

FEATURE_DIR = pathlib.Path("/mnt/feature/mobilenet_v3_large_100_224_feature_vector_v5")

feature_paths = sorted(FEATURE_DIR.glob("*/*/*.npy"))
print(len(feature_paths))

maximum = 10000

object_ids = []
features = []
for feature_path in feature_paths[0:maximum]:
    print(feature_path)
    object_id = feature_path.stem
    feature = np.load(feature_path)
    object_ids.append(object_id)
    features.append(feature)

object_ids = np.array(object_ids)
features = np.array(features)

for output_index in range(20):
    object_ids_path = FEATURE_DIR / "{:04d}.object_ids.npy".format(output_index)
    features_path = FEATURE_DIR / "{:04d}.features.npy".format(output_index)
    if object_ids_path.exists():
        continue

    np.save(object_ids_path, object_ids)
    np.save(features_path, features)

    for feature_path in feature_paths[0:maximum]:
        feature_path.unlink()

    break
```

上記のスクリプトを11回実行し、`*.features.npy`、`*.object_ids.npy`のペアを11組作成しました。

```
# 11回実行する
$ ./concat.py

$ cd /mnt/feature/mobilenet_v3_large_100_224_feature_vector_v5/
$ ls -lh *.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:42 0000.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:42 0000.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0001.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0001.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0002.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0002.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0003.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0003.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0004.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0004.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0005.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0005.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0006.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0006.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0007.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0007.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:43 0008.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:43 0008.object_ids.npy
-rw-r--r-- 1 1000 1000  49M May 23 10:44 0009.features.npy
-rw-r--r-- 1 1000 1000 1.7M May 23 10:44 0009.object_ids.npy
-rw-r--r-- 1 1000 1000 1.9M May 23 10:44 0010.features.npy
-rw-r--r-- 1 1000 1000  66K May 23 10:44 0010.object_ids.npy
```

結合後の特徴量は、1万枚あたり約50MBになりました。
