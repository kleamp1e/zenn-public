---
title: "けしからん画像分類器を作ってみる (8) 学習 その2"
emoji: "👙"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "keras"]
published: true
---

# 目次

* [けしからん画像分類器を作ってみる (1) 序章](202102-pornography-classifier-1)
* [けしからん画像分類器を作ってみる (2) データ収集 その1](202102-pornography-classifier-2)
* [けしからん画像分類器を作ってみる (3) データ収集 その2](202102-pornography-classifier-3)
* [けしからん画像分類器を作ってみる (4) データ収集 その3](202103-pornography-classifier-4)
* [けしからん画像分類器を作ってみる (5) データ管理 その1](202103-pornography-classifier-5)
* [けしからん画像分類器を作ってみる (6) データ管理 その2](202103-pornography-classifier-6)
* [けしからん画像分類器を作ってみる (7) 学習 その1](202104-pornography-classifier-7)
* けしからん画像分類器を作ってみる (8) 学習 その2（本記事）
* [けしからん画像分類器を作ってみる (9) 推論](202104-pornography-classifier-9)
* 番外編:
    * [EfficientNet B0〜B7で画像分類器を転移学習してみる](202104-efficientnet)

# 続 学習

[前回](202104-pornography-classifier-7)はデータの分割までを行いました。今回は実際に学習を行っていきます。

# 結論

最初に結論を書いておくと、精度（Accuracy）が85%程度の画像分類モデルを得ることができました。
初回なのでほとんど工夫はしておらず、かなりシンプルな構成です。様々な手法を適用することで、さらに数%は精度を高められそうな気がしています。

# データの前処理

Kerasを使ってモデルを学習するにあたり、データの読み込みに`tf.keras.preprocessing.image.ImageDataGenerator`を使用しました。（手抜きのために）

`ImageDataGenerator`は特定のディレクトリ構造が前提となっていますので、既存の画像データをその構造に配置します。
ファイルをコピーするのは無駄なので、今回はシンボリックリンクを作成しています。

具体的には`/tmp/cache/pornography/<タイプ>/<ラベル>/オブジェクトID.jpg`みたいなパスでシンボリックリンクを作成します。
スクリプトは以下の通りです。

```py:make_link.py
#!/usr/bin/env python3

import os
import pathlib

import pandas as pd

def make_nested_id_path(dir, id, ext=""):
    return dir / id[0:2] / id[2:4] / (id + ext)

MEDIA_DIR = pathlib.Path(os.environ.get("MEDIA_DIR"))
OBJECT_DIR = MEDIA_DIR / "object"
CACHE_DIR = pathlib.Path("/tmp/cache/pornography")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("split_label.csv")

for type in ["train", "test", "validation"]:
  for index, (object_id, value) in df[["objectId", "value"]][df["type"] == type].iterrows():
      image_path = make_nested_id_path(OBJECT_DIR, object_id)
      target_path = CACHE_DIR / type / str(value) / object_id
      target_path.parent.mkdir(parents=True, exist_ok=True)
      if not target_path.exists():
          target_path.symlink_to(image_path)
```

# 学習の環境

学習はDockerコンテナ内で実行しました。`Dockerfile`は以下の通りです。

```Dockerfile:Dockerfile
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    build-essential \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-setuptools \
    tzdata \
  && rm --recursive --force /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools
WORKDIR /opt/app
COPY requirements.txt ./
RUN python3 -m pip install --requirement requirements.txt
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
```

また、`requirements.txt`は以下の通りです。関係する主要なパッケージのみ記載しています。

```:requirements.txt
numpy==1.19.5
pandas==1.2.3
Pillow==8.2.0
scipy==1.6.2
tensorflow-estimator==2.4.0
tensorflow-hub==0.11.0
tensorflow==2.4.1
```

なお、NVIDIAのGPUを使用するため、[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)が必要です。

# 学習

さて、いよいよ学習です。主なハイパーパラメータなどは以下の通りです。

* バックボーン（Backbone）: EfficientNet B0
* ヘッド（Head）: 全結合層（FC: Fully Connected） + Sigmoid
* オプティマイザ（Optimizer）: Adam
* 入力画像サイズ: 224x224
* バッチサイズ: 512
* エポック数: 10

学習に使用したスクリプトは以下の通りです。

```py:train.py
#!/usr/bin/env python3

import datetime

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

BATCH_SIZE = 512
TARGET_SIZE = (224, 224)
EPOCHS = 10

model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
            trainable=False,
        ),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.build([None, 224, 224, 3])
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
train_generator = train_datagen.flow_from_directory(
    "/tmp/cache/pornography/train",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=(1.0 / 255)
)
validation_generator = validation_datagen.flow_from_directory(
    "/tmp/cache/pornography/validation",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
test_generator = test_datagen.flow_from_directory(
    "/tmp/cache/pornography/test",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=EPOCHS,
    workers=8,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir="log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            histogram_freq=1,
        )
    ],
)

model.evaluate(x=test_generator, steps=(test_generator.n // BATCH_SIZE), workers=8)

model.save("model.h5")
```

実行結果は以下の通りです。学習時間は約10分でした。

```
$ ./src/train.py
...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
keras_layer (KerasLayer)     (None, 1280)              4049564
_________________________________________________________________
dense (Dense)                (None, 1)                 1281
=================================================================
Total params: 4,050,845
Trainable params: 1,281
Non-trainable params: 4,049,564
_________________________________________________________________
Found 19658 images belonging to 2 classes.
Found 1494 images belonging to 2 classes.
Found 1488 images belonging to 2 classes.
...
Epoch 1/10
38/38 [==============================] - 68s 1s/step - loss: 0.6039 - accuracy: 0.6356 - val_loss: 0.4211 - val_accuracy: 0.8115
Epoch 2/10
38/38 [==============================] - 54s 1s/step - loss: 0.3425 - accuracy: 0.8594 - val_loss: 0.3497 - val_accuracy: 0.8447
Epoch 3/10
38/38 [==============================] - 52s 1s/step - loss: 0.3100 - accuracy: 0.8672 - val_loss: 0.3444 - val_accuracy: 0.8486
Epoch 4/10
38/38 [==============================] - 53s 1s/step - loss: 0.2925 - accuracy: 0.8726 - val_loss: 0.3309 - val_accuracy: 0.8506
Epoch 5/10
38/38 [==============================] - 53s 1s/step - loss: 0.2813 - accuracy: 0.8815 - val_loss: 0.3483 - val_accuracy: 0.8457
Epoch 6/10
38/38 [==============================] - 54s 1s/step - loss: 0.2754 - accuracy: 0.8820 - val_loss: 0.3282 - val_accuracy: 0.8545
Epoch 7/10
38/38 [==============================] - 52s 1s/step - loss: 0.2696 - accuracy: 0.8855 - val_loss: 0.3247 - val_accuracy: 0.8564
Epoch 8/10
38/38 [==============================] - 53s 1s/step - loss: 0.2603 - accuracy: 0.8900 - val_loss: 0.3111 - val_accuracy: 0.8672
Epoch 9/10
38/38 [==============================] - 53s 1s/step - loss: 0.2597 - accuracy: 0.8881 - val_loss: 0.3178 - val_accuracy: 0.8564
Epoch 10/10
38/38 [==============================] - 53s 1s/step - loss: 0.2523 - accuracy: 0.8911 - val_loss: 0.3320 - val_accuracy: 0.8477
2/2 [==============================] - 8s 1s/step - loss: 0.3170 - accuracy: 0.8535
```

良い感じに学習が進んでいますね。テストデータによる評価では、精度約85%となりました。

TensorBoardで確認した精度のチャートは以下の通りです。

![](https://storage.googleapis.com/zenn-user-upload/kzc5xkuuasgxfsj2up4st4ozc4lf)

同じく、Lossのチャートは以下の通りです。

![](https://storage.googleapis.com/zenn-user-upload/wn45s1d1g0p8617mf3xgobx4wha4)

# 今後の展望

特に工夫なく比較的高精度な画像分類モデルを作成することができました。ただ、タスクとしては割と単純な方だと思うので、少なくとも精度90%は越えたいところです。
工夫できそうな点をいくつか上げてみます。

* 精度向上のために:
    * **エポック数を増やす:** チャートを見る限り、エポック数を増やせばまだ伸びそうな気がします。1エポック1分ほどで終わるので、比較的低コストです。
    * **学習データを増やす:** データを増やせば精度が上がる可能性があります。ただ、なかなか高コストです。
    * **データの質を高める:** 初期にラベル付けしたデータは、ラベル付け基準が明確で無かったため、質の悪いラベルとなっている可能性があります。すべてのラベルを見直すことで精度の向上が期待できます。ただし、これも高コストです。
    * **クラスの偏りを考慮する:** 今回は学習データに含まれる「けしからん画像」が「けしからんくない画像」の約2倍存在し、不均衡が生じています。不均衡を考慮することで精度の向上が期待できます。
    * **データ拡張（Data Augmentation）する:** 反転や回転、ノイズを加えるなどしてデータを水増しすることで精度が向上する可能性があります。`ImageDataGenerator`を使っているので手軽に試すことができます。
    * **B0より大きいモデルを使用する:** EfficientNetにはB0からB7までのバリエーションがあり、数字が大きいほど大規模（だけれど処理負荷も高い）となります。より大きなモデルに変更することで、精度が向上する可能性があります。
    * **ファインチューニングする:** 今回はバックボーン部分は学習せず、単なる特徴量抽出器として使用し、ヘッドのみを学習しました。バックボーンも含めて学習することで精度の向上が期待できます。
* 高速化のために:
    * **画像をすべてメモリに読み込んでおく:** 今回使用した画像は約2万枚と比較的少なく、画像サイズも小さいため4GBに収まります。すべての画像を予めメモリに読み込んでおくことで高速化が期待できます。
    * **事前に画像を縮小する:** バックボーンへの入力は224x224と比較的小さいサイズですが、このサイズへの縮小処理が毎回行われていて効率が悪いです。データ拡張との兼ね合いもありますが、事前に処理できるのであれば高速化できる可能性があります。
    * **バックボーンとヘッドの学習を分離する:** これもデータ拡張との兼ね合いがありますが、画像を予めバックボーンを使って特徴量に変換しておけば、ヘッドのみの学習で済みます。ファインチューニングする場合は、もちろんこの手段は使えません。

# 学習したモデルについて

学習したモデルは、要望があれば公開します。16MBほどあるので、ファイルを置ける場所の提案も含まれていると嬉しいです。

# 今回はここまで

ついに学習まで到達しました。今後は、ラベル付け、高精度化、高速化、評価、推論、蒸留などについて書きたいと思っています。今日はここまで！

「[けしからん画像分類器を作ってみる (9) 推論](202104-pornography-classifier-9)」に続く。
