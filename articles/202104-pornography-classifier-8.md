---
title: "けしからん画像分類器を作ってみる (8) 学習 その2"
emoji: "👙"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "keras"]
published: false
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

# 続 学習

[前回](202104-pornography-classifier-7)はデータの分割までを行いました。
今回は実際に学習を行ってみたいと思います。

# 結論

最初に結論を書いておくと、精度（Accuracy）が85%程度の画像分類モデルを得ることができました。
今回はほとんど工夫はしておらず、かなりシンプルな構成です。様々な手法を適用することで、さらに数%は精度を高められそうな気がしています。

# データの前処理

Kerasを使ってモデルを学習するにあたり、データの読み込みに`tf.keras.preprocessing.image.ImageDataGenerator`を使用しました。（手抜きのために）

`ImageDataGenerator`は特定のディレクトリ構造が前提となっていますので、既存の画像データをその構造に配置します。
ファイルをコピーするのは無駄なので、今回はシンボリックリンクを作成しています。

具体的には`/tmp/cache/pornography/<タイプ>/<ラベル>/オブジェクトID.jpg`みたいなパスでシンボリックリンクを作成します。

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

```Dockerfile
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
ENV TZ=Asia/Tokyo
```

また、`requirements.txt`は以下の通りです。関係する主要なパッケージのみ記載しています。

```txt:requirements.txt
numpy==1.19.5
pandas==1.2.3
Pillow==8.2.0
scipy==1.6.2
tensorflow-estimator==2.4.0
tensorflow-hub==0.11.0
tensorflow==2.4.1
```

# 学習

さて、いよいよ学習です。主なメタパラメータなどは以下の通りです。

* バックボーン（Backbone）: EfficientNet B0
* ヘッド（Head）: 全結合層（FC: Fully Connected） + Sigmoid
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

実行結果は以下の通りです。

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

良い感じに学習が進んでいますね。テストデータによる評価は精度約85%でした。
