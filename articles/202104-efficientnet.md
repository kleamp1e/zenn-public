---
title: "EfficientNet B0〜B7で画像分類器を転移学習してみる"
emoji: "🎓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "keras"]
published: false
---

# はじめに

「[けしからん画像分類器を作ってみる](202102-pornography-classifier-1)」シリーズで、画像分類モデルとしてEfficientNetを使ってみました。
今回はEfficientNetのバリエーションであるB0〜B7について、実際に学習を行って、実例での相違を見ていきます。

# データ

使用した画像データには1クラスのラベル（`0`と`1`の2値分類）が付けられており、学習データ、検証データ、テストデータは8:1:1の比率に近づくようにハッシュ値ベースで切り出しています。
また、検証データ、テストデータについてはラベル数が同数になるように調整しています。

データ数は以下の通りです。

| 種別 | ラベル`0` | ラベル`1` | 合計 |
|:---|---:|---:|---:|
| 学習データ | 6,126 | 13,532 | 19,658 |
| 検証データ | 747 | 747 | 1,494 |
| テストデータ | 744 | 744 | 1,488 |
| 合計 | 7,617 | 15,023 | 22,640 |

# 環境

使用した環境は以下の通りです。学習はDockerコンテナ内で実施しています。

* ハードウェア:
    * CPU: AMD Ryzen 7 3700X（8コア/16スレッド）
    * メモリ: 64GB
    * GPU: GeForce GTX 1070（メモリ8GB）
* ソフトウェア:
    * OS: Ubuntu 20.04.2 LTS
    * Docker: 19.03.8
    * NVIDIAドライバ: 460.39
    * Dockerコンテナ内:
        * CUDA: 11.0.3
        * cuDNN: 8
        * Python: 3.8.5
        * TensorFlow: 2.4.1

# フレームワーク

機械学習フレームワークとしては「Keras」を使用しました。
Kerasでは、TensorFlow Hubから取得したEfficientNetの学習済みモデルを簡単に使うことができます。

# 学習結果

それぞれのモデルバリエーションについて1回ずつ、10エポックの転移学習を行い、結果は以下の通りでした。
Bxの数字が大きくなるにつれてモデルサイズが大きくなりますが、それに応じてテストデータで評価した精度が高まり、学習時間が延びる結果となりました。

最小のB0と最大のB7を比べると、精度は+5%、学習時間は31倍となりました。

| モデル | 画像サイズ [px] | バッチサイズ | ファイルサイズ | パラメータ数 | テストLoss | B0との差 | テスト精度 | B0との差 | 学習時間 | B0との倍率 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet B0 | 224 | 512 | 16,493,696 |  4,050,845 | 0.3236 | -       | 0.8516 | -      |   9m48.603s |  1.0 |
| EfficientNet B1 | 240 | 256 | 26,702,696 |  6,576,513 | 0.2958 | -0.0278 | 0.8781 | +0.027 |  12m29.039s |  1.3 |
| EfficientNet B2 | 260 | 256 | 31,477,888 |  7,769,971 | 0.2696 | -0.0540 | 0.8750 | +0.023 |  15m38.572s |  1.6 |
| EfficientNet B3 | 300 | 128 | 43,586,544 | 10,785,065 | 0.2599 | -0.0637 | 0.8864 | +0.035 |  24m24.092s |  2.5 |
| EfficientNet B4 | 380 | 128 | 71,244,456 | 17,675,609 | 0.2387 | -0.0849 | 0.8991 | +0.048 |  49m44.266s |  5.1 |
| EfficientNet B5 | 456 | 64 | 114,713,416 | 28,515,569 | 0.2347 | -0.0889 | 0.9042 | +0.053 |  98m26.809s | 10.0 |
| EfficientNet B6 | 528 | 32 | 164,599,656 | 40,962,441 | 0.2351 | -0.0885 | 0.9042 | +0.053 | 177m17.915s | 18.1 |
| EfficientNet B7 | 600 | 32 | 257,306,968 | 64,100,241 | 0.2382 | -0.0854 | 0.9076 | +0.056 | 306m44.716s | 31.3 |

共通のネットワーク構造、ハイパーパラメータなどは以下の通りです。

* バックボーン（Backbone）: EfficientNet
* ヘッド（Head）: 全結合層（FC: Fully Connected） + Sigmoid
* オプティマイザ（Optimizer）: Adam
* エポック数: 10

モデルのファイルサイズとパラメータ数の比較を以下に示します。

![](https://storage.googleapis.com/zenn-user-upload/wvcv0s080an02jwfn19wktw6ebw5)

テストデータで評価した精度とLoss値の比較を以下に示します。

![](https://storage.googleapis.com/zenn-user-upload/rjwmytnxadunuu1phhf1zunnkt7s)

精度とパラメータ数の比較を以下に示します。B1とB2が入れ替わっていますが誤差の範囲内かと思います。

![](https://storage.googleapis.com/zenn-user-upload/pu51ingqie3oryoo6rta2il38mg0)

各モデルにおける精度、Loss値について、学習データ、検証データにおける変遷は以下の通りです。

![](https://storage.googleapis.com/zenn-user-upload/7rs4ppl0n5rx18f97nkn25hbxm7s)

# 学習処理

学習に使用したPythonスクリプトは以下の通りです。B0〜B7について、コメントアウトしている箇所を調整しつつ実行しました。

```py
#!/usr/bin/env python3

import datetime

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

BATCH_SIZE = 512 # B0
# BATCH_SIZE = 256 # B1, B2
# BATCH_SIZE = 128 # B3, B4
# BATCH_SIZE = 64 # B5
# BATCH_SIZE = 32 # B6, B7

TARGET_SIZE = 224 # B0
# TARGET_SIZE = 240 # B1
# TARGET_SIZE = 260 # B2
# TARGET_SIZE = 300 # B3
# TARGET_SIZE = 380 # B4
# TARGET_SIZE = 456 # B5
# TARGET_SIZE = 528 # B6
# TARGET_SIZE = 600 # B7

EPOCHS = 10

model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", # B0
            # "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1", # B1
            # "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1", # B2
            # "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1", # B3
            # "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1", # B4
            # "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1", # B5
            # "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1", # B6
            # "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", # B7
            trainable=False,
        ),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.build([None, TARGET_SIZE, TARGET_SIZE, 3])
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
train_generator = train_datagen.flow_from_directory(
    "/tmp/cache/pornography/train",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=(1.0 / 255)
)
validation_generator = validation_datagen.flow_from_directory(
    "/tmp/cache/pornography/validation",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
test_generator = test_datagen.flow_from_directory(
    "/tmp/cache/pornography/test",
    target_size=(TARGET_SIZE, TARGET_SIZE),
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

# 最後に

今回の結果を見る限り、B3、B4あたりが学習時間と精度のバランスが良いかなと思いました。

ファインチューニングした場合の相違についても、いつか調べてみたいと思います。
