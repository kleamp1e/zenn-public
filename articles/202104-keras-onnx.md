---
title: "EfficientNet B0のKerasモデルをONNXモデルに変換して推論する"
emoji: "🎓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "keras", "onnx"]
published: false
---

# はじめに

「[けしからん画像分類器を作ってみる](202102-pornography-classifier-1)」シリーズでは、KerasとEfficientNet B0を使って画像分類器を実装しました。
その画像分類モデルを、ONNXモデルに変換して推論してみたいと思います。

# ONNXとは？

ONNX（Open Neural Network Exchange）は、Facebook、Microsoftが主導して、機械学習フレームワークの相互運用を実現するためのプロジェクトです。詳しくは、まぁ、ググってください。

* 公式: [ONNX | Home](https://onnx.ai/)
* Wikipedia: [Open Neural Network Exchange](https://ja.wikipedia.org/wiki/Open_Neural_Network_Exchange)

# モデルを変換する方法

KerasのモデルをONNXのモデルに変換する方法は、大きく以下の2つがあります。

* tf2onnxで変換する ← オススメ！
* keras2onnxで変換する

前者が圧倒的にオススメです。勉強のために後者も試してみましたが、なかなか大変でした。

どちらの変換、推論も以下の環境で実行しました。Docker内で実行しており、今回はGPUは使用していません。

* ハードウェア:
    * CPU: AMD Ryzen 7 3700X（8コア/16スレッド）
    * メモリ: 64GB
    * GPU: GeForce GTX 1070（メモリ8GB）
* ソフトウェア:
    * OS: Ubuntu 20.04.2 LTS
    * Docker: 19.03.8
    * NVIDIAドライバ: 460.39

# tf2onnxで変換する

[tf2onnx](https://github.com/onnx/tensorflow-onnx)は、TensorFlowのモデルをONNXのモデルに変換するツールです。
Kerasで学習した後、SavedModel形式でモデルを保存すると、このツールで変換することができまず。
Keras H5形式には対応していないのでご注意ください。

今回は学習は行わず、変換、推論だけを行っています。

## Dockerイメージをビルドする

今回は以下の`Dockerfile`、`requirements.txt`を使用しました。

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

```text:requirements.txt
numpy==1.19.5
onnxruntime==1.7.0
tensorflow-hub==0.11.0
tensorflow==2.4.1
tf2onnx==1.8.4
```

## モデルを生成する

今回はTensorFlow HubにあるEfficientNet B0をそのまま保存することでモデルファイルを生成します。
全結合層の重み、バイアスはランダムな値で初期化されているため、その部分を含めた推論結果が変換前後で確認することで、変換の成否を判断します。

```py:save_model.py
#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub

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
model.summary()
model.save("efficientnet-b0")
```

実行例を以下に示します。成功すると`efficientnet-b0`ディレクトリが生成されます。

```
$ ./save_model.py
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
2021-04-23 00:07:12.365759: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
```

## Kerasで推論する

ONNXモデルに変換する前に、Kerasで推論できること、その結果を確認しておきましょう。

```py:predict_keras.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("efficientnet-b0")

images = np.array([
  np.zeros((224, 224, 3), dtype=np.float32),
  np.ones((224, 224, 3), dtype=np.float32),
])

results = model.predict(images)
print(results)
```

実行例を以下に示します。

```
$ ./predict_keras.py
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
[[0.5295639]
 [0.5148043]]
```

# keras2onnxで変換する
