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
Kerasで学習した後、SavedModel形式でモデルを保存すると、このツールで変換することができます。
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

```:requirements.txt
onnxruntime==1.7.0
tensorflow-hub==0.11.0
tensorflow==2.4.1
tf2onnx==1.8.4
```

## モデルを生成する

今回はTensorFlow HubにあるEfficientNet B0をそのまま保存することでモデルファイルを生成します。
推論結果を変換前後で確認することで、変換の成否を判断します。

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

実行例を以下に示します。全結合層が乱数で初期化されているため、モデルを保存する度に値は変わることにご注意ください。

```
$ ./predict_keras.py
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
[[0.5295639]
 [0.5148043]]
```

## モデルを変換する

`tf2onnx`を使ってモデルを変換します。

```sh:convert.sh
#!/bin/bash
python3 -m tf2onnx.convert --saved-model efficientnet-b0 --output efficientnet-b0.onnx
```

実行例を以下に示します。いくつか警告が出力されていますが今回は無視します。

```
$ ./convert.sh
/usr/lib/python3.8/runpy.py:127: RuntimeWarning: 'tf2onnx.convert' found in sys.modules after import of package 'tf2onnx', but prior to execution of 'tf2onnx.convert'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
2021-04-23 00:14:01,125 - WARNING - '--tag' not specified for saved_model. Using --tag serve
2021-04-23 00:14:06,873 - INFO - Signatures found in model: [serving_default].
2021-04-23 00:14:06,873 - WARNING - '--signature_def' not specified, using first signature: serving_default
2021-04-23 00:14:06,873 - INFO - Output names: ['dense']
WARNING:tensorflow:From /usr/local/lib/python3.8/dist-packages/tf2onnx/tf_loader.py:557: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
2021-04-23 00:14:09,553 - WARNING - From /usr/local/lib/python3.8/dist-packages/tf2onnx/tf_loader.py:557: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
2021-04-23 00:14:10,275 - INFO - Using tensorflow=2.4.1, onnx=1.9.0, tf2onnx=1.8.4/cd55bf
2021-04-23 00:14:10,275 - INFO - Using opset <onnx, 9>
2021-04-23 00:14:11,053 - INFO - Computed 0 values for constant folding
2021-04-23 00:14:13,763 - INFO - Optimizing ONNX model
2021-04-23 00:14:17,027 - INFO - After optimization: BatchNormalization -42 (49->7), Const -240 (442->202), Identity -926 (926->0), Squeeze -16 (16->0), Transpose -275 (276->1), Unsqueeze -64 (64->0)
2021-04-23 00:14:17,056 - INFO -
2021-04-23 00:14:17,057 - INFO - Successfully converted TensorFlow model efficientnet-b0 to ONNX
2021-04-23 00:14:17,057 - INFO - Model inputs: ['keras_layer_input:0']
2021-04-23 00:14:17,057 - INFO - Model outputs: ['dense']
2021-04-23 00:14:17,057 - INFO - ONNX model is saved at efficientnet-b0.onnx
```

## ONNXで推論する

変換したONNXモデルを使って推論してみます。

```py:predict_onnx.py
#!/usr/bin/env python3

import numpy as np
import onnxruntime

session = onnxruntime.InferenceSession("efficientnet-b0.onnx")

images = np.array([
  np.zeros((224, 224, 3), dtype=np.float32),
  np.ones((224, 224, 3), dtype=np.float32),
])

results = session.run(["dense"], {"keras_layer_input:0": images})
print(results)
```

実行例を以下に示します。

```
$ ./predict_onnx.py
[array([[0.52956396],
       [0.5148051 ]], dtype=float32)]
```

Kerasでの推論結果とは厳密には一致しませんが、小数点第5位まで一致しているので問題はなさそうです。

# keras2onnxで変換する

続いて、[keras2onnx](https://github.com/onnx/keras-onnx)を使って変換してみます。

`tf2onnx`の変換についてはサクッと一発で成功しましたが、`keras2onnx`を使った変換にはなかなか難儀しました。
注意点は以下の通りです。

* `keras2onnx`はTensorFlow v2.4に対応しておらず、TensorFlow v2.2までしか対応していません。
* TensorFlow v2.4で生成したモデルはTensorFlow v2.2では読み込むことができなかったため、学習もv2.2で行う必要がありました。
* TensorFlow v2.2を使うためにはCUDA 11.0/cuDNN 8ではなくCUDA 10.1/cuDNN 7を使う必要がありました。

## Dockerイメージをビルドする

今回は以下の`Dockerfile`、`requirements.txt`を使用しました。

```Dockerfile:Dockerfile
```

```:requirements.txt
```

## モデルを生成する

基本的な手順は`tf2onnx`の場合と同様ですが、なぜか`fit`か`predict`を呼び出さないとモデルの保存時にエラーが発生しました。

```py:save_model.py
#!/usr/bin/env python3

import numpy as np
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
model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32))
model.save("efficientnet-b0")
```

実行例は以下の通りです。

```
$ ./save_model.py
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
keras_layer (KerasLayer)     multiple                  4049564
_________________________________________________________________
dense (Dense)                multiple                  1281
=================================================================
Total params: 4,050,845
Trainable params: 1,281
Non-trainable params: 4,049,564
_________________________________________________________________
2021-04-23 00:35:08.379870: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
```

## Kerasで推論する

ソースコードは`tf2onnx`の場合と同様なので省略します。

実行例を以下に示します。

```
$ ./predict_keras.py
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
[[0.41439614]
 [0.43379608]]
```

## モデルを変換する

`keras2onnx`を使ってモデルを変換します。

```py:convert.py
#!/usr/bin/env python3

import keras2onnx
import onnx
import tensorflow as tf

model = tf.keras.models.load_model("efficientnet-b0")
onnx_model = keras2onnx.convert_keras(model, "efficientnet-b0")
onnx.save_model(onnx_model, "efficientnet-b0.onnx")
```

実行例を以下に示します。

```
$ ./convert.py
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
tf executing eager_mode: True
tf.keras model eager_mode: False
2021-04-23 00:41:39.446674: W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARN: No corresponding ONNX op matches the tf.op node sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/tf_op_layer_BroadcastTo_1/PartitionedCall/BroadcastTo_1 of type BroadcastTo
      The generated ONNX model needs run with the custom op supports.
The ONNX operator number change on the optimization: 4007 -> 492
```

## ONNXで推論する

ソースコードは`tf2onnx`の場合と同様なので省略します。

実行例を以下に示します。

```
root@a30864c4b2b2:/mnt/app# ./predict_onnx.py
Traceback (most recent call last):
  File "./predict_onnx.py", line 6, in <module>
    session = onnxruntime.InferenceSession("efficientnet-b0.onnx")
  File "/usr/local/lib/python3.6/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 280, in __init__
    self._create_inference_session(providers, provider_options)
  File "/usr/local/lib/python3.6/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 307, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from efficientnet-b0.onnx failed:Fatal error: BroadcastTo is not a registered function/op
```

・・・エラーになっちゃいました。
変換時のメッセージにもある通り、ONNXではサポートされていないオペレータ`BroadcastTo`が原因かと思います。
カスタムオペレータを追加すれば対応できるかもしれませんが、`tf2onnx`での変換は成功しているので調査は中断しました。

# 結論

`tf2onnx`を使いましょう。
