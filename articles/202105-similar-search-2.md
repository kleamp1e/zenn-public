---
title: "類似画像検索ツールを作ってみる (2) 特徴化 その1"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# 目次

* [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)
* 類似画像検索ツールを作ってみる (2) 特徴化 その1（本記事）
* [類似画像検索ツールを作ってみる (3) 特徴化 その2](202105-similar-search-3)
* [類似画像検索ツールを作ってみる (4) 類似画像検索](202105-similar-search-4)
* [類似画像検索ツールを作ってみる (5) 類似画像検索サーバ](202105-similar-search-5)
* [類似画像検索ツールを作ってみる (6) Next.js + SVGで可視化](202106-similar-search-6)

# 特徴化

[前回](202105-similar-search-1)は、類似画像検索の戦略を決め、特徴量を抽出するモデル（特徴量抽出器）として「MobileNet V3」を選定しました。

今回は、特徴量抽出器を使って画像から特徴量を抽出する「特徴化」を行ってみたいと思います。

# TensorFlow Hubモデルの取得

MobileNet V3の学習済みモデルは[TensorFlow Hub](https://tfhub.dev/)から取得することができます。
しかも素晴らしいことに、画像分類器としてのモデルと、特徴量抽出器としてのモデルが別れており、用途によって使い分けやすくなっています。

今回は、特徴量抽出器としてのモデルである[imagenet/mobilenet_v3_large_100_224/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5)を使います。（執筆時点でのモデルのバージョンはv5）

以下のコマンドでSavedModel形式のモデルを取得し、展開しました。

```
$ wget \
  --output-document mobilenet_v3_large_100_224_feature_vector_v5.tar.gz \
  "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5?tf-hub-format=compressed"
$ ls -l mobilenet_v3_large_100_224_feature_vector_v5.tar.gz
-rw-r--r-- 1 root root 15840597 Feb  9 12:30 mobilenet_v3_large_100_224_feature_vector_v5.tar.gz

$ sha1sum mobilenet_v3_large_100_224_feature_vector_v5.tar.gz
b4181065be4258956e249ea56e27cbeb8306372b  mobilenet_v3_large_100_224_feature_vector_v5.tar.gz

$ mkdir mobilenet_v3_large_100_224_feature_vector_v5
$ tar zxfv mobilenet_v3_large_100_224_feature_vector_v5.tar.gz -C mobilenet_v3_large_100_224_feature_vector_v5
```

# ONNXモデルへの変換

取得したモデルはTensorFlow SavedModel形式で、その推論には当然TensorFlowが必要です。
TensorFlowは依存するライブラリが多く、CUDAなどのバージョンもシビアなので、今回はONNX形式に変換して、ONNX Runtimeで推論することにします。

以下のコマンドでSavedModel形式のモデルをONNX形式に変換しました。ONNX形式は単一ファイルなので扱いやすいですね。

```
$ python3 -m tf2onnx.convert \
  --saved-model mobilenet_v3_large_100_224_feature_vector_v5 \
  --output mobilenet_v3_large_100_224_feature_vector_v5.onnx

$ ls -l mobilenet_v3_large_100_224_feature_vector_v5.onnx
-rw-r--r-- 1 root root 16911818 May 22 15:46 mobilenet_v3_large_100_224_feature_vector_v5.onnx

$ sha1sum mobilenet_v3_large_100_224_feature_vector_v5.onnx
798e24e34c701a0158e3c7494d12fd8fa0f01a92  mobilenet_v3_large_100_224_feature_vector_v5.onnx
```

なお、TensorFlow Hubモデルの取得、ONNXモデルへの変換は、以下の`Dockerfile`、`requirements.txt`から生成したDockerコンテナ内で実行しました。

```Dockerfile:Dockerfile
FROM ubuntu:20.04
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-setuptools \
    wget \
  && rm --recursive --force /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools
WORKDIR /opt/app
COPY requirements.txt ./
RUN python3 -m pip install --requirement requirements.txt
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
```

```:requirements.txt
tensorflow==2.4.1
tf2onnx==1.8.4
```

# ONNXモデルでの推論

変換して出来上がったONNXモデルをONNX Runtimeで推論してみます。今回はGPU版を使用しました。

ひとまず`0`、`1`で埋められた画像から推論してみたところ、何やらそれっぽい特徴量が出力されました。

```
$ nvidia-smi | head -n 12
Sat May 22 16:09:53 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 00000000:2D:00.0 Off |                  N/A |
|  0%   32C    P8    12W / 195W |      0MiB /  8117MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

$ python3
Python 3.8.5 (default, Jan 27 2021, 15:41:15)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> import onnxruntime
>>> onnxruntime.get_device()
'GPU'
>>> session = onnxruntime.InferenceSession("/mnt/model/mobilenet_v3_large_100_224_feature_vector_v5.onnx")
>>> session.get_providers()
['CUDAExecutionProvider', 'CPUExecutionProvider']
>>> session.run(["feature_vector"], {"inputs:0": np.zeros((1, 224, 224, 3), dtype=np.float32)})
[array([[-0.19617268, -0.32971063,  0.01838597, ...,  1.1617106 ,
        -0.36636546, -0.33906454]], dtype=float32)]
>>> session.run(["feature_vector"], {"inputs:0": np.ones((1, 224, 224, 3), dtype=np.float32)})
[array([[ 0.46933305,  0.20881261,  0.30621225, ..., -0.3671668 ,
        -0.31486648, -0.33280426]], dtype=float32)]
```

なお、ONNX Runtimeでの推論は、以下の`Dockerfile`、`requirements.txt`から生成したDockerコンテナ内で実行しました。

```Dockerfile:Dockerfile
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-setuptools \
  && rm --recursive --force /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools
WORKDIR /opt/app
COPY requirements.txt ./
RUN python3 -m pip install --requirement requirements.txt
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
```

```:requirements.txt
onnxruntime-gpu==1.7.0
```

# 今日はここまで

今回はMobileNet V3の学習済みモデルをTensorFlow Hubから取得し、ONNX形式に変換した上でONNX Runtimeで推論してみました。
本当は検索対象の画像の特徴量を取得するところまで行きたかったのですが、体力切れなので今日はここまで。

次回は、検索対象である10万枚の画像の特徴化を行ってみたいと思っています。

『[類似画像検索ツールを作ってみる (3) 特徴化 その2](202105-similar-search-3)』に続く。
