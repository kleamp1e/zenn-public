---
title: "類似画像検索ツールを作ってみる (2) 特徴化"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: false
---

# 目次

* [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)
* （この記事）

# 特徴化

[前回](202105-similar-search-1)は、類似画像検索の戦略を決め、特徴量を抽出するモデル（特徴量抽出器）として「MobileNet V3」を選定しました。

今回は、特徴量抽出器を使って画像から特徴量を抽出する「特徴化」を行ってみたいと思います。

# TensorFlow Hubモデルの取得

MobileNet V3の学習済みモデルは[TensorFlow Hub](https://tfhub.dev/)から取得することができます。
しかも素晴らしいことに、画像分類器としてのモデルと、特徴量抽出器としてのモデルが別れており、用途によって使い分けやすくなっています。

今回は、特徴量抽出器としてのモデルである[imagenet/mobilenet_v3_large_100_224/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5)を使います。（執筆時点でのモデルのバージョンはv5）

以下のようなコマンドでSavedModel形式のモデルを取得し、展開します。

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
TensorFlowは依存するライブラリが多く、CUDAなどのバージョンもシビアなので、今回はONNX形式変換して、ONNX Runtimeで推論することにします。

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
