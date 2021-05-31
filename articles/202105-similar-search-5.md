---
title: "類似画像検索ツールを作ってみる (5) 類似画像検索サーバ"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# 目次

* [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)
* [類似画像検索ツールを作ってみる (2) 特徴化 その1](202105-similar-search-2)
* [類似画像検索ツールを作ってみる (3) 特徴化 その2](202105-similar-search-3)
* [類似画像検索ツールを作ってみる (4) 類似画像検索](202105-similar-search-4)
* 類似画像検索ツールを作ってみる (5) 類似画像検索サーバ（本記事）

# 類似画像検索サーバ

[前回](202105-similar-search-4)、類似画像検索を実装して、それっぽい検索結果を得ることができました。
ただ、目標とした1秒を下回ることができず、約10万枚の画像からの検索に1.5秒を要しました。

今回は検索の高速化を目的に、類似画像検索サーバを実装したいと思います。

# Flaskを使ってウェブアプリケーション化

前回実装した検索スクリプト`search.py`では、Pythonスクリプトの起動の度に、ONNXモデル、特徴量ファイルを読み込み、初期化していました。
これらのファイルの内容はほとんど変化しないため、読み込み、初期化を1度だけ行うことで高速化を図ってみます。

今回、ウェブアプリケーションフレームワークとしては「Flask」を使用しました。
また、別のウェブアプリケーションから利用することも考慮してCORSに関する指定を行っています。

類似画像検索サーバのコードは以下の通りです。空行を含めても100行ちょっととシンプルですね。Flask様々です。
処理の内容は前回紹介したスクリプトと同じなので、詳細は割愛します。

```py:app.py
import datetime
import os
import pathlib

import flask
import flask_cors
import numpy as np
import onnxruntime
import PIL.Image

MEDIA_DIR = pathlib.Path(os.environ["MEDIA_DIR"])
OBJECT_DIR = MEDIA_DIR / "object"
FEATURE_DIR = pathlib.Path(os.environ["FEATURE_DIR"])
ONNX_MODEL_PATH = pathlib.Path(os.environ["ONNX_MODEL_PATH"])


def make_nested_id_path(dir, id, ext=""):
    return dir / id[0:2] / id[2:4] / (id + ext)


onnx_session = onnxruntime.InferenceSession(str(ONNX_MODEL_PATH))

object_ids_features_pairs = []
for index in range(10):
    object_ids_path = FEATURE_DIR / "{:04d}.object_ids.npy".format(index)
    features_path = FEATURE_DIR / "{:04d}.features.npy".format(index)
    if not object_ids_path.exists():
        break
    object_ids = np.load(object_ids_path)
    features = np.load(features_path)
    assert len(object_ids) == len(features)
    object_ids_features_pairs.append((object_ids, features))

app = flask.Flask(__name__)
flask_cors.CORS(app)


@app.route("/similar/<string:query_object_id>")
def similar(query_object_id):
    image_path = make_nested_id_path(OBJECT_DIR, query_object_id)

    image = PIL.Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255

    query_feature = onnx_session.run(
        ["feature_vector"], {"inputs:0": np.expand_dims(image, 0)}
    )[0][0]

    results = []
    limit = 100
    for object_ids, features in object_ids_features_pairs:
        query_features = np.tile(query_feature, (len(features), 1))
        distances = np.linalg.norm(query_features - features, axis=1)
        distance_indexes = np.argsort(distances)[:limit]
        results.extend(zip(object_ids[distance_indexes], distances[distance_indexes]))

    results = sorted(results, key=lambda item: item[1])[:limit]

    return flask.jsonify(
        {
            "time": int(datetime.datetime.now().timestamp() * 1000),
            "queryObjectId": query_object_id,
            "similarImages": [
                {"objectId": object_id, "distance": float(distance),}
                for object_id, distance in results
            ],
        }
    )
```

`Dockerfile`、`requirements.txt`は以下の通りです。

```Dockerfile:Dockerfile
FROM ubuntu:20.04
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    build-essential \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-setuptools \
  && rm --recursive --force /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools
WORKDIR /opt/app/
COPY requirements.txt ./
RUN python3 -m pip install --requirement requirements.txt
COPY src/ ./src/
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
EXPOSE 8080
CMD ["uwsgi", "--wsgi-file=src/app.py", "--callable=app", "--http=:8080", "--wsgi-disable-file-wrapper"]
```

```
Flask-Cors==3.0.10
Flask==2.0.1
onnxruntime==1.7.0
Pillow==8.2.0
uWSGI==2.0.19.1
```

# 検索してみる

では実際に、約10万枚分の特徴量から検索してみましょう。今回はHTTPクライアントに`curl`を使用しました。

```
$ time curl http://localhost:8080/similar/5577b06378df4cbf5fa04237ac767205a944a360.jpg
{
  "queryObjectId": "5577b06378df4cbf5fa04237ac767205a944a360.jpg",
  "similarImages": [
    {
      "distance": 2.5465216822340153e-05,
      "objectId": "5577b06378df4cbf5fa04237ac767205a944a360.jpg"
    },
...
    {
      "distance": 13.204506874084473,
      "objectId": "6af0302039125508ffbc0302a7cd16df4922d26f.jpg"
    }
  ],
  "time": 1622477510480
}

real    0m0.286s
user    0m0.003s
sys     0m0.005s
```

約0.3秒で検索できました。目標達成です！

ちなみに約60万枚の画像データセット全体で検索したところ、約1.5秒掛かりました。60万枚で1.5秒なら、まあ実用的な範囲かと思います。個人的には。

# 今日はここまで

今回は類似検索処理をサーバ化し、目標の1秒を下回る約0.3秒で10万枚の画像から類似画像を検索することができました。
ただ、このままでは結果を確認しづらいので、次回は検索結果の可視化にチャレンジしてみたいと思います。今日はここまで！
