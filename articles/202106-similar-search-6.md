---
title: "類似画像検索ツールを作ってみる (6) 可視化"
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
* [類似画像検索ツールを作ってみる (5) 類似画像検索サーバ](202105-similar-search-5)
* 類似画像検索ツールを作ってみる (6) 可視化（本記事）

# 可視化

[前回](202105-similar-search-5)、類似画像検索サーバを実装し、ウェブAPI経由で類似画像を検索できるようになりました。

ただ、画像のオブジェクトIDと距離が出力されるだけでは味気なく、画像の距離感を感じることができません。
そこで今回は、類似画像検索の結果を可視化してみたいと思います。

# 可視化…その前に

今回、インデックスに含まれる画像は「けしからん画像」ばかりです。
そのため、スクリーンショットをそのまま掲載すると何らかの規約に抵触し、記事が削除される、あるいはアカウントごとバンされかねません。
そのため、今回はわざわざ「画像にモザイクを掛けて配信するサーバ」を実装しました。

特に解説しませんが、コードは以下の通りです。

```py:censored-distributor/src/app.py
import io
import os
import pathlib

import flask
import flask_cors
import PIL.Image

MEDIA_DIR = pathlib.Path(os.environ["MEDIA_DIR"])
OBJECT_DIR = MEDIA_DIR / "object"


def make_nested_id_path(dir, id, ext=""):
    return dir / id[0:2] / id[2:4] / (id + ext)


app = flask.Flask(__name__)
flask_cors.CORS(app)


@app.route("/<string:object_id>")
def censored(object_id):
    image_path = make_nested_id_path(OBJECT_DIR, object_id)

    image = PIL.Image.open(image_path).convert("RGB")
    width, height = image.width, image.height
    block_size = 64
    image = image.resize((width // block_size, height // block_size), PIL.Image.NEAREST)
    image = image.resize((width, height), PIL.Image.NEAREST)

    bio = io.BytesIO()
    image.save(bio, format="JPEG")

    response = flask.make_response(bio.getvalue())
    response.headers.set("Content-Type", "image/jpeg")

    return response
```
