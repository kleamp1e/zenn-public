---
title: "類似画像検索ツールを作ってみる (6) Next.js + SVGで可視化"
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
* 類似画像検索ツールを作ってみる (6) Next.js + SVGで可視化（本記事）

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

# Next.js + SVGで可視化

今回は、Next.jsを用いて可視化ツールを実装しました。また、描画部分にはSVG（Scalable Vector Graphics）を用いています。

類似画像検索の結果を、距離に応じて同心円状に描画してみました。（実行例は巻末を参照）
中央はクエリ画像で、中央に近いほど「類似度が高い画像」、遠ざかるほど「類似度が低い画像」となっています。
また、画像をクリックすることでその画像をクエリ画像として再検索し、画像群を探索できるようになっています。

今回は距離を5乗し、正規化することで表示位置を調整しています。また、配置角度はオブジェクトIDからテキトーに求めています。
本当は画像が重ならないように良い感じに配置したかったのですが、今回は力尽きました。

主要なコードは以下の通りです。

```js:similar-browser/pages/object/[objectId].js
import Head from "next/head";
import Router from "next/router";
import { useState } from "react";

function makeImageUrlFromObjectId(objectId) {
  return "http://localhost:20583/" + objectId;
}

function CircleImage({ cx, cy, dx, dy, objectId, onMouseEnter }) {
  const imageUrl = makeImageUrlFromObjectId(objectId);
  const pageUrl = "/object/" + objectId;
  return (
    <image
        x={cx - (dx / 2)}
        y={cy - (dy / 2)}
        width={dx}
        height={dy}
        href={imageUrl}
        style={{cursor: "pointer"}}
        onClick={() => Router.push(pageUrl)}
        onMouseEnter={onMouseEnter} />
  );
}

function Circle({ width, height, minRadius, maxRadius, queryImage, images }) {
  const cx = width / 2;
  const cy = height / 2;
  const deltaRadius = maxRadius - minRadius;
  const positionedImages = images.map((image) => ({
    cx: cx + Math.cos(image.angle) * (minRadius + image.distance * deltaRadius),
    cy: cy + Math.sin(image.angle) * (minRadius + image.distance * deltaRadius),
    ...image,
  }));
  positionedImages.sort((a, b) => b.distance - a.distance);

  const numOfCircles = 5;
  const radiuses = Array.from({length: numOfCircles}, (v, k) => k).map((i) => (
    minRadius + (deltaRadius / (numOfCircles - 1)) * i
  ));

  const [selectedImage, setSelectedImage] = useState(null);
  return (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        version="1.1"
        viewBox={`0 0 ${width} ${height}`}>
      <rect
          x={0}
          y={0}
          width={width}
          height={height}
          stroke="none"
          fill="#E0E0E0" />
      <image
          x={0}
          y={0}
          width={300}
          height={300}
          href={makeImageUrlFromObjectId(queryImage.objectId)} />
      {selectedImage == null ? null :
        <image
            x={width - 300}
            y={0}
            width={300}
            height={300}
            href={makeImageUrlFromObjectId(selectedImage.objectId)} />
      }
      <circle
          cx={cx}
          cy={cy}
          r={50}
          stroke="none"
          fill="#CCCCCC" />
      {radiuses.map((radius) => (
        <circle
            key={radius}
            cx={cx}
            cy={cy}
            r={radius}
            stroke="#CCCCCC"
            strokeWidth="1px"
            fill="none" />
      ))}
      {positionedImages.map((image) => (
        <CircleImage
            key={image.objectId}
            cx={image.cx}
            cy={image.cy}
            dx={80}
            dy={80}
            objectId={image.objectId}
            onMouseEnter={() => setSelectedImage(image)} />
      ))}
      <CircleImage
          cx={cx}
          cy={cy}
          dx={100}
          dy={100}
          objectId={queryImage.objectId} />
    </svg>
  );
}

const transformDistance = (distance) => Math.pow(distance, 5);
const calcAngle = (objectId) => (parseInt(objectId.substring(0, 4), 16) / 0xFFFF) * (Math.PI * 2);

export default function Page({ objectId, data }) {
  const similarImages = data.similarImages.filter((image) => image.objectId != objectId);
  const distances = similarImages.map((image) => transformDistance(image.distance));
  const minDistance = Math.min(...distances);
  const maxDistance = Math.max(...distances);
  const deltaDistance = maxDistance - minDistance;
  const normalizeDistance = (distance) => (transformDistance(distance) - minDistance) / deltaDistance;

  const images = similarImages.map((image) => ({
    objectId: image.objectId,
    distance: normalizeDistance(image.distance),
    angle: calcAngle(image.objectId),
  }));

  return (
    <>
      <Head>
        <title>{objectId}</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div>objectId: <code>{objectId}</code></div>
      <Circle
          width={1000}
          height={1000}
          minRadius={100}
          maxRadius={450}
          queryImage={{objectId}}
          images={images} />
    </>
  );
}

export async function getServerSideProps(context) {
  const { objectId } = context.query;
  const url = "http://localhost:25160/similar/" + objectId;
  const response = await fetch(url);
  const data = await response.json()

  return {
    props: {
      objectId,
      data,
    },
  };
}
```

# 実行例

スクール水着っぽいクエリ画像を指定した場合の実行例を以下に示します。
諸般の事情で強烈にモザイクを掛けていますが、何となく「水着っぽい」画像が検索できていることが分かるかと思います。

ちなみに左上にはクエリ画像が、右上にはカーソル位置の画像が拡大して表示されています。

![](https://storage.googleapis.com/zenn-user-upload/6102d012c4493b6813d38b61.png)

# このシリーズはここまで

本記事まで6本に渡り「類似画像検索」に関して記事を書いてみました。
当初の目標も実現でき、記事を書く過程で色々と勉強になりました。やっぱりアウトプットは重要ですね。

このシリーズはここまでです。お付き合い頂きありがとうございました！
