---
title: "けしからん画像分類器を作ってみる (10) 公開"
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
* [けしからん画像分類器を作ってみる (8) 学習 その2](202104-pornography-classifier-8)
* [けしからん画像分類器を作ってみる (9) 推論](202104-pornography-classifier-9)
* けしからん画像分類器を作ってみる (10) 公開（本記事）
* 番外編:
    * [EfficientNet B0〜B7で画像分類器を転移学習してみる](202104-efficientnet)
    * [EfficientNet B0のKerasモデルをONNXモデルに変換して推論する](202104-keras-onnx)
    * [類似画像検索ツールを作ってみる (1) 序章](202105-similar-search-1)

# 「けしからん画像分類器」を公開しました

[前回の記事](202104-pornography-classifier-9)からかなりの時間が空いてしまいましたが、Twitterでモデル公開のリクエストを頂いたので久しぶりに再学習し、モデルを公開しました。

https://twitter.com/discord_nana/status/1587419833653932032

ONNXモデル、サンプルコード、ライセンス情報を含むZIPファイルは、以下からダウンロードできます。
なお、すべてのファイルを含むZIPファイルのままであれば、再配布して頂いて構いません。

* URL: https://drive.google.com/file/d/1y1tHJ8oxJTWUOYrgXDIdEhBlMIynwVzt/view?usp=share_link
* ファイル名: `kleamp1e-classifier-pornography-20221113_164725.zip`
* ファイルサイズ: `74974610`
* SHA-1ハッシュ値: `6ccf0c61b44af5673cecd1140093f6296c734576`

ONNXモデル、サンプルコードはMITライセンスとしています。機械学習モデルの公開に際して、より適切なライセンスがあれば情報を頂けると幸いです。

# 実行例

ZIPファイルにはONNXモデルの他、推論のサンプルコード`predict.py`が含まれています。実行例を以下に示します。

```sh
pip3 install -r requirements.txt
python3 predict.py foo.jpg
```

`predict.py`の内容は以下の通りです。

```py:predict.py
#!/usr/bin/env python3

import os
import sys

import numpy as np
import onnxruntime
import PIL.Image

ONNX_MODEL_FILE = os.path.join(os.path.dirname(__file__), "kleamp1e-classifier-pornography-20221113_164725.onnx")
TARGET_SIZE = 384


def main():
    image_path = sys.argv[1]

    session = onnxruntime.InferenceSession(
        ONNX_MODEL_FILE, providers=["CPUExecutionProvider"]
    )
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image = PIL.Image.open(image_path)
    image = image.resize(
        (TARGET_SIZE, TARGET_SIZE), resample=PIL.Image.Resampling.BICUBIC
    )
    image = image.convert("RGB")
    image = np.array(image, dtype=np.float32) / 255
    image = np.expand_dims(image, 0)

    predictions = session.run([output_name], {input_name: image})
    print(predictions[0][0][0])


if __name__ == "__main__":
    main()
```

# モデルについて

## ネットワーク構造

以前の記事では、バックボーンネットワークとして「EfficientNet B0」を使用していましたが、今回のモデルは「EfficientNet V2 S」を使用しています。

## データ数

テストデータ、バリデーションデータ、学習データの内訳は以下の通りです。

| タイプ | ラベル:0 | ラベル1: | 合計 |
|:---|---:|---:|---:|
| test | 1002 | 1002 | 2004 |
| validation | 1036 | 1036 | 2072 |
| train | 8343 | 13326 | 21669 |
| 合計 | 10381 | 15364 | 25745 |

## 混同行列

テストデータを推論した結果の混同行列（Confusion Matrix）は以下の通りです。

|         | 推論結果:0 | 推論結果:1 |
|:---|---:|---:|
| ラベル:0 | 882 | 120 |
| ラベル:1 |  60 | 942 |

（TODO: 混同行列）
（TODO: 各種メトリクス）
（TODO: ROC曲線）
