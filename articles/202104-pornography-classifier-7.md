---
title: "けしからん画像分類器を作ってみる (6) 学習 その1"
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
* けしからん画像分類器を作ってみる (6) 学習 その1（本記事）

# 今回の内容

[前回](202103-pornography-classifier-6)の記事から随分と間が開いてしまいました。
今回は「ラベル付け」について書くつもりでしたが、そろそろ前置きが長くなって飽きてきた読者がいそうなので、今回は「学習」について書きたいと思います。

# 結論

最初に結論を書いておくと、精度（Accuracy）が84%程度のモデルを得ることができました。
今回はほとんど工夫はしておらず、かなりシンプルな構成です。様々な手法を適用することで、さらに数%は精度を高められそうな気がしています。

# 環境

学習では、以下の環境を使用しました。学習はDocker内で行っているので、GPUさえ認識できればDockerホスト側の環境はあまり関係ないと思います。

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

今回は機械学習フレームワークとして、TensorFlowに内蔵されている高レベルライブラリの「Keras」（[Wikipedia](https://ja.wikipedia.org/wiki/Keras)）を使用しました。
EfficientNetの学習済みモデルが提供されていたのが主な理由です。

# モデル

今回は画像分類のモデルとして「EfficientNet B0」を使用することにしました。
EfficientNetは2019年5月にGoogleの研究者が発表したモデルで、比較的少ないパラメータ数でよい精度を得られるらしいです。
また、EfficientNetにはB0からB7までのバリエーションがありますが、一番コンパクトならB0を使います。

具体的には、[TensorFlow Hub](https://tfhub.dev/google/collections/efficientnet)で提供されているEfficientNet B0の学習済みモデル（Pretrained Model）を使用しました。この学習済みモデルはImageNetで事前学習されています。

今回はファインチューニング（Fine Tuning）は行わず、転移学習（Transfer Learning）のみを行いました。ファインチューニングには別の機会にトライしてみたいです。

**参考:**

* [2019年最強の画像認識モデルEfficientNet解説 - Qiita](https://qiita.com/omiita/items/83643f78baabfa210ab1)
* [EfficientNet Explained | Papers With Code](https://paperswithcode.com/method/efficientnet)
* [EfficientNetを最速で試す方法 - Qiita](https://qiita.com/wakame1367/items/d90fa56bd9d11c4db50e)

# データの概要

今回はラベル付きの画像データ22,640枚を使用しました。内訳は以下の通りです。

| 内容 | 枚数 | ラベル |
|:---|---:|:---:|
| けしからんくない画像 | 7,617枚 | `0` |
| けしからん画像 | 15,023枚 | `1` |

世の中には「けしからん画像」ばかりなので、いわゆる「不均衡データ」（Imbalanced Data）になっています。ちなみに今回は不均衡さは考慮していません。今後の課題ということで。

データの具体例を以下に示します。先頭行はヘッダ行で、以降の行はオブジェクトID（過去の記事を参照）とラベルで構成されています。

```text
$ wc -l /mnt/data/label.csv
22641 /mnt/data/label.csv

$ head -n 5 /mnt/data/label.csv
objectId,value
00002562b453e832e61233eadc2f883a22ad3853.jpg,1
00005166b19153cb07cdf6d4ec5cd1980470483a.jpg,1
00007055678527787afd8bc8f26b6f6def4656a3.jpg,1
00007d225637a27040fee6408184ee4621a2264e.jpg,0
```

# データの分割

上記のデータを学習データ（Training Data、訓練データとも）80%、検証データ（Validation Data）10%、テストデータ（Test Data）10%に分割して使用しました。
可能な限り分割先を維持するために、オブジェクトID（SHA-1ハッシュ値）に基づいて分割を行いました。

具体的には、SHA-1ハッシュ値は160ビット（20バイト）ありますが、その下位1バイト（256種類）を使ってデータを分割しました。
下位1バイトが0〜24をテストデータ、25〜49を検証用データ、残りを学習データとしました。
また、テストデータ、検証データによる評価を容易にするために、それぞれラベルの少ない方に数を合わせ、余ったデータは学習データとしました。

言葉で説明するよりコードを見て頂く方が確実ですね。データを分割するPythonスクリプトは以下の通りです。

```py
#!/usr/bin/env python3

import re

import pandas as pd

df = pd.read_csv("/mnt/data/label.csv")
print("df:", len(df))

value_0 = df["value"] == 0
value_1 = df["value"] == 1
print("df.0:", len(df[value_0]))
print("df.1:", len(df[value_1]))

df["type"] = "train"

split_key = df["objectId"].apply(
    lambda object_id: int(re.sub("\\..+$", "", object_id)[-2:], 16)
)
df.loc[(split_key >= 0) & (split_key <= 24), "type"] = "test"
df.loc[(split_key >= 25) & (split_key <= 49), "type"] = "validation"

test_0 = (df["type"] == "test") & value_0
test_1 = (df["type"] == "test") & value_1
test_min = min(len(df[test_0]), len(df[test_1]))
print("test_min:", test_min)
df.loc[df[test_0].index[test_min:], "type"] = "train"
df.loc[df[test_1].index[test_min:], "type"] = "train"

validation_0 = (df["type"] == "validation") & value_0
validation_1 = (df["type"] == "validation") & value_1
validation_min = min(len(df[validation_0]), len(df[validation_1]))
print("validation_min:", validation_min)
df.loc[df[validation_0].index[validation_min:], "type"] = "train"
df.loc[df[validation_1].index[validation_min:], "type"] = "train"

print("train.0:", len(df[(df["type"] == "train") & value_0]))
print("train.1:", len(df[(df["type"] == "train") & value_1]))
print("test.0:", len(df[(df["type"] == "test") & value_0]))
print("test.1:", len(df[(df["type"] == "test") & value_1]))
print("validation.0:", len(df[(df["type"] == "validation") & value_0]))
print("validation.1:", len(df[(df["type"] == "validation") & value_1]))

df.to_csv("split_label.csv", index=False)
```

実行結果は以下の通りです。

```text
$ ./split.py
df: 22640
df.0: 7617
df.1: 15023
test_min: 744
validation_min: 747
train.0: 6126
train.1: 13532
test.0: 744
test.1: 744
validation.0: 747
validation.1: 747
```

表にすると以下の通りです。

| 種別 | ラベル`0` | ラベル`1` | 合計 |
|:---|---:|---:|---:|
| 学習データ | 6,126 | 13,532 | 19,658 |
| 検証データ | 747 | 747 | 1,494 |
| テストデータ | 744 | 744 | 1,488 |
| 合計 | 7,617 | 15,023 | 22,640 |

**参考:**

* [ML Design Pattern #5: Repeatable sampling | by Lak Lakshmanan | Towards Data Science](https://towardsdatascience.com/ml-design-pattern-5-repeatable-sampling-c0ccb2889f39)

# データの前処理


# 学習


# 参考
