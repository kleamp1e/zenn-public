---
title: "けしからん画像分類器を作ってみる (8) 学習 その2"
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
* [けしからん画像分類器を作ってみる (7) 学習 その1](202104-pornography-classifier-7)
* けしからん画像分類器を作ってみる (8) 学習 その2（本記事）

# 続 学習

[前回](202104-pornography-classifier-7)はデータの分割までを行いました。
今回は実際に学習を行ってみたいと思います。

# 結論

最初に結論を書いておくと、精度（Accuracy）が84%程度の画像分類モデルを得ることができました。
今回はほとんど工夫はしておらず、かなりシンプルな構成です。様々な手法を適用することで、さらに数%は精度を高められそうな気がしています。

# データの前処理

Kerasを使ってモデルを学習するにあたり、データの読み込みに`tf.keras.preprocessing.image.ImageDataGenerator`を使用しました。（手抜きのために）

`ImageDataGenerator`は特定のディレクトリ構造が前提となっていますので、既存の画像データをその構造に配置します。
ファイルをコピーするのは無駄なので、今回はシンボリックリンクを作成しています。

```py:make_link.py
#!/usr/bin/env python3

import os
import pathlib

import pandas as pd

def make_nested_id_path(dir, id, ext=""):
    return dir / id[0:2] / id[2:4] / (id + ext)

MEDIA_DIR = pathlib.Path(os.environ.get("MEDIA_DIR"))
OBJECT_DIR = MEDIA_DIR / "object"
CACHE_DIR = pathlib.Path("/tmp/cache/pornography")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("split_label.csv")

for type in ["train", "test", "validation"]:
  for index, (object_id, value) in df[["objectId", "value"]][df["type"] == type].iterrows():
      image_path = make_nested_id_path(OBJECT_DIR, object_id)
      target_path = CACHE_DIR / type / str(value) / object_id
      target_path.parent.mkdir(parents=True, exist_ok=True)
      if not target_path.exists():
          target_path.symlink_to(image_path)
```


# 学習


# 参考
