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


[前回](202103-pornography-classifier-6)の記事から随分と間が開いてしまいました。
今回は「ラベル付け」について書くつもりでしたが、そろそろ前置きが長くなって飽きてきた読者がいそうなので、今回は「学習」について書きたいと思います。

今回、画像分類のモデルとして「EfficientNet B0」を使用することにしました。
EfficientNetは2019年5月にGoogleの研究者が発表したモデルで、比較的少ないパラメータ数でよい精度を得られるらしいです。
また、EfficientNetにはB0からB7までのバリエーションがありますが、一番コンパクトならB0を使います。



# 参考

* [EfficientNet Explained | Papers With Code](https://paperswithcode.com/method/efficientnet)
* [2019年最強の画像認識モデルEfficientNet解説 - Qiita](https://qiita.com/omiita/items/83643f78baabfa210ab1)
* [EfficientNetを最速で試す方法 - Qiita](https://qiita.com/wakame1367/items/d90fa56bd9d11c4db50e)
