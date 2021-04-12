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
    * TensorFlow: 2.4.1

# フレームワーク

今回は機械学習フレームワークとして、TensorFlowに内蔵されている高レベルライブラリの「Keras」（[Wikipedia](https://ja.wikipedia.org/wiki/Keras)）を使用しました。
EfficientNetの学習済みモデルが提供されていたのが主な理由です。

# モデル

今回は画像分類のモデルとして「EfficientNet B0」を使用することにしました。
EfficientNetは2019年5月にGoogleの研究者が発表したモデルで、比較的少ないパラメータ数でよい精度を得られるらしいです。
また、EfficientNetにはB0からB7までのバリエーションがありますが、一番コンパクトならB0を使います。

具体的には、[TensorFlow Hub](https://tfhub.dev/google/collections/efficientnet)で提供されているEfficientNet B0の学習済みモデルを使用しました。この学習済みモデルはImageNetで事前学習されています。

今回はファインチューニング（Fine Tuning）は行わず、転移学習（Transfer Learning）のみを行いました。ファインチューニングには別の機会にトライしてみたいです。

# データの概要


# データの前処理


# 学習


# 参考

* [EfficientNet Explained | Papers With Code](https://paperswithcode.com/method/efficientnet)
* [2019年最強の画像認識モデルEfficientNet解説 - Qiita](https://qiita.com/omiita/items/83643f78baabfa210ab1)
* [EfficientNetを最速で試す方法 - Qiita](https://qiita.com/wakame1367/items/d90fa56bd9d11c4db50e)
